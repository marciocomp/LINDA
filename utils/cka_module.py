# @author: Marcio Lopes
import torch
import numpy as np
from sklearn.cluster import SpectralClustering
from utils.linda_logger import logger


class CKAEvaluator:
    """
    Gradient similarity computation using Linear CKA (Centered Kernel Alignment).
    Used in the Profiling phase to group clients with statistically similar data distributions.
    """

    @staticmethod
    def linear_CKA(g1, g2):
        """
        Computes Linear CKA between two flattened gradient vectors.
        For centered 1D vectors: CKA(x, y) ~ (x . y)^2 / ((x . x) * (y . y))

        Uses float64 to avoid NaN/Overflow in squared norm computation.
        """
        # float64 on CPU to prevent gradient^2 from overflowing float16/float32
        g1 = g1.double().cpu()
        g2 = g2.double().cpu()

        norm_g1 = torch.dot(g1, g1)
        norm_g2 = torch.dot(g2, g2)

        if norm_g1 == 0 or norm_g2 == 0:
            return 0.0

        dot_product = torch.dot(g1, g2)
        cka_score = (dot_product ** 2) / (norm_g1 * norm_g2)

        return cka_score.item()

    @staticmethod
    def compute_similarity_matrix(gradients_dict):
        """
        Generates the symmetric N x N similarity matrix.

        Args:
            gradients_dict: {node_id: torch.Tensor}

        Returns:
            matrix (np.array): Similarity matrix (0.0 to 1.0)
            node_ids (list): Sorted IDs mapping to matrix indices
        """
        node_ids = sorted(list(gradients_dict.keys()))
        n = len(node_ids)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    score = 1.0
                else:
                    g1 = gradients_dict[node_ids[i]]
                    g2 = gradients_dict[node_ids[j]]
                    score = CKAEvaluator.linear_CKA(g1, g2)

                # If math fails and produces NaN, assume dissimilarity
                if np.isnan(score):
                    score = 0.0

                matrix[i][j] = score
                matrix[j][i] = score

        return matrix, node_ids

    @staticmethod
    def cluster_clients_greedy(similarity_matrix, node_ids, threshold=0.9):
        """
        Greedy grouping based on similarity threshold.
        If sim(A, B) > threshold, they are placed in the same group.
        Does not guarantee a fixed number of groups.

        Returns:
            groups (dict): {group_id: [node_id_1, ...]}
        """
        groups = {}
        visited = set()
        group_counter = 0

        n = len(node_ids)

        for i in range(n):
            if i in visited: continue

            current_group = [node_ids[i]]
            visited.add(i)

            for j in range(i + 1, n):
                if j not in visited:
                    if similarity_matrix[i][j] >= threshold:
                        current_group.append(node_ids[j])
                        visited.add(j)

            groups[group_counter] = current_group
            group_counter += 1

        return groups

    @staticmethod
    def cluster_clients_spectral(similarity_matrix, node_ids, n_clusters=2, seed=42):
        """
        Groups clients using Spectral Clustering to enforce N optimal groups.
        Ideal for splitting clients into IID vs Non-IID or Strong vs Weak.

        Args:
            similarity_matrix: Precomputed CKA affinity matrix.
            node_ids: List of IDs mapped to matrix indices.
            n_clusters: Maximum desired number of groups.
        """
        num_clients = len(node_ids)

        real_n_clusters = min(n_clusters, num_clients)

        if real_n_clusters <= 1:
            logger.info(f"[CKA] Only {num_clients} client(s) or n_clusters=1. Returning single group.")
            return {0: node_ids}

        logger.info(f"[CKA] Clustering into {real_n_clusters} groups using Spectral Clustering...")

        try:
            # affinity='precomputed' since we already have the CKA matrix
            clustering = SpectralClustering(
                n_clusters=real_n_clusters,
                affinity='precomputed',
                random_state=seed,
                assign_labels='kmeans'
            )

            if np.any(np.isnan(similarity_matrix)):
                logger.warning("[CKA] NaN detected in matrix. Replacing with 0.0 for clustering.")
                similarity_matrix = np.nan_to_num(similarity_matrix, nan=0.0)

            labels = clustering.fit_predict(similarity_matrix)

            groups = {}
            for idx, label in enumerate(labels):
                label = int(label)
                if label not in groups:
                    groups[label] = []
                groups[label].append(node_ids[idx])

            return groups
            # return {0: node_ids}

        except Exception as e:
            logger.error(f"[CKA] Spectral Clustering failed: {e}. Fallback to single group.")
            return {0: node_ids}