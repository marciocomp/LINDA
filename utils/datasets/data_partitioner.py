# utils/datasets/data_partitioner.py
# @author: Marcio Lopes
import os
import numpy as np
import random
from utils.linda_logger import logger


class DataPartitioner:
    """
    Utility to split datasets into partitions for Federated/Split Learning.
    Supports:
      1) Global reduction (new_size) -> Universe U
      2) Global fixed eval split (E) from Universe U
      3) Per-client partition from TrainPool T = U \ E
      4) Support for both LLM (Alpaca) and Vision (CIFAR/ImageNet)
    """

    # ----------------------------
    # Helpers (load + sanitize)
    # ----------------------------
    @staticmethod
    def _expand_cache_dir(cache_dir):
        if cache_dir is None:
            return None
        return os.path.expanduser(cache_dir)

    @staticmethod
    def _load_hf_dataset(dataset_name: str, cache_dir=None):
        from datasets import load_dataset
        cache_dir = DataPartitioner._expand_cache_dir(cache_dir)
        logger.info(f"Partitioner: Loading HF dataset={dataset_name} split=train cache_dir={cache_dir} ...")
        ds = load_dataset(dataset_name, split="train", cache_dir=cache_dir)
        return ds

    @staticmethod
    def _coerce_new_size(new_size, total_len: int):
        """
        new_size:
          - None or <=0 -> full
          - float (0,1] -> percentage of total
          - int >=1 -> absolute count
        """
        if new_size is None:
            return total_len

        try:
            if isinstance(new_size, float):
                if new_size <= 0:
                    return total_len
                if new_size <= 1.0:
                    return max(1, int(total_len * new_size))
                # float > 1.0: treat as absolute count
                return min(total_len, int(new_size))

            # int/other numeric
            new_size_int = int(new_size)
            if new_size_int <= 0:
                return total_len
            return min(total_len, new_size_int)

        except Exception:
            logger.warning(f"[DATA] Invalid new_size={new_size} -> using full dataset.")
            return total_len

    # ----------------------------------------
    # NEW (1): global reduced universe indices
    # ----------------------------------------
    @staticmethod
    def get_reduced_universe_indices(dataset_name="tatsu-lab/alpaca",
                                     new_size=0.1,
                                     seed=42,
                                     cache_dir=None):
        """
        Returns 'U' = globally reduced universe (indices of the original dataset).
        Deterministic by seed.
        """
        np.random.seed(seed)
        random.seed(seed)

        if "alpaca" in dataset_name.lower():
            ds = DataPartitioner._load_hf_dataset(dataset_name, cache_dir=cache_dir)
            num_samples = len(ds)
        elif dataset_name == "CIFAR10":
            from torchvision import datasets
            temp_ds = datasets.CIFAR10(root=cache_dir if cache_dir else "./data", train=True, download=True)
            num_samples = len(temp_ds)
        else:
            # ImageNet and others: return None so the Dataset class handles it
            return None

        universe_len = DataPartitioner._coerce_new_size(new_size, num_samples)
        all_indices = np.arange(num_samples, dtype=np.int64)

        if universe_len >= num_samples:
            logger.info(f"[DATA] Universe uses FULL dataset: {num_samples} samples.")
            return all_indices.tolist()

        rng = np.random.RandomState(seed)
        universe = rng.choice(all_indices, size=universe_len, replace=False)

        logger.warning(
            f"[DATA] Universe reduced by new_size={new_size}: {num_samples} -> {len(universe)} samples (seed={seed})."
        )
        return universe.tolist()

    # ----------------------------------------------------
    # NEW (2): split global eval indices from the universe
    # ----------------------------------------------------
    @staticmethod
    def split_global_eval_from_universe(universe_indices,
                                        eval_fraction=0.10,
                                        seed=42,
                                        eval_seed=None):
        """
        Takes U and returns:
          E = eval set (fixed for all clients)
          T = U \\ E  (train pool, no leakage)
        """
        if universe_indices is None:
            return [], []

        if not universe_indices:
            raise ValueError("Universe indices is empty.")

        u = np.array(universe_indices, dtype=np.int64)
        u_len = len(u)

        if eval_fraction <= 0:
            logger.warning("[DATA] eval_fraction<=0 -> eval set empty (not recommended).")
            return [], u.tolist()

        n_eval = int(u_len * float(eval_fraction))
        n_eval = max(1, n_eval)

        if eval_seed is None:
            eval_seed = seed + 1337

        rng = np.random.RandomState(eval_seed)
        eval_idx = rng.choice(u, size=n_eval, replace=False)

        train_pool = np.setdiff1d(u, eval_idx, assume_unique=False)

        logger.info(
            f"[DATA] Global eval split: Universe={u_len} | Eval={len(eval_idx)} ({eval_fraction:.0%}) | "
            f"TrainPool={len(train_pool)} | seed={seed}"
        )

        return eval_idx.tolist(), train_pool.tolist()

    # -----------------------------------------------------
    # NEW (3): convenience builder for NodeAgent
    # -----------------------------------------------------
    @staticmethod
    def get_global_eval_and_train_pool(dataset_name="tatsu-lab/alpaca",
                                       new_size=0.1,
                                       eval_fraction=0.10,
                                       seed=42,
                                       cache_dir=None):

        # Only used for Alpaca/LLM or datasets that fit in memory
        universe = DataPartitioner.get_reduced_universe_indices(
            dataset_name=dataset_name,
            new_size=new_size,
            seed=seed,
            cache_dir=cache_dir
        )

        if universe is None:
            # ImageNet/Others: return empty so the Dataset class handles it
            return [], [], []

        eval_idx, train_pool = DataPartitioner.split_global_eval_from_universe(
            universe_indices=universe,
            eval_fraction=eval_fraction,
            seed=seed
        )
        return eval_idx, train_pool, universe

    # ----------------------------------------
    # UPDATED: per-client partition (Main Logic)
    # ----------------------------------------
    @staticmethod
    def get_partition_indices(dataset_name="tatsu-lab/alpaca",
                              total_clients=3,
                              partition_id=0,
                              method="content_non_iid",
                              alpha=0.5,
                              seed=42,
                              cache_dir=None,
                              pool_indices=None,
                              existing_targets=None):
        """
        Returns training indices for a specific client.
        Supports Alpaca (text) and Vision (CIFAR/ImageNet).
        """
        if partition_id < 0 or partition_id >= total_clients:
            raise ValueError(f"partition_id={partition_id} out of range for total_clients={total_clients}")

        np.random.seed(seed)
        random.seed(seed)

        # =========================================================================
        # BRANCH 1: LLM (Alpaca) - Instruction-based partitioning
        # =========================================================================
        if "alpaca" in dataset_name.lower():
            ds = DataPartitioner._load_hf_dataset(dataset_name, cache_dir=cache_dir)

            if pool_indices is None:
                pool = np.arange(len(ds), dtype=np.int64)
                logger.warning("[DATA] Alpaca: pool_indices=None -> using FULL dataset.")
            else:
                pool = np.array(pool_indices, dtype=np.int64)

            logger.info(f"Partitioner (LLM): {method}, clients={total_clients}, alpha={alpha}")

            if method == "content_non_iid":
                # Sort lexicographically by instruction text
                instructions = np.array(ds["instruction"], dtype=object)
                pool_instr = instructions[pool]
                order = np.argsort(pool_instr)
                sorted_pool = pool[order]
                splits = np.array_split(sorted_pool, total_clients)
                return splits[partition_id].tolist()

            elif method == "iid":
                rng = np.random.RandomState(seed)
                shuffled = pool.copy()
                rng.shuffle(shuffled)
                splits = np.array_split(shuffled, total_clients)
                return splits[partition_id].tolist()

            elif method == "quantity_skew":
                rng = np.random.RandomState(seed)
                shuffled = pool.copy()
                rng.shuffle(shuffled)
                min_size = 0
                pool_len = len(shuffled)
                while min_size < 1:
                    proportions = rng.dirichlet(np.repeat(alpha, total_clients))
                    min_size = int(np.min(proportions * pool_len))
                split_points = (np.cumsum(proportions) * pool_len).astype(int)[:-1]
                splits = np.split(shuffled, split_points)
                return splits[partition_id].tolist()

            elif method == "content_dirichlet_balanced":
                # ---------------------------------------------------------
                # LLM non-IID controlled by alpha (Dirichlet) + EXACT size
                # using pseudo-classes (buckets) based on instruction text.
                # ---------------------------------------------------------
                if pool_indices is None:
                    pool = np.arange(len(ds), dtype=np.int64)
                    logger.warning("[DATA] Alpaca: pool_indices=None -> using FULL dataset.")
                else:
                    pool = np.array(pool_indices, dtype=np.int64)

                K = int(total_clients)
                if K <= 0:
                    raise ValueError("total_clients must be > 0")

                per_client = len(pool) // K
                if per_client <= 0:
                    raise ValueError(f"[DATA][LLM] per_client=0. pool={len(pool)} K={K}")

                # Discard remainder for exact partitioning
                pool = pool[: per_client * K]

                # Create deterministic buckets from instruction text
                B = 100
                instructions = np.array(ds["instruction"], dtype=object)
                pool_instr = instructions[pool]

                # Stable hash (python hash is not stable across processes)
                def fnv1a_64(s: str) -> int:
                    # FNV-1a 64-bit (explicit modular overflow)
                    h = 1469598103934665603
                    prime = 1099511628211
                    for b in s.encode("utf-8", errors="ignore"):
                        h ^= b
                        h = (h * prime) & 0xFFFFFFFFFFFFFFFF
                    return h

                bucket_ids = np.zeros(len(pool), dtype=np.int32)
                for i, txt in enumerate(pool_instr):
                    bucket_ids[i] = int(fnv1a_64(str(txt)) % np.uint64(B))

                # Indices per bucket (no replacement)
                idx_by_bucket = []
                for b in range(B):
                    idx_b = pool[bucket_ids == b]
                    idx_by_bucket.append(idx_b)

                rng = np.random.RandomState(seed)
                for b in range(B):
                    rng.shuffle(idx_by_bucket[b])

                # Dirichlet per client (K x B)
                alpha_eff = float(alpha) if alpha is not None else 0.5
                if alpha_eff <= 0:
                    alpha_eff = 1e-6
                probs = rng.dirichlet([alpha_eff] * B, size=K)

                ptr = np.zeros(B, dtype=np.int64)
                idx_batch = [[] for _ in range(K)]

                for j in range(K):
                    desired = probs[j] * per_client
                    base = np.floor(desired).astype(int)
                    remain = per_client - int(base.sum())
                    if remain > 0:
                        frac = desired - base
                        order = np.argsort(frac)[::-1]
                        for t in order[:remain]:
                            base[t] += 1

                    indices_j = []
                    deficit = 0

                    # Try to follow per-bucket counts
                    for b, cnt in enumerate(base):
                        if cnt <= 0:
                            continue
                        avail = len(idx_by_bucket[b]) - ptr[b]
                        take = min(cnt, avail)
                        if take > 0:
                            indices_j.extend(idx_by_bucket[b][ptr[b]:ptr[b] + take].tolist())
                            ptr[b] += take
                        deficit += (cnt - take)

                    # Fill deficit from available buckets (prefer high-probability ones)
                    if deficit > 0:
                        pref = np.argsort(probs[j])[::-1]
                        for b in pref:
                            if deficit <= 0:
                                break
                            avail = len(idx_by_bucket[b]) - ptr[b]
                            if avail <= 0:
                                continue
                            take = min(deficit, avail)
                            indices_j.extend(idx_by_bucket[b][ptr[b]:ptr[b] + take].tolist())
                            ptr[b] += take
                            deficit -= take

                    if len(indices_j) != per_client:
                        raise RuntimeError(
                            f"[DATA][LLM] Could not allocate per_client={per_client} for client {j}. "
                            f"Allocated={len(indices_j)}. Consider increasing pool size or buckets."
                        )

                    rng.shuffle(indices_j)
                    idx_batch[j] = indices_j

                return idx_batch[partition_id]

        # =========================================================================
        # BRANCH 2: VISION (CIFAR-10 / ImageNet) - Target/Class-based partitioning
        # =========================================================================
        else:
            # Load targets: either passed explicitly (ImageNet) or loaded (CIFAR)
            if existing_targets is not None:
                y_train = np.array(existing_targets)
                logger.info(f"Partitioner (Vision): Using existing_targets (len={len(y_train)})")
            elif dataset_name == "CIFAR10":
                from torchvision import datasets
                root = cache_dir if cache_dir else "./data"
                temp_ds = datasets.CIFAR10(root=root, train=True, download=True)
                y_train = np.array(temp_ds.targets)
            else:
                raise ValueError(f"Unknown dataset for partitioner: {dataset_name}")

            # If pool_indices provided (dataset reduction), filter y_train accordingly
            if pool_indices is not None:
                indices_to_use = np.array(pool_indices, dtype=np.int64)
                y_train = y_train[indices_to_use]
            else:
                indices_to_use = np.arange(len(y_train), dtype=np.int64)

            num_samples = len(indices_to_use)

            if method == "iid":
                rng = np.random.RandomState(seed)
                idxs = indices_to_use.copy()
                rng.shuffle(idxs)
                splits = np.array_split(idxs, total_clients)
                return splits[partition_id].tolist()

            elif method == "content_non_iid":  # Sort by Label (Class Non-IID)
                order = np.argsort(y_train)
                sorted_indices = indices_to_use[order]
                splits = np.array_split(sorted_indices, total_clients)
                return splits[partition_id].tolist()

            elif method == "dirichlet" or method == "quantity_skew":
                # Classic Dirichlet Non-IID (Label Distribution Skew)
                min_size = 0
                min_require_size = 10
                label_list = np.unique(y_train)
                num_classes = len(label_list)

                rng = np.random.RandomState(seed)

                client_idx_map = {i: [] for i in range(total_clients)}

                while min_size < min_require_size:
                    idx_batch = [[] for _ in range(total_clients)]
                    for k in label_list:
                        idx_k = np.where(y_train == k)[0]
                        idx_k = indices_to_use[idx_k]

                        rng.shuffle(idx_k)
                        proportions = rng.dirichlet(np.repeat(alpha, total_clients))

                        # Balance to avoid empty clients per class
                        proportions = np.array([p * (len(idx_j) < num_samples / total_clients) for p, idx_j in
                                                zip(proportions, idx_batch)])
                        proportions = proportions / proportions.sum()

                        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                        splits = np.split(idx_k, proportions)
                        for i in range(total_clients):
                            idx_batch[i].extend(splits[i])

                    min_size = min([len(idx_j) for idx_j in idx_batch])

                return idx_batch[partition_id]

            else:
                raise ValueError(f"Unknown partition method: {method}")