# utils/datasets/cifar_10.py
# @author: Marcio Lopes
import torch
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, Subset, DataLoader
from PIL import Image
import os
from utils.linda_logger import logger


class Cifar10Dataset(Dataset):
    """
    Unified CIFAR-10 loader with Non-IID Dirichlet partitioning
    and dataset truncation support (new_size) for debugging.
    """

    def __init__(self,
                 root,
                 train=True,
                 client_id=None,
                 total_clients=None,
                 alpha=0.5,
                 seed=42,
                 download=True,
                 new_size=None):
        """
        Args:
            new_size (int/float, optional): Truncates the global dataset BEFORE partitioning.
        """
        self.root = root
        self.train = train

        # 1. Transforms
        if self.train:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        # 2. Load base dataset
        base_dataset = torchvision.datasets.CIFAR10(root=self.root, train=self.train, download=download)

        self.data = base_dataset.data
        self.targets = np.array(base_dataset.targets)

        # --- Truncation (NEW_SIZE) ---
        if new_size is not None and new_size > 0:
            total_len = len(self.data)
            truncate_to = total_len

            if isinstance(new_size, float) and new_size < 1.0:
                truncate_to = max(1, int(total_len * new_size))
            else:
                truncate_to = int(new_size)

            if truncate_to < total_len:
                logger.warning(
                    f"[CIFAR10] Truncating dataset from {total_len} to {truncate_to} samples. Seed={seed}."
                )
                np.random.seed(seed)
                indices = np.random.choice(total_len, truncate_to, replace=False)
                self.data = self.data[indices]
                self.targets = self.targets[indices]

        # 3. Partitioning (train only)
        if self.train:
            if client_id is None or total_clients is None:
                raise ValueError("For train=True, client_id and total_clients are required.")
            self._partition_data_balanced(client_id, total_clients, alpha, seed)
        else:
            logger.info(f"Loaded CIFAR-10 Global Test Set ({len(self.data)} samples).")

    def _partition_data_balanced(self, client_id, total_clients, alpha, seed):
        """
        Non-IID Dirichlet with EXACT size per client:
          - Each client gets exactly floor(N / K) samples (no replacement).
          - Remainder is discarded.
          - alpha controls class skew (smaller alpha => more concentrated).
        """
        rng = np.random.RandomState(seed)

        num_classes = 10
        n_samples_total = len(self.targets)
        per_client = n_samples_total // total_clients
        discarded = n_samples_total - per_client * total_clients

        if per_client <= 0:
            raise ValueError(f"[CIFAR10] per_client=0. N={n_samples_total}, K={total_clients}")

        if per_client < 10:
            logger.warning(f"[CIFAR10] per_client={per_client} < 10; BatchNorm may be unstable.")

        if alpha is None or alpha <= 0:
            alpha = 1e-6

        # --- Build per-class indices efficiently (sort + searchsorted) ---
        sorted_idx = np.argsort(self.targets)
        sorted_y = self.targets[sorted_idx]

        idx_by_class = []
        ptr = []
        for k in range(num_classes):
            lo = np.searchsorted(sorted_y, k, side="left")
            hi = np.searchsorted(sorted_y, k, side="right")
            idx_k = sorted_idx[lo:hi].copy()
            rng.shuffle(idx_k)
            idx_by_class.append(idx_k)
            ptr.append(0)

        # Dirichlet per client (K x C)
        probs = rng.dirichlet([alpha] * num_classes, size=total_clients)

        idx_batch = [[] for _ in range(total_clients)]

        for j in range(total_clients):
            desired = probs[j] * per_client
            base = np.floor(desired).astype(int)

            # Distribute remainder by largest fractional parts
            remain = per_client - int(base.sum())
            if remain > 0:
                frac = desired - base
                order = np.argsort(frac)[::-1]
                for t in order[:remain]:
                    base[t] += 1

            indices_j = []
            deficit = 0

            # 1) Try to follow desired per-class counts
            for k, cnt in enumerate(base):
                if cnt <= 0:
                    continue
                avail = len(idx_by_class[k]) - ptr[k]
                take = min(cnt, avail)
                if take > 0:
                    indices_j.extend(idx_by_class[k][ptr[k]:ptr[k] + take].tolist())
                    ptr[k] += take
                deficit += (cnt - take)

            # 2) Fill deficit from remaining classes (prefer high-probability ones)
            if deficit > 0:
                pref = np.argsort(probs[j])[::-1]
                for k in pref:
                    if deficit <= 0:
                        break
                    avail = len(idx_by_class[k]) - ptr[k]
                    if avail <= 0:
                        continue
                    take = min(deficit, avail)
                    indices_j.extend(idx_by_class[k][ptr[k]:ptr[k] + take].tolist())
                    ptr[k] += take
                    deficit -= take

            if len(indices_j) != per_client:
                raise RuntimeError(
                    f"[CIFAR10] Could not allocate per_client={per_client} for client {j}. "
                    f"Allocated={len(indices_j)}. (new_size likely too small or empty classes.)"
                )

            rng.shuffle(indices_j)
            idx_batch[j] = indices_j

        my_indices = np.array(idx_batch[client_id], dtype=np.int64)
        self.data = self.data[my_indices]
        self.targets = self.targets[my_indices]

        logger.info(
            f"[CIFAR10] Partition complete (Dirichlet alpha={alpha}, exact size). "
            f"Client {client_id}/{total_clients} -> {len(self.data)} samples | "
            f"per_client={per_client} | discarded_global={discarded}"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target