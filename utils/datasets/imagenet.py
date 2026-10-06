# utils/datasets/imagenet.py
# @author: Marcio Lopes

import torch
import numpy as np
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from PIL import Image
import os
from utils.linda_logger import logger


class ImageNetDataset(Dataset):
    """
    Split-Learning friendly wrapper for ImageNet (ILSVRC2012).
    Same contract as Cifar10Dataset:
      - Non-IID partitioning via Dirichlet
      - Dataset truncation (new_size) for debug/testing
      - Separate train/val paths

    REQUIREMENT: ImageNet must be pre-downloaded and organized as:
        root/
        ├── train/
        │   ├── n01440764/
        │   │   ├── n01440764_10026.JPEG
        │   │   └── ...
        │   └── ...  (1000 class folders)
        └── val/
            ├── n01440764/
            │   ├── ILSVRC2012_val_00000293.JPEG
            │   └── ...
            └── ...  (1000 class folders)
    """

    # --- ImageNet normalization stats (standard PyTorch) ---
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)
    NUM_CLASSES   = 1000

    def __init__(self,
                 root,
                 train=True,
                 client_id=None,
                 total_clients=None,
                 alpha=0.5,
                 seed=42,
                 download=False,      # ImageNet nao suporta auto-download
                 new_size=None,
                 image_size=224):      # Resolucao nativa do ResNet50
        """
        Args:
            root (str): ImageNet root path (containing train/ and val/).
            train (bool): True = 'train' split, False = 'val' split.
            client_id (int): Client ID for partitioning (0-indexed).
            total_clients (int): Total clients in the experiment.
            alpha (float): Dirichlet parameter. Smaller = more heterogeneous.
            seed (int): Seed for reproducibility.
            download (bool): Ignored (ImageNet requires manual download).
            new_size (int/float): Truncate dataset. Float < 1.0 = fraction. Int >= 1 = absolute.
            image_size (int): Resize resolution (default 224 for ResNet50).
        """
        self.root = root
        self.train = train
        self.image_size = image_size

        # =================================================================
        # 1. Transforms (ImageNet standard)
        # =================================================================
        if self.train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size + 32),   # ex: 256
                transforms.CenterCrop(image_size),     # 224
                transforms.ToTensor(),
                transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD)
            ])

        # =================================================================
        # 2. Load base dataset via ImageFolder
        # =================================================================
        split_dir = os.path.join(self.root, "train" if self.train else "val")

        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"[ImageNet] Directory not found: {split_dir}\n"
                f"ImageNet must be manually downloaded and extracted.\n"
                f"Expected: {self.root}/train/<class_folders>/ "
                f"and {self.root}/val/<class_folders>/"
            )

        logger.info(f"[ImageNet] Loading from: {split_dir} (train={self.train})...")
        base_dataset = datasets.ImageFolder(split_dir)

        # Store only (path, class_idx) — pixels loaded on demand (ImageNet ~150GB)
        self.samples = base_dataset.samples
        self.targets = np.array([s[1] for s in self.samples])
        self.classes = base_dataset.classes

        logger.info(
            f"[ImageNet] Found {len(self.samples)} images "
            f"in {len(self.classes)} classes."
        )

        # =================================================================
        # 3. Truncation (new_size)
        # =================================================================
        if new_size is not None and new_size > 0:
            total_len = len(self.samples)
            truncate_to = total_len

            if isinstance(new_size, float) and new_size < 1.0:
                truncate_to = max(1, int(total_len * new_size))
            elif isinstance(new_size, (int, float)):
                truncate_to = min(total_len, int(new_size))

            if truncate_to < total_len:
                logger.warning(
                    f"[ImageNet] Truncating from {total_len} to "
                    f"{truncate_to} samples. Seed={seed}."
                )
                np.random.seed(seed)
                indices = np.random.choice(total_len, truncate_to, replace=False)
                self.samples = [self.samples[i] for i in indices]
                self.targets = self.targets[indices]

        # =================================================================
        # 4. Non-IID Partitioning (train only)
        # =================================================================
        if self.train:
            if client_id is None or total_clients is None:
                raise ValueError(
                    "For train=True, client_id and total_clients are required."
                )
            self._partition_data_balanced(client_id, total_clients, alpha, seed)
        else:
            logger.info(
                f"[ImageNet] Loaded Global Val Set ({len(self.samples)} samples)."
            )

    def _partition_data_balanced(self, client_id, total_clients, alpha, seed):
        """
        Non-IID Dirichlet with EXACT size per client:
          - Each client gets exactly floor(N / K) samples (no replacement).
          - Remainder is discarded.
          - alpha controls class skew (smaller alpha => more concentrated).

        Efficient: uses sort + searchsorted instead of np.where for 1000 classes.
        """
        rng = np.random.RandomState(seed)

        num_classes = self.NUM_CLASSES
        n_samples_total = len(self.targets)
        per_client = n_samples_total // total_clients
        discarded = n_samples_total - per_client * total_clients

        if per_client <= 0:
            raise ValueError(f"[ImageNet] per_client=0. N={n_samples_total}, K={total_clients}")

        if per_client < 10:
            logger.warning(f"[ImageNet] per_client={per_client} < 10; BatchNorm may be unstable.")

        if alpha is None or alpha <= 0:
            alpha = 1e-6

        # --- Build per-class indices efficiently ---
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

            remain = per_client - int(base.sum())
            if remain > 0:
                frac = desired - base
                order = np.argsort(frac)[::-1]
                for t in order[:remain]:
                    base[t] += 1

            indices_j = []
            deficit = 0

            for k, cnt in enumerate(base):
                if cnt <= 0:
                    continue
                avail = len(idx_by_class[k]) - ptr[k]
                take = min(cnt, avail)
                if take > 0:
                    indices_j.extend(idx_by_class[k][ptr[k]:ptr[k] + take].tolist())
                    ptr[k] += take
                deficit += (cnt - take)

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
                    f"[ImageNet] Could not allocate per_client={per_client} for client {j}. "
                    f"Allocated={len(indices_j)}."
                )

            rng.shuffle(indices_j)
            idx_batch[j] = indices_j

        my_indices = np.array(idx_batch[client_id], dtype=np.int64)
        self.samples = [self.samples[i] for i in my_indices]
        self.targets = self.targets[my_indices]

        logger.info(
            f"[ImageNet] Partition complete (Dirichlet alpha={alpha}, exact size). "
            f"Client {client_id}/{total_clients} -> {len(self.samples)} samples | "
            f"per_client={per_client} | discarded_global={discarded}"
        )

    def _partition_data_balanced_old(self, client_id, total_clients, alpha, seed):
        """
        Non-IID Dirichlet with balanced volume (legacy).
        Same as Cifar10Dataset, scaled to 1000 classes.
        """
        np.random.seed(seed)

        num_classes = self.NUM_CLASSES
        n_samples_total = len(self.targets)

        min_size = 0
        min_require_size = 10

        while min_size < min_require_size:
            proportions_matrix = np.random.dirichlet(
                [alpha] * total_clients,
                num_classes
            )

            idx_batch = [[] for _ in range(total_clients)]
            for k in range(num_classes):
                idx_k = np.where(self.targets == k)[0]
                np.random.shuffle(idx_k)

                proportions = proportions_matrix[k]
                proportions = np.array([
                    p * (len(idx_last) < n_samples_total / total_clients)
                    for p, idx_last in zip(proportions, idx_batch)
                ])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]

            min_size = min(len(idx_j) for idx_j in idx_batch)

        # Apply this client's partition
        my_indices = np.array(idx_batch[client_id])
        np.random.shuffle(my_indices)

        self.samples = [self.samples[i] for i in my_indices]
        self.targets = self.targets[my_indices]

        logger.info(
            f"[ImageNet] Partition complete (Dirichlet alpha={alpha}). "
            f"Client {client_id}/{total_clients} -> {len(self.samples)} samples."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """Lazy loading: opens image from disk on demand."""
        path, target = self.samples[index]

        # Convert to RGB (some ImageNet images are grayscale)
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target