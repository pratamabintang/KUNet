import os
import random
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from PIL import Image
import cv2


def load_blacklist(blacklist_path: Optional[str]) -> set:
    """Loads sample identifiers to ignore from a blacklist text file."""
    if not blacklist_path or not os.path.exists(blacklist_path):
        return set()
    with open(blacklist_path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


class LandslideDataset(Dataset):
    """
    Multi-modal Landslide Dataset supporting flexible modality selection,
    per-tile normalization, synchronized spatial transforms, and blacklist filtering.
    """

    SUPPORTED_MODALITIES = [
        "IMAGE",
        "DTM",
        "SLOPE",
        "ASPECT",
        "ASPECT_COS",
        "ASPECT_SIN",
        "HILLSHADE",
    ]

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        data_dir: str = "datasets/landslide",
        split: str = "train",
        size: int = 512,
        modalities: Optional[List[str]] = None,
        blacklist_path: Optional[str] = "datasets/landslide/black_list.txt",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.target_size = size

        if modalities is None:
            self.modalities = ["IMAGE"]
        else:
            self.modalities = list(modalities)

        for m in self.modalities:
            if m not in self.SUPPORTED_MODALITIES:
                raise ValueError(
                    f"Modality '{m}' is not supported. Supported: {self.SUPPORTED_MODALITIES}"
                )

        if os.path.isdir(os.path.join(data_dir, split)):
            self.split_dir = os.path.join(data_dir, split)
        elif os.path.isdir(data_dir):
            self.split_dir = data_dir
        else:
            raise FileNotFoundError(f"Split directory not found for data_dir='{data_dir}', split='{split}'")

        self.blacklist = load_blacklist(blacklist_path)

        image_dir = os.path.join(self.split_dir, "IMAGE")
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"IMAGE directory missing in {self.split_dir}")

        all_files = sorted(os.listdir(image_dir))
        self.samples = []
        for f in all_files:
            base_name, _ = os.path.splitext(f)
            if base_name in self.blacklist:
                continue
            self.samples.append(base_name)

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found in {self.split_dir} (total files: {len(all_files)})")

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_photometric_augmentations(self, rgb: np.ndarray) -> np.ndarray:
        """
        Applies color and texture augmentations EXCLUSIVELY to optical RGB.
        Never applied to physical DTM / Slope rasters.
        """
        # ---------------------------------------------------------------------
        # 2.5 Color Jitter / Random Brightness & Contrast (p=0.5)
        # Brightness +-0.2, Contrast +-0.3, Saturation +-0.2
        # Resolves optical luminance shift across validation corridors
        # ---------------------------------------------------------------------
        if random.random() < 0.5:
            # Contrast [0.7, 1.3], Brightness [-51, 51]
            contrast_factor = random.uniform(0.7, 1.3)
            brightness_offset = random.uniform(-51.0, 51.0)
            rgb = np.clip(
                contrast_factor * rgb.astype(np.float32) + brightness_offset, 0, 255
            ).astype(np.uint8)

            # Saturation scaling [0.8, 1.2] in HSV space
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            sat_factor = random.uniform(0.8, 1.2)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
            rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # ---------------------------------------------------------------------
        # 2.6 CLAHE (p=0.3, clip=2.0, tile=(8, 8))
        # Enhances ground texture and fractures in shadowed mountain flanks
        # ---------------------------------------------------------------------
        if random.random() < 0.3:
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # ---------------------------------------------------------------------
        # 2.7 Gaussian Blur / Motion Blur (p=0.2, kernel 3x3 or 5x5)
        # Simulates drone platform motion vibration blur
        # ---------------------------------------------------------------------
        if random.random() < 0.2:
            k = random.choice([3, 5])
            if random.random() < 0.5:
                # Gaussian blur
                rgb = cv2.GaussianBlur(rgb, (k, k), 0)
            else:
                # Directional motion blur
                kernel = np.zeros((k, k), dtype=np.float32)
                if random.random() < 0.5:
                    kernel[int((k - 1) / 2), :] = np.ones(k, dtype=np.float32)
                else:
                    kernel[:, int((k - 1) / 2)] = np.ones(k, dtype=np.float32)
                kernel /= k
                rgb = cv2.filter2D(rgb, -1, kernel)

        return rgb

    def _load_modality(self, sample_name: str, modality: str) -> torch.Tensor:
        """Loads and normalizes an individual raster modality as a PyTorch Tensor."""
        ext = ".png" if modality == "IMAGE" else ".tif"
        file_path = os.path.join(self.split_dir, modality, sample_name + ext)
        if not os.path.exists(file_path):
            alt_ext = ".tif" if ext == ".png" else ".png"
            file_path = os.path.join(self.split_dir, modality, sample_name + alt_ext)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File for modality '{modality}' not found for sample '{sample_name}'")

        img = Image.open(file_path)

        if modality == "IMAGE":
            rgb_arr = np.array(img.convert("RGB"), dtype=np.uint8)
            if self.split == "train":
                rgb_arr = self._apply_photometric_augmentations(rgb_arr)
            tensor = TF.to_tensor(rgb_arr)  # Shape (3, H, W), range [0, 1]
            tensor = TF.normalize(tensor, mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
            return tensor

        arr = np.array(img, dtype=np.float32)

        if modality == "DTM":
            # Per-tile Min-Max Normalization: captures relative local topography
            if np.isnan(arr).any() or np.isinf(arr).any():
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

            valid_mask = (arr > -1000.0) & (arr < 10000.0)
            if valid_mask.any():
                val_min = float(arr[valid_mask].min())
                val_max = float(arr[valid_mask].max())
            else:
                val_min, val_max = 0.0, 0.0

            if val_max > val_min:
                norm_arr = np.clip((arr - val_min) / (val_max - val_min), 0.0, 1.0)
            else:
                norm_arr = np.zeros_like(arr)
            return torch.from_numpy(norm_arr).unsqueeze(0)  # Shape (1, H, W)

        elif modality == "SLOPE":
            # Slope in degrees [0, 90] -> scale to [0, 1]
            norm_arr = np.clip(arr / 90.0, 0.0, 1.0)
            return torch.from_numpy(norm_arr).unsqueeze(0)

        elif modality == "ASPECT":
            # Aspect in degrees [0, 360] -> scale to [0, 1]
            norm_arr = np.clip(arr / 360.0, 0.0, 1.0)
            return torch.from_numpy(norm_arr).unsqueeze(0)

        elif modality in ("ASPECT_COS", "ASPECT_SIN"):
            # Cyclical components already in [-1.0, 1.0]
            norm_arr = np.clip(arr, -1.0, 1.0)
            return torch.from_numpy(norm_arr).unsqueeze(0)

        elif modality == "HILLSHADE":
            # Topographic illumination already in [0.0, 1.0]
            norm_arr = np.clip(arr, 0.0, 1.0)
            return torch.from_numpy(norm_arr).unsqueeze(0)

        else:
            raise NotImplementedError(f"Loader logic for modality '{modality}' is not implemented.")

    def _load_label(self, sample_name: str) -> torch.Tensor:
        """Loads ground truth binary landslide label mask as float tensor {0.0, 1.0}."""
        label_path = os.path.join(self.split_dir, "LABEL", sample_name + ".png")
        if not os.path.exists(label_path):
            label_path = os.path.join(self.split_dir, "LABEL", sample_name + ".tif")
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label file not found for sample '{sample_name}'")

        img = Image.open(label_path)
        arr = np.array(img)
        # Landslide is 255 (or > 0), background is 0
        bin_arr = (arr > 0).astype(np.float32)
        return torch.from_numpy(bin_arr).unsqueeze(0)  # Shape (1, H, W)

    def _get_landslide_preserving_crop_params(
        self, label: torch.Tensor, scale_range: Tuple[float, float] = (0.8, 0.95)
    ) -> Tuple[int, int, int, int]:
        """
        Calculates crop coordinates [y0, x0, ch, cw] targeting 0.8-0.95 of original dimensions.
        If landslide pixels are present, anchors crop around the landslide centroid and
        bounds to ensure landslide regions are preserved without being truncated.
        """
        _, H, W = label.shape
        scale = random.uniform(*scale_range)
        ch = int(H * scale)
        cw = int(W * scale)

        pos_idx = torch.nonzero(label.squeeze(0) > 0)
        if len(pos_idx) > 0:
            y_min = int(pos_idx[:, 0].min().item())
            y_max = int(pos_idx[:, 0].max().item())
            x_min = int(pos_idx[:, 1].min().item())
            x_max = int(pos_idx[:, 1].max().item())

            cy = (y_min + y_max) // 2
            cx = (x_min + x_max) // 2

            # Vertical window preserving landslide bounding box
            y_low = max(0, y_max - ch)
            y_high = min(H - ch, y_min)
            ideal_y = cy - ch // 2
            if y_low <= y_high:
                y0 = max(y_low, min(ideal_y, y_high))
            else:
                y0 = max(0, min(ideal_y, H - ch))

            # Horizontal window preserving landslide bounding box
            x_low = max(0, x_max - cw)
            x_high = min(W - cw, x_min)
            ideal_x = cx - cw // 2
            if x_low <= x_high:
                x0 = max(x_low, min(ideal_x, x_high))
            else:
                x0 = max(0, min(ideal_x, W - cw))
        else:
            y0 = random.randint(0, max(0, H - ch))
            x0 = random.randint(0, max(0, W - cw))

        return y0, x0, ch, cw

    def _apply_transforms(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Synchronously applies geometric augmentations to multi-modal image and label mask."""
        if self.split == "train":
            # 1. Landslide-preserving random crop (0.8 - 0.95 scale of original size)
            y0, x0, ch, cw = self._get_landslide_preserving_crop_params(label, scale_range=(0.8, 0.95))
            image = TF.crop(image, y0, x0, ch, cw)
            label = TF.crop(label, y0, x0, ch, cw)

            # 2. Synchronized random horizontal flip
            if random.random() < 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)

            # 3. Synchronized random vertical flip
            if random.random() < 0.5:
                image = TF.vflip(image)
                label = TF.vflip(label)

        # Synchronized resize back to target size
        # Bilinear for continuous inputs, Nearest for discrete binary label
        image = TF.resize(
            image,
            [self.target_size, self.target_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        label = TF.resize(
            label,
            [self.target_size, self.target_size],
            interpolation=InterpolationMode.NEAREST,
        )

        return image, label

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_name = self.samples[idx]

        # Load and stack requested modalities along channel dimension
        modality_tensors = [self._load_modality(sample_name, m) for m in self.modalities]
        image_tensor = torch.cat(modality_tensors, dim=0)  # Shape (C, H, W)

        # Load binary label mask
        label_tensor = self._load_label(sample_name)  # Shape (1, H, W)

        # Synchronous spatial transforms
        image_tensor, label_tensor = self._apply_transforms(image_tensor, label_tensor)

        return {
            "image": image_tensor,
            "label": label_tensor,
            "name": sample_name,
        }