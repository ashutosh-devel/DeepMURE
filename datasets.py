"""
PyTorch Dataset classes for SAR image denoising.

Provides DataGenerator and RealDataGenerator classes for loading and
preprocessing SAR images with gamma noise simulation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from random import shuffle
import os
from typing import List, Tuple, Optional, Union
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import warnings

class DataGenerator(Dataset):
    """
    PyTorch Dataset for loading paired noisy/clean images.
    
    Generates batches of noisy and clean image pairs for training
    denoising models on SAR-like data.
    
    Args:
        image_paths: List of paths to noisy image files
        batchsize: Number of images per batch
        scaling: Scaling factor for images
        make_even: Whether to ensure even image dimensions
        n_looks: Number of looks for gamma noise (SAR parameter)
        outsize: Output image size [height, width, channels]
        augment: Whether to apply data augmentation
    """
    
    def __init__(
        self,
        image_paths: List[str],
        batchsize: int = 1,
        scaling: float = 1.0,
        make_even: bool = False,
        n_looks: int = 1,
        outsize: Optional[List[int]] = None,
        augment: bool = False
    ):
        super().__init__()
        self.image_paths = image_paths
        self.n_looks = n_looks  # gamma density parameter
        self.img_size = outsize
        self.make_even = make_even
        self.max_allowed_val = 10000.0
        self.augment = augment
        self.outsize = outsize
        self.scaling = scaling
        self.batchsize = batchsize
        self.num_images = len(image_paths)
        self.peak_val = 256.0
        self.seed = None
        self.pad = False
        
        # Device for tensor operations
        self.device = torch.device('cpu')
    
    def __len__(self) -> int:
        """Returns the number of batches per epoch."""
        return int(math.floor(len(self.image_paths) / self.batchsize))
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate one batch of data.
        
        Args:
            index: Batch index
            
        Returns:
            Tuple of (noisy, clean, factors) tensors
        """
        indices = np.array(range(
            self.batchsize * index,
            self.batchsize * (index + 1)
        )) % self.num_images
        
        noisy, clean, factors = self._data_generation(indices)
        
        # Stack lists into tensors
        clean = torch.stack([torch.from_numpy(np.array(c)).float() for c in clean])
        noisy = torch.stack([torch.from_numpy(np.array(n)).float() for n in noisy])
        factors = torch.tensor(factors, dtype=torch.float32)
        
        return noisy, clean, factors
    
    def shuffle_paths(self) -> None:
        """Shuffle the image paths for a new epoch."""
        shuffle(self.image_paths)
    
    def get_gamma_noise(
        self,
        alpha: float = 1.0,
        beta: float = 1.0
    ) -> torch.Tensor:
        """
        Generate gamma-distributed noise.
        
        Args:
            alpha: Shape parameter (concentration)
            beta: Rate parameter
            
        Returns:
            Tensor of gamma-distributed random values
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        
        # Use numpy for gamma distribution (more reliable)
        # PyTorch's Gamma: rate = 1/scale, so we use beta directly
        noise = np.random.gamma(shape=alpha, scale=1.0/beta, size=self.img_size)
        return torch.from_numpy(noise).float()
    
    def _add_noise_to_clean(self, clean: np.ndarray, L: int = 1) -> np.ndarray:
        """Add gamma noise to clean image."""
        noisy = self.forward_transform(clean)
        noisy = noisy * self.get_gamma_noise().numpy()
        return noisy
    
    def _load_grayscale_image(self, image_path: str) -> np.ndarray:
        """
        Load grayscale image from file.
        
        Args:
            image_path: Path to image file (.npy or image format)
            
        Returns:
            Grayscale image as numpy array
        """
        if image_path.endswith('.npy'):
            img = np.load(image_path)
        else:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    
    def _load_clean_image(self, image_path: str) -> np.ndarray:
        """
        Load corresponding clean image.
        
        Args:
            image_path: Path to noisy image (replaces 'noisy' with 'clean')
            
        Returns:
            Clean grayscale image as numpy array
        """
        clean_image_path = image_path.replace("noisy", "clean")
        
        if image_path.endswith('.npy'):
            img = np.load(clean_image_path)
        else:
            img = cv2.imread(clean_image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    
    def forward_transform(self, img: np.ndarray) -> np.ndarray:
        """
        Transform image from magnitude to intensity domain.
        
        Args:
            img: Input image in magnitude domain
            
        Returns:
            Image in intensity domain
        """
        img = ((img + 1) ** 2 / (self.peak_val ** 2))
        return img
    
    def reverse_transform(self, img: torch.Tensor) -> torch.Tensor:
        """
        Transform image from intensity back to magnitude domain.
        
        Args:
            img: Input image in intensity domain
            
        Returns:
            Image in magnitude domain
        """
        img = torch.sqrt(img) * (self.peak_val ** 2) - 1
        return img
    
    def _data_generation(
        self,
        indices: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """
        Generate batch of samples.
        
        Args:
            indices: Array of image indices to load
            
        Returns:
            Tuple of (noisy_images, clean_images, multiplication_factors)
        """
        batch_input_imgs = []
        batch_output_imgs = []
        mult_factors = []
        
        for idx in indices:
            # Load noisy image
            noisy_img = self._load_grayscale_image(self.image_paths[idx])
            noisy_img = np.expand_dims(noisy_img, axis=-1).astype(np.float32)
            
            # Load corresponding clean image
            clean_img = self._load_clean_image(self.image_paths[idx])
            clean_img = np.expand_dims(clean_img, axis=-1).astype(np.float32)
            
            # Apply padding if needed
            if self.pad:
                noisy_img = np.expand_dims(
                    np.pad(noisy_img[:, :, 0], (6, 6), mode='reflect'),
                    axis=-1
                )
                clean_img = np.expand_dims(
                    np.pad(clean_img[:, :, 0], (6, 6), mode='reflect'),
                    axis=-1
                )
            
            # Center crop to output size
            outsize = self.outsize
            h, w = noisy_img.shape[:2]
            h_start = h // 2 - outsize[0] // 2
            w_start = w // 2 - outsize[1] // 2
            
            noisy_img = noisy_img[
                h_start:h_start + outsize[0],
                w_start:w_start + outsize[1]
            ]
            clean_img = clean_img[
                h_start:h_start + outsize[0],
                w_start:w_start + outsize[1]
            ]
            
            # Apply forward transform (magnitude to intensity)
            noisy_img = self.forward_transform(noisy_img)
            noisy_img = np.clip(noisy_img, 0.00001, 4.0)
            
            clean_img = self.forward_transform(clean_img)
            clean_img = np.clip(clean_img, 0.00001, 1.0)
            
            self.img_size = clean_img.shape
            mult_factor = 1.0
            
            # Transpose to [C, H, W] format for PyTorch
            batch_input_imgs.append(np.transpose(noisy_img, (2, 0, 1)))
            batch_output_imgs.append(np.transpose(clean_img, (2, 0, 1)))
            mult_factors.append(mult_factor)
        
        return batch_input_imgs, batch_output_imgs, mult_factors

class SynthMultiLookDataset(Dataset):
    """
    Synthetic multi‑look SAR‑style dataset.

    Parameters
    ----------
    data_folder : str
        Folder that contains the source ``.png`` images.
    crop_size : Tuple[int, int], default (256, 256)
        Height × width of the returned crops.
    samples_per_epoch : int, default 128
        Length of the virtual dataset (how many crops are sampled per epoch).
    gamma_shape : float, default 1.0
        Shape (α) of the Gamma distribution.  The scale (β) is set equal to α
        so that the noise has unit mean (E[noise]=1) and variance 1/α.
    isval : bool, default False
        If ``True`` the dataset returns *the same* noisy crop for a given
        ``idx`` (deterministic validation set).  If ``False`` the crop
        and the noise are drawn anew at every call (training mode).
    val_seed : int, default 0
        Base seed for the deterministic validation sampling.  Changing the
        seed gives a different deterministic validation split while still
        being reproducible.
    """

    def __init__(
        self,
        data_folder: str,
        crop_size: Tuple[int, int] = (256, 256),
        samples_per_epoch: int = 128,
        gamma_shape: float = 1.0,
        isval: bool = False,
        val_seed: int = 21,
    ) -> None:
        # --------------------------------------------------------------
        # Store basic configuration
        # --------------------------------------------------------------
        self.crop_h, self.crop_w = crop_size
        self.samples_per_epoch = int(samples_per_epoch)
        self.gamma_shape = float(gamma_shape)
        self.isval = bool(isval)
        self.val_seed = int(val_seed)
        self.img_size = crop_size

        # --------------------------------------------------------------
        # Load all images into RAM (as float32, normalised, squared)
        # --------------------------------------------------------------
        self.images: List[np.ndarray] = []
        file_list = glob.glob(os.path.join(data_folder, "*.png"))
        if not file_list:
            raise RuntimeError(f"No .png files found in {data_folder}")

        print(f"Loading {len(file_list)} large images into RAM...")
        for f in file_list:
            img = Image.open(f).convert("L")               # grayscale
            img = np.array(img).astype(np.float32) / 255.0  # [0,1]
            img = img ** 2.0                               # apply the intensity model

            # Ensure a 3‑D array (H, W, C) – C will always be 1 here
            if img.ndim == 2:
                img = img[:, :, np.newaxis]

            h, w, _ = img.shape
            if h < self.crop_h or w < self.crop_w:
                print(f"Warning: Image {f} is smaller than crop size – skipping.")
                continue
            self.images.append(img)

        if not self.images:
            raise RuntimeError("No valid images loaded (check dimensions).")
        print("Data loading complete.")

        # --------------------------------------------------------------
        # Pre‑compute a Gamma‑noise field for every image **only** when
        # we are in validation mode.  This field has the same spatial size
        # as the original image, so we can later crop the exact same noise
        # patch that belongs to a deterministic crop.
        # --------------------------------------------------------------
        self.noise_maps: Optional[List[torch.Tensor]] = None
        if self.isval:
            self._precompute_noise_maps()

        # --------------------------------------------------------------
        # Simple transform – we already have a numpy array, we only need
        # to turn it into a torch Tensor.  ``ToTensor`` would also convert
        # a PIL Image, but using ``torch.from_numpy`` is a little faster.
        # --------------------------------------------------------------
        self.transform = transforms.Compose([transforms.ToTensor()])

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Helper that creates a deterministic crop position from ``idx``.
    # The arithmetic is deliberately simple (modulo) and therefore
    # repeatable without any external RNG.
    # ------------------------------------------------------------------
    def _deterministic_crop_coords(
        self, idx: int, img_h: int, img_w: int
    ) -> Tuple[int, int]:
        """Return (top, left) that is reproducible for a given ``idx``."""
        # Use the validation‑seed as a base so that different seeds give
        # different but still deterministic splits.
        base = self.val_seed + idx
        # Simple linear congruential scheme – any scheme would do as long
        # as it is deterministic and stays inside the valid range.
        top = (base * 12345) % (img_h - self.crop_h + 1)
        left = (base * 67890) % (img_w - self.crop_w + 1)
        return int(top), int(left)

    # ------------------------------------------------------------------
    # If we are in validation mode we generate one Gamma‑noise map per
    # image *once* and store it for later cropping.
    # ------------------------------------------------------------------
    def _precompute_noise_maps(self) -> None:
        """Create a full‑size Gamma noise tensor for every cached image."""
        self.noise_maps: List[torch.Tensor] = []
        alpha = torch.tensor(self.gamma_shape, dtype=torch.float32)
        beta = torch.tensor(self.gamma_shape, dtype=torch.float32)
        gamma_dist = torch.distributions.Gamma(alpha, beta)

        for img in self.images:
            # ``img`` shape is (H, W, C).  We need a noise map of the same shape.
            h, w, c = img.shape
            # Sample on CPU – we will move it to the appropriate device later.
            noise_np = gamma_dist.sample((h, w, c)).squeeze(-1).numpy()
            self.noise_maps.append(noise_np.astype(np.float32))

    # ------------------------------------------------------------------
    # Required ``__len__`` implementation – the virtual length of the
    # dataset is ``samples_per_epoch`` irrespective of how many images we
    # actually have in memory.
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.samples_per_epoch

    # ------------------------------------------------------------------
    # ``__getitem__`` – the only place where the behaviour diverges
    # between training and validation.
    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        noisy : torch.Tensor
            Shape (C, H, W) – clean image multiplied by a Gamma noise map.
        clean : torch.Tensor
            Shape (C, H, W) – the ground‑truth (noise‑free) crop.
        """
        # --------------------------------------------------------------
        # Pick an image (wrap around if ``idx`` exceeds the number of cached
        # images – this mirrors the behaviour of the original code).
        # --------------------------------------------------------------
        img_idx = idx % len(self.images)
        clean_np = self.images[img_idx]                     # (H, W, C)
        h, w, c = clean_np.shape
        # print(clean_np.shape)
        # Make dimensions even if they are odd   
        # --------------------------------------------------------------
        # Crop – random for training, deterministic for validation
        # --------------------------------------------------------------
        if self.isval:
            top, left = self._deterministic_crop_coords(idx, h, w)
        else:
            top = np.random.randint(0, h - self.crop_h + 1)
            left = np.random.randint(0, w - self.crop_w + 1)

        clean_crop = clean_np[
            top : top + self.crop_h,
            left : left + self.crop_w,
            :,
        ]                                                  # (crop_h, crop_w, C)

        # --------------------------------------------------------------
        # Convert the clean crop to a torch Tensor (C, H, W)
        # --------------------------------------------------------------
        clean_tensor = (
            torch.from_numpy(clean_crop)
            .permute(2, 0, 1)          # (C, H, W)
            .contiguous()
            .float()
        )

        # --------------------------------------------------------------
        # Noise – fresh sample for training, pre‑computed slice for val
        # --------------------------------------------------------------
        if self.isval:
            # ``self.noise_maps`` exists because we called
            # ``_precompute_noise_maps`` in ``__init__`` when ``isval=True``.
            noise_full = self.noise_maps[img_idx]          # (H, W, C) torch Tensor
            noise_crop = torch.from_numpy(noise_full[
                top : top + self.crop_h,
                left : left + self.crop_w]).contiguous()
            # ``noise_crop`` already has shape (C, H, W)
            noise_tensor = noise_crop
        else:
            # Fresh Gamma noise for the *exact* shape of the crop
            alpha = torch.tensor(self.gamma_shape, dtype=torch.float32)
            beta = torch.tensor(self.gamma_shape, dtype=torch.float32)
            gamma_dist = torch.distributions.Gamma(alpha, beta)
            # ``sample`` returns shape (C, H, W) because we pass the
            # desired sample shape directly.
            noise_tensor = gamma_dist.sample(clean_tensor.shape).squeeze(-1)

        # --------------------------------------------------------------
        # Build the noisy observation
        # --------------------------------------------------------------
        noisy_tensor = clean_tensor * noise_tensor

        # if noisy_tensor.shape[1] != 512:
        #     clean_tensor = F.interpolate(clean_tensor.unsqueeze(0),size=(512, 512),mode='bilinear',align_corners=True).squeeze(0)
        #     noisy_tensor = F.interpolate(noisy_tensor.unsqueeze(0),size=(512, 512),mode='bilinear',align_corners=True).squeeze(0)

        #prevent zeros in the tensors which can cause issues with estimation
        clean_tensor = torch.clamp(clean_tensor, 1e-7)
        noisy_tensor = torch.clamp(noisy_tensor, 1e-7)

        return noisy_tensor, clean_tensor, 0.0

class SynthMultiLookDatasetNpy(Dataset):
    """
    Synthetic multi‑look SAR‑style dataset that reads *raw NumPy arrays*
    instead of PNG files.

    Parameters
    ----------
    data_folder : str
        Folder that contains the source ``*.npy`` (or any other extension) files.
    array_ext : str, default ".npy"
        Extension of the NumPy files to read.  Change to ``".npz"`` or any
        custom suffix you use.
    crop_size : Tuple[int, int], default (256, 256)
        Height × width of the returned crops.
    samples_per_epoch : int, default 128
        Length of the virtual dataset (how many crops are sampled per epoch).
    gamma_shape : float, default 1.0
        Shape (α) of the Gamma distribution.  The scale (β) is set equal to α
        so that the noise has unit mean (E[noise]=1) and variance 1/α.
    isval : bool, default False
        If ``True`` the dataset returns *the same* noisy crop for a given
        ``idx`` (deterministic validation set).  If ``False`` the crop
        and the noise are drawn anew at every call (training mode).
    val_seed : int, default 42
        Base seed for the deterministic validation sampling.  Changing the
        seed gives a different deterministic validation split while still
        being reproducible.
    """

    # ----------------------------------------------------------------------
    # Construction / loading
    # ----------------------------------------------------------------------
    def __init__(
        self,
        data_folder: str,
        array_ext: str = ".npy",
        crop_size: Tuple[int, int] = (256, 256),
        samples_per_epoch: int = 128,
        gamma_shape: float = 1.0,
        isval: bool = False,
        val_seed: int = 42,
    ) -> None:

        # --------------------------------------------------------------
        # Store basic configuration
        # --------------------------------------------------------------
        self.crop_h, self.crop_w = crop_size
        self.samples_per_epoch = int(samples_per_epoch)
        self.gamma_shape = float(gamma_shape)
        self.isval = bool(isval)
        self.val_seed = int(val_seed)

        # --------------------------------------------------------------
        # Load **all** NumPy arrays into RAM (as float32, **no** normalisation)
        # --------------------------------------------------------------
        self.images: List[np.ndarray] = []

        # Accept any extension – we simply glob for it
        file_list = glob.glob(os.path.join(data_folder, f"*{array_ext}"))
        if not file_list:
            raise RuntimeError(f"No {array_ext!r} files found in {data_folder}")

        print(f"Loading {len(file_list)} NumPy arrays into RAM ...")
        for f in file_list:
            # ``np.load`` works for both .npy and .npz (the latter returns a dict)
            arr = np.load(f, allow_pickle=False)
            if isinstance(arr, np.lib.npyio.NpzFile):          # .npz case
                # Expect a single array inside – pick the first one
                arr = list(arr.values())[0]

            # -------------------------f---------------------------------
            # We assume the data is *single‑channel* intensity (H, W) or
            # already 3‑D (H, W, C).  Convert to float32 if necessary.
            # ----------------------------------------------------------
            img = arr.astype(np.float32)

            # Ensure a 3‑D array (H, W, C) – C will always be 1 here
            if img.ndim == 2:
                img = img[:, :, np.newaxis]

            h, w, _ = img.shape
            if h < self.crop_h or w < self.crop_w:
                print(f"Warning: Array {f} is smaller than crop size – skipping.")
                continue

            # The original SAR‑style code squares the intensity to model power.
            # Keep that behaviour (but **do not** divide by 255).
            img = img ** 2.0

            self.images.append(img)

        if not self.images:
            raise RuntimeError("No valid arrays loaded (check dimensions).")
        print("Data loading complete.")

        # --------------------------------------------------------------
        # Pre‑compute a Gamma‑noise field for every image only when we are
        # in validation mode.  This field has the same spatial size as the
        # original image, so we can later crop the exact same noise patch
        # that belongs to a deterministic crop.
        # --------------------------------------------------------------
        self.noise_maps: Optional[List[np.ndarray]] = None
        if self.isval:
            self._precompute_noise_maps()

        # --------------------------------------------------------------
        # Simple transform – we already have a NumPy array, we only need
        # to turn it into a torch Tensor.
        # --------------------------------------------------------------
        self.transform = transforms.Compose([transforms.ToTensor()])

    # ----------------------------------------------------------------------
    # Helper: deterministic crop coordinates (validation mode)
    # ----------------------------------------------------------------------
    def _deterministic_crop_coords(
        self, idx: int, img_h: int, img_w: int
    ) -> Tuple[int, int]:
        """Return (top, left) that is reproducible for a given ``idx``."""
        base = self.val_seed + idx
        top = (base * 12345) % (img_h - self.crop_h + 1)
        left = (base * 67890) % (img_w - self.crop_w + 1)
        return int(top), int(left)

    # ----------------------------------------------------------------------
    # Helper: pre‑compute full‑size Gamma noise maps (validation mode)
    # ----------------------------------------------------------------------
    def _precompute_noise_maps(self) -> None:
        """Create a full‑size Gamma noise tensor for every cached image."""
        self.noise_maps = []
        alpha = torch.tensor(self.gamma_shape, dtype=torch.float32)
        beta = torch.tensor(self.gamma_shape, dtype=torch.float32)
        gamma_dist = torch.distributions.Gamma(alpha, beta)

        for img in self.images:
            h, w, c = img.shape
            # Sample on CPU – we will move it to the appropriate device later.
            noise_np = (
                gamma_dist.sample((h, w, c))
                .squeeze(-1)
                .numpy()
                .astype(np.float32)
            )
            self.noise_maps.append(noise_np)

    # ----------------------------------------------------------------------
    # PyTorch protocol ----------------------------------------------------
    # ----------------------------------------------------------------------
    def __len__(self) -> int:
        """Virtual length – ``samples_per_epoch`` regardless of #images."""
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Returns
        -------
        noisy : torch.Tensor
            Shape (C, H, W) – clean image multiplied by a Gamma noise map.
        clean : torch.Tensor
            Shape (C, H, W) – the ground‑truth (noise‑free) crop.
        weight : float
            Dummy weight (kept for backward compatibility; always 0.0).
        """

        # --------------------------------------------------------------
        # Pick an image (wrap around if ``idx`` exceeds the number of cached
        # images – mirrors the behaviour of the original code).
        # --------------------------------------------------------------
        img_idx = idx % len(self.images)
        clean_np = self.images[img_idx]               # (H, W, C)
        h, w, c = clean_np.shape

        # --------------------------------------------------------------
        # Crop – random for training, deterministic for validation
        # --------------------------------------------------------------
        if self.isval:
            top, left = self._deterministic_crop_coords(idx, h, w)
        else:
            top = np.random.randint(0, h - self.crop_h + 1)
            left = np.random.randint(0, w - self.crop_w + 1)

        clean_crop = clean_np[
            top : top + self.crop_h,
            left : left + self.crop_w,
            :
        ]  # (crop_h, crop_w, C)

        # --------------------------------------------------------------
        # Convert the clean crop to a torch Tensor (C, H, W)
        # --------------------------------------------------------------
        clean_tensor = (
            torch.from_numpy(clean_crop)
            .permute(2, 0, 1)          # (C, H, W)
            .contiguous()
            .float()
        )

        # --------------------------------------------------------------
        # Noise – fresh sample for training, pre‑computed slice for val
        # --------------------------------------------------------------
        if self.isval:
            # ``self.noise_maps`` exists because we called
            # ``_precompute_noise_maps`` in ``__init__`` when ``isval=True``.
            noise_full = self.noise_maps[img_idx]          # (H, W, C) np.ndarray
            noise_crop = torch.from_numpy(
                noise_full[
                    top : top + self.crop_h,
                    left : left + self.crop_w
                ]
            ).contiguous()
            noise_tensor = noise_crop
        else:
            # Fresh Gamma noise for the exact shape of the crop
            alpha = torch.tensor(self.gamma_shape, dtype=torch.float32)
            beta = torch.tensor(self.gamma_shape, dtype=torch.float32)
            gamma_dist = torch.distributions.Gamma(alpha, beta)
            # ``sample`` returns (C, H, W) because we pass the desired shape.
            noise_tensor = gamma_dist.sample(clean_tensor.shape).squeeze(-1)

        # --------------------------------------------------------------
        # Build the noisy observation
        # --------------------------------------------------------------
        noisy_tensor = clean_tensor * noise_tensor
        # print(f"{torch.max(clean_tensor)},{torch.max(noisy_tensor)}")
        # ------------------------------------------------------------------
        # (Optional) keep the same dummy return value that the original code
        # used – a zero weight.  This makes the new class a drop‑in replacement.
        # ------------------------------------------------------------------
        return noisy_tensor, clean_tensor, 0.0


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create a PyTorch DataLoader from a dataset.
    
    Args:
        dataset: PyTorch Dataset instance
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

class CachedNumpyDataset(Dataset):
    def __init__(self, 
                 data_folder, 
                 crop_size=(256, 256), 
                 samples_per_epoch=128, 
                 gamma_shape=1.0):
        """
        Args:
            data_folder (str): Path to folder containing .npy files.
            crop_size (tuple): Desired output size (height, width).
            samples_per_epoch (int): Virtual length of the dataset (how many crops per epoch).
            gamma_shape (float): The 'alpha' parameter for Gamma distribution. 
                                 Higher value = less noise variance.
                                 We assume mean=1, so beta is set equal to alpha.
        """
        self.crop_h, self.crop_w = crop_size
        self.samples_per_epoch = samples_per_epoch
        self.gamma_shape = gamma_shape
        
        
        # 1. Load all images into memory
        self.images = []
        # file_list = glob.glob(os.path.join(data_folder, "*.npy"))
        file_list = glob.glob(os.path.join(data_folder, "*.png"))
        
        if not file_list:
            raise RuntimeError(f"No .npy files found in {data_folder}")
            
        print(f"Loading {len(file_list)} large images into RAM...")
        
        for f in file_list:
            # Load numpy file
            # img = np.load(f)
            
            # # Ensure float32 for PyTorch training
            # if img.dtype != np.float32:
            #     img = img.astype(np.float32)
            
            img = Image.open(f).convert('L') 
            img = np.array(img).astype(np.float32)/255.0
            img = img**2.0
            #img = self.transform(img)
                
            # Handle dimensions: 
            # If 2D (H, W) -> make it (H, W, 1) to unify logic
            if img.ndim == 2:
                img = img[:, :, np.newaxis]
            
            # Check if image is large enough for the crop
            h, w, c = img.shape
            if h < self.crop_h or w < self.crop_w:
                print(f"Warning: Image {f} is smaller than crop size. Skipping.")
                continue
                
            self.images.append(img)
            
        if not self.images:
            raise RuntimeError("No valid images loaded (check dimensions).")
            
        print("Data loading complete.")
        
        self.transform = transforms.Compose([
            transforms.ToTensor(), # [0, 1]
        ])

    def __len__(self):
        # We return a virtual length since we are cropping randomly
        return self.samples_per_epoch

    def __getitem__(self, idx):
        """
        Returns:
            noisy (Tensor): (C, H, W) with Gamma noise
            clean (Tensor): (C, H, W) Ground Truth crop
        """
        # 1. Select a random image from the cached list
        # We use modulo idx for reproducibility if shuffle=False, 
        # or just random integer if shuffle=True
        img_idx = idx % len(self.images)
        img = self.images[img_idx]
        #img = img**2.0
        h, w, c = img.shape
        
        # 2. Random Crop
        # Calculate random top-left coordinate
        top = np.random.randint(0, h - self.crop_h + 1)
        left = np.random.randint(0, w - self.crop_w + 1)
        
        crop = img[top : top + self.crop_h, left : left + self.crop_w, :]
        
        # 3. Convert to Tensor and rearrange to (C, H, W)
        # Current shape is (H, W, C), PyTorch needs (C, H, W)
        clean_tensor = torch.from_numpy(crop).permute(2, 0, 1).contiguous()
        #clean_tensor = clean_tensor #/ torch.max(clean_tensor) #self.transform(clean_tensor)
        
        # 4. Apply Gamma Multiplicative Noise
        # Formula: Noisy = Clean * Noise
        # Noise ~ Gamma(alpha, beta). 
        # To keep mean intensity same, Mean = alpha/beta = 1 -> alpha = beta
        
        alpha = self.gamma_shape
        beta = self.gamma_shape
        
        # Create a Gamma distribution
        m = torch.distributions.Gamma(torch.tensor([alpha]), torch.tensor([beta]))
        
        # Sample noise map of the same shape as image
        noise_map = m.sample(clean_tensor.shape).squeeze(-1).to(clean_tensor.device)
        
        noisy_tensor = clean_tensor * noise_map
        

             
        return noisy_tensor, clean_tensor

# ==========================================
# 2. Dataset with On-the-Fly Gamma Noise
# ==========================================
class SpeckleDataset(Dataset):
    """
    Reads clean images and applies Multiplicative Gamma Noise on the fly.
    Model: Y = X * N, where N ~ Gamma(L, 1/L)
    """
    def __init__(self, img_dir, img_size=256, mode='train', noise_level=10.0):
        super().__init__()
        # Assuming images are png/jpg. Change extensions if needed.
        self.files = sorted(glob.glob(os.path.join(img_dir, '*.*')))
        # Filter only images
        self.files = [f for f in self.files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
        self.files = [f for f in self.files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
        
        self.img_size = img_size
        self.L = noise_level  # Number of looks (Shape parameter)
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(), # [0, 1]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        # Open as grayscale for SAR/Ultrasound
        clean_img = Image.open(path).convert('L') 
        clean = self.transform(clean_img)
        
        # Generate Multiplicative Gamma Noise
        # PyTorch Gamma distribution parametrization: concentration (alpha), rate (beta)
        # Mean = alpha/beta = 1. For Speckle, Mean should be 1. 
        # So alpha = L, beta = L.
        noise = torch.distributions.Gamma(self.L, self.L).sample(clean.shape)
        
        # Y = X * N
        noisy = clean * noise
        
        return noisy, clean, path

class SpeckleNpyDataset(Dataset):
    """
    Reads clean numpy images (.npy) and applies Multiplicative Gamma Noise on the fly.
    Model: Y = X * N, where N ~ Gamma(L, 1/L)
    """
    def __init__(self, img_dir, img_size=256, mode='train', noise_level=1.0):
        super().__init__()
        
        # Filter only .npy files
        # self.files = sorted(glob.glob(os.path.join(img_dir, '*.npy')))
        self.files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        
        self.img_size = img_size
        self.L = noise_level  # Number of looks (Shape parameter)
        
        # We only use Resize here. We handle ToTensor manually for numpy arrays.
                
        self.transform = transforms.Compose([
            transforms.ToTensor(), # [0, 1]
            transforms.Resize((img_size, img_size),antialias=True)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        
        # Load Numpy array
        # Assuming data is (H, W) or (H, W, C)
        # img_np = np.load(path)
        img_np = Image.open(path).convert('L') 
        img_np = np.array(img_np).astype(np.float32)/255.0
        clean = self.transform(img_np)
        clean = clean**2.0
        
        # Ensure float32
        

        # Normalization check: 
        # If data is in range [0, 255], normalize to [0, 1]. 
        # If it is already [0, 1] or raw SAR data, leave as is.
        # if img_np.max() > 1.0:
        #     img_np /= 255.0

        # if img_np.shape[0]%2 != 0 or img_np.shape[1]%2 != 0:
        #     img_np = img_np[:img_np.shape[0]-img_np.shape[0]%2, :img_np.shape[1]-img_np.shape[1]%2]
            
        # Convert to Tensor
        # clean = torch.from_numpy(img_np)

        # Handle dimensions: PyTorch expects (C, H, W)
        # If input is (H, W), make it (1, H, W)
        if clean.ndim == 2:
            clean = clean.unsqueeze(0)
        # If input is (H, W, C), permute to (C, H, W)
        elif clean.ndim == 3 and clean.shape[2] <= 4:
            clean = clean.permute(2, 0, 1)

        # Apply Resize
        #clean = self.resize_transform(clean)
        
        # Generate Multiplicative Gamma Noise
        # PyTorch Gamma distribution parametrization: concentration (alpha), rate (beta)
        # Mean = alpha/beta = 1. So alpha = L, beta = L.
        noise = torch.distributions.Gamma(self.L, self.L).sample(clean.shape)
        
        # Y = X * N
        noisy = clean * noise
        # if True:
        #     display_noisy_clean_batch(noisy, clean)
        
        return noisy, clean, path
    
class SpeckleRealDataset(Dataset):
    """
    Reads clean images and applies Multiplicative Gamma Noise on the fly.
    Model: Y = X * N, where N ~ Gamma(L, 1/L)
    """
    def __init__(self, img_dir, img_size=256, mode='train', noise_level=1.0):
        super().__init__()
        # Assuming images are png/jpg. Change extensions if needed.
        #print(img_dir + "/noisy/")
        self.noisy_files = sorted(glob.glob(os.path.join(img_dir + "/noisy/", '*.*')))
        self.gt_files = sorted(glob.glob(os.path.join(img_dir + "/gt/", '*.*')))
        # Filter only images
        self.gt_files = [f for f in self.gt_files if f.lower().endswith(('.png', '.jpg', '.jpeg','.tif', '.tiff', '.bmp'))]
        self.noisy_files = [f for f in self.noisy_files if f.lower().endswith(('.png', '.jpg', '.jpeg','.tif', '.tiff', '.bmp'))]
        
        self.img_size = img_size
        self.L = noise_level  # Number of looks (Shape parameter)
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            #transforms.ToTensor(), # [0, 1]
        ])

        print(f"There are {len(self.gt_files)} files in the dataset.")

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, idx):
        gt_path = self.gt_files[idx]
        # Open as grayscale for SAR/Ultrasound
        clean_img = torch.Tensor(np.array(Image.open(gt_path).convert('L')) / 255.0)
        
        # Convert HWC to CHW format
        #if clean_img.dim() == 2:
        clean_img = clean_img.unsqueeze(0)  # Add channel dimension (H,W) -> (1,H,W)
        # elif clean_img.dim() == 3:
        #     clean_img = clean_img.permute(2, 0, 1)  # Convert HWC to CHW (H,W,C) -> (C,H,W)
        
        #clean = self.transform(clean_img)
        clean = clean_img ** 2  # Intensity not amplitude
        noisy_path = self.noisy_files[idx]
        # Open as grayscale for SAR/Ultrasound
        noisy_img = torch.Tensor(np.array(Image.open(noisy_path).convert('L')) / 255.0)
        
        # Convert HWC to CHW format
        # if noisy_img.dim() == 2:
        noisy_img = noisy_img.unsqueeze(0)  # Add channel dimension (H,W) -> (1,H,W)
        # elif noisy_img.dim() == 3:
        #     noisy_img = noisy_img.permute(2, 0, 1)  # Convert HWC to CHW (H,W,C) -> (C,H,W)
        
        #noisy = self.transform(noisy_img)
        noisy = noisy_img ** 2  # Intensity not amplitude

        return noisy, clean, 0.0

class SpeckleRealCachedDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        img_size: int = 256,
        mode: str = "train",
        noise_level: float = 1.0,
    ):
        super().__init__()

        self.gt_files: List[str] = sorted(
            glob.glob(os.path.join(img_dir, "gt", "*.*"))
        )
        self.noisy_files: List[str] = sorted(
            glob.glob(os.path.join(img_dir, "noisy", "*.*"))
        )

        img_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        self.gt_files = [f for f in self.gt_files if f.lower().endswith(img_exts)]
        self.noisy_files = [f for f in self.noisy_files if f.lower().endswith(img_exts)]

        self.img_size = img_size
        self.L = noise_level
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
            ]
        )

        self.gt_cache: List[torch.Tensor] = []
        self.noisy_cache: List[torch.Tensor] = []

        def _load_and_preprocess(path: str) -> torch.Tensor:
            img = Image.open(path).convert("L")
            img = self.transform(img)
            img_np = (1 + np.array(img, dtype=np.float32)) / 256.0
            return torch.from_numpy(img_np).unsqueeze(0)

        for gt_path in self.gt_files:
            self.gt_cache.append(_load_and_preprocess(gt_path))

        for noisy_path in self.noisy_files:
            self.noisy_cache.append(_load_and_preprocess(noisy_path))

        total_gb = self._total_mem_footprint()
        print(
            f"[SpeckleRealDataset] {len(self.gt_files)} GT files + "
            f"{len(self.noisy_files)} noisy files cached → "
            f"{total_gb:.2f} GB in RAM."
        )

    def _total_mem_footprint(self) -> float:
        total_bytes = 0
        for cache in (self.gt_cache, self.noisy_cache):
            total_bytes += sum(t.numel() * t.element_size() for t in cache)
        return total_bytes / (1024 ** 3)

    def __len__(self) -> int:
        return len(self.gt_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        clean_img = self.gt_cache[idx]
        clean = clean_img ** 2

        noisy_img = self.noisy_cache[idx]
        noisy = noisy_img ** 2

        return noisy, clean, 0.0
# ------------------------------------------------------------
class SynthMultiLookLogDataset(Dataset):
    """
    Synthetic SAR‑style dataset **in the log domain**.

    Parameters
    ----------
    data_folder : str
        Folder that contains the source ``.png`` images (grayscale).
    crop_size : Tuple[int, int], default (256, 256)
        Height × width of the returned crops.
    samples_per_epoch : int, default 128
        Length of the virtual dataset (how many crops are sampled per epoch).
    gamma_shape : float, default 1.0
        Shape (α) of the Gamma distribution.  Scale (β) is set equal to α
        so that the noise has unit mean (E[noise]=1) and variance 1/α.
    isval : bool, default False
        If ``True`` the dataset returns *the same* noisy crop for a given
        ``idx`` (deterministic validation set).  If ``False`` the crop
        and the noise are drawn anew at every call (training mode).
    val_seed : int, default 21
        Base seed for the deterministic validation sampling.
    zero_mean : bool, default True
        If ``True`` the additive log‑noise is shifted by –E[log N] so that it
        has zero mean (for L=1 the shift is +γ ≈ 0.5772).
    normalise : bool, default True
        If ``True`` the log‑noise (after the optional zero‑mean shift) is
        divided by its standard deviation σ_η = sqrt(ψ₁(L)).  For L=1
        σ_η ≈ 1.28255.
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        data_folder: str,
        crop_size: Tuple[int, int] = (256, 256),
        samples_per_epoch: int = 128,
        gamma_shape: float = 1.0,
        isval: bool = False,
        val_seed: int = 21,
        zero_mean: bool = False,
        normalise: bool = False,
    ) -> None:

        # --------------------------------------------------------------
        # Store configuration
        # --------------------------------------------------------------
        self.crop_h, self.crop_w = crop_size
        self.samples_per_epoch = int(samples_per_epoch)
        self.gamma_shape = float(gamma_shape)
        self.isval = bool(isval)
        self.val_seed = int(val_seed)
        self.zero_mean = bool(zero_mean)
        self.normalise = bool(normalise)

        # --------------------------------------------------------------
        # Load all source images into RAM (float32, normalised, squared)
        # --------------------------------------------------------------
        self.images: List[np.ndarray] = []
        file_list = glob.glob(os.path.join(data_folder, "*.png"))
        if not file_list:
            raise RuntimeError(f"No .png files found in {data_folder}")

        print(f"Loading {len(file_list)} large images into RAM …")
        for f in file_list:
            img = Image.open(f).convert("L")                     # grayscale
            img = (np.array(img)).astype(np.float32) #/ 255.0      # [0, 1]
            img  = img  + 1.0
            #img = img ** 2.0                                     # intensity model

            # ensure explicit channel dimension (H, W, 1)
            if img.ndim == 2:
                img = img[:, :, np.newaxis]

            h, w, _ = img.shape
            if h < self.crop_h or w < self.crop_w:
                print(f"Warning: Image {f} is smaller than crop size – skipping.")
                continue
            self.images.append(img)

        if not self.images:
            raise RuntimeError("No valid images loaded (check dimensions).")
        print("Data loading complete.")

        # --------------------------------------------------------------
        # Pre‑compute a full‑size Gamma noise map for every image **once**
        # when we are in validation mode.  Later we just crop the same
        # slice, guaranteeing deterministic behaviour.
        # --------------------------------------------------------------
        self.noise_maps: Optional[List[np.ndarray]] = None
        if self.isval:
            self._precompute_noise_maps()

        # --------------------------------------------------------------
        # Simple transform – we already have a numpy array, we only need
        # to turn it into a torch Tensor.
        # --------------------------------------------------------------
        self.transform = transforms.Compose([transforms.ToTensor()])

        # --------------------------------------------------------------
        # Pre‑compute constants that appear in the log‑noise statistics.
        # For a Gamma(shape=L, scale=1/L) distribution:
        #   μ_η = ψ(L) - log L
        #   σ_η² = ψ₁(L)
        # where ψ is digamma and ψ₁ trigamma.
        # --------------------------------------------------------------
        L = self.gamma_shape
        self._log_noise_mean = float(torch.digamma(torch.tensor(L)) - math.log(L))
        self._log_noise_std  = float(torch.polygamma(1, torch.tensor(L)).sqrt())

    # ------------------------------------------------------------------
    def _deterministic_crop_coords(
        self, idx: int, img_h: int, img_w: int
    ) -> Tuple[int, int]:
        """Return (top, left) that is reproducible for a given ``idx``."""
        base = self.val_seed + idx
        top = (base * 12345) % (img_h - self.crop_h + 1)
        left = (base * 67890) % (img_w - self.crop_w + 1)
        return int(top), int(left)

    # ------------------------------------------------------------------
    def _precompute_noise_maps(self) -> None:
        """Create a full‑size Gamma noise tensor for every cached image."""
        self.noise_maps = []
        alpha = torch.tensor(self.gamma_shape, dtype=torch.float32)
        beta = torch.tensor(self.gamma_shape, dtype=torch.float32)   # scale = α
        gamma_dist = torch.distributions.Gamma(alpha, beta)

        for img in self.images:
            h, w, c = img.shape
            # Sample on CPU – later we will move to the appropriate device.
            noise_np = gamma_dist.sample((h, w, c)).squeeze(-1).numpy()
            self.noise_maps.append(noise_np.astype(np.float32))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.samples_per_epoch

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Returns
        -------
        noisy_log : torch.Tensor
            Shape (C, H, W) – ``log(noisy)`` (additive speckle + optional
            zero‑mean / variance normalisation).
        clean_log : torch.Tensor
            Shape (C, H, W) – ``log(clean)`` (ground‑truth).
        scale : float
            The factor by which the additive noise was divided
            (``σ_η`` if ``normalise=True``, otherwise ``1.0``).  You can
            use it to re‑scale the loss if you wish.
        """

        # ----------------------------------------------------------
        # Choose which source image we work with (wrap around)
        # ----------------------------------------------------------
        img_idx = idx % len(self.images)
        clean_np = self.images[img_idx]                 # (H, W, C)
        h, w, c = clean_np.shape

        # ----------------------------------------------------------
        # Crop – random for training, deterministic for validation
        # ----------------------------------------------------------
        if self.isval:
            top, left = self._deterministic_crop_coords(idx, h, w)
        else:
            top = np.random.randint(0, h - self.crop_h + 1)
            left = np.random.randint(0, w - self.crop_w + 1)

        clean_crop = clean_np[
            top : top + self.crop_h,
            left : left + self.crop_w,
            :
        ]                                                # (crop_h, crop_w, C)

        # ----------------------------------------------------------
        # Convert clean crop to torch Tensor (C, H, W)
        # ----------------------------------------------------------
        clean_tensor = (
            torch.from_numpy(clean_crop)
            .permute(2, 0, 1)          # (C, H, W)
            .contiguous()
            .float()
        )

        # ----------------------------------------------------------
        # -----------------------------------------------------------------
        #   ----------   NOISE (Gamma)   ----------
        # -----------------------------------------------------------------
        # ----------------------------------------------------------
        if self.isval:
            # Full‑size pre‑computed noise map → slice the same crop
            noise_full = self.noise_maps[img_idx]      # (H, W, C) numpy
            noise_crop = noise_full[
                top : top + self.crop_h,
                left : left + self.crop_w]                                           # (crop_h, crop_w, C)
            noise_tensor = torch.from_numpy(noise_crop).contiguous().float()
        else:
            # Fresh Gamma noise for the exact shape of the crop
            alpha = torch.tensor(self.gamma_shape, dtype=torch.float32)
            beta = torch.tensor(self.gamma_shape, dtype=torch.float32)
            gamma_dist = torch.distributions.Gamma(alpha, beta)
            # ``sample`` returns (C, H, W) because we give the exact shape
            noise_tensor = gamma_dist.sample(clean_tensor.shape).squeeze(-1)

        # ----------------------------------------------------------
        # Build the *multiplicative* noisy observation first
        # ----------------------------------------------------------
        noisy_tensor = clean_tensor * noise_tensor      # Y = X·N

        # ----------------------------------------------------------
        # -------------------  LOG TRANSFORM  -------------------------
        # ----------------------------------------------------------
        # 1) log‑clean  (ground truth)
        log_clean = torch.log(clean_tensor.clamp(min=1e-12))

        # 2) log‑noisy  = log X + log N
        #    (torch.log is numerically safe for the tiny positive values)
        log_noisy = torch.log(noisy_tensor.clamp(min=1e-12))

        # ----------------------------------------------------------
        # 3) OPTIONAL zero‑mean shift  (add +γ for L=1)
        # ----------------------------------------------------------
        scale = 1.0                     # will be returned for convenience
        if self.zero_mean:
            # We want η' = log N – μ_η   (zero‑mean)
            # μ_η = ψ(L) – log L   (pre‑computed as self._log_noise_mean)
            # Hence we add –μ_η to the whole log image.
            log_clean = log_clean - self._log_noise_mean
            log_noisy = log_noisy - self._log_noise_mean

        # ----------------------------------------------------------
        # 4) OPTIONAL variance normalisation
        # ----------------------------------------------------------
        if self.normalise:
            log_clean = log_clean / self._log_noise_std
            log_noisy = log_noisy / self._log_noise_std
            scale = self._log_noise_std          # remember the divisor

        # ----------------------------------------------------------
        # Return tensors in (C, H, W) format together with the scale
        # ----------------------------------------------------------
        return log_noisy, log_clean, scale