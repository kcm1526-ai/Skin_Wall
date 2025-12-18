"""
Dataset module for Skin and Abdominal Wall Segmentation
Handles DICOM loading, NIfTI mask loading, preprocessing, and augmentation
"""

import os
import glob
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import pydicom
from scipy import ndimage
from scipy.ndimage import zoom
import SimpleITK as sitk
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, RandShiftIntensityd, RandScaleIntensityd,
    RandGaussianNoised, RandGaussianSmoothd, RandAffined, ToTensord,
    NormalizeIntensityd, EnsureTyped, ConcatItemsd, DeleteItemsd,
    RandSpatialCropd, SpatialPadd, CenterSpatialCropd
)
from monai.data import CacheDataset, SmartCacheDataset, list_data_collate
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DICOMLoader:
    """Load DICOM series and convert to 3D volume"""

    @staticmethod
    def load_dicom_series(dicom_dir: str) -> Tuple[np.ndarray, Dict]:
        """
        Load DICOM series from directory

        Args:
            dicom_dir: Path to directory containing DICOM files

        Returns:
            Tuple of (3D volume array, metadata dict)
        """
        # Find all DICOM files (with or without .dcm extension)
        dicom_files = []
        for f in os.listdir(dicom_dir):
            file_path = os.path.join(dicom_dir, f)
            if os.path.isfile(file_path):
                # Skip known non-DICOM files
                if f.endswith(('.nii', '.nii.gz', '.json', '.txt', '.xml')):
                    continue
                # Check if it's a DICOM file
                try:
                    # Use force=True to read files without .dcm extension or DICOM preamble
                    dcm = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
                    # Check for essential DICOM attributes (PixelData may not show with stop_before_pixels)
                    if hasattr(dcm, 'SOPClassUID') or hasattr(dcm, 'Modality') or hasattr(dcm, 'PixelData'):
                        dicom_files.append(file_path)
                except Exception:
                    # Try loading with SimpleITK as fallback
                    try:
                        reader = sitk.ImageFileReader()
                        reader.SetFileName(file_path)
                        reader.ReadImageInformation()
                        dicom_files.append(file_path)
                    except Exception:
                        continue

        if not dicom_files:
            raise ValueError(f"No DICOM files found in {dicom_dir}")

        # Use SimpleITK for robust DICOM series reading
        reader = sitk.ImageSeriesReader()

        # Try to get series IDs
        series_ids = reader.GetGDCMSeriesIDs(dicom_dir)

        if series_ids:
            # Use the first series
            series_file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
            reader.SetFileNames(series_file_names)
        else:
            # Fallback: sort files and use directly
            dicom_files.sort()
            reader.SetFileNames(dicom_files)

        try:
            image = reader.Execute()
        except:
            # Alternative: Load slice by slice using pydicom
            return DICOMLoader._load_dicom_pydicom(dicom_files)

        # Extract metadata
        metadata = {
            'spacing': image.GetSpacing(),
            'origin': image.GetOrigin(),
            'direction': image.GetDirection(),
            'size': image.GetSize()
        }

        # Convert to numpy array
        volume = sitk.GetArrayFromImage(image)  # Shape: (Z, Y, X)

        return volume, metadata

    @staticmethod
    def _load_dicom_pydicom(dicom_files: List[str]) -> Tuple[np.ndarray, Dict]:
        """Fallback DICOM loading using pydicom"""
        slices = []
        for f in dicom_files:
            try:
                dcm = pydicom.dcmread(f)
                if hasattr(dcm, 'pixel_array'):
                    slices.append(dcm)
            except:
                continue

        if not slices:
            raise ValueError("Could not load any DICOM slices")

        # Sort by slice location or instance number
        try:
            slices.sort(key=lambda x: float(x.SliceLocation))
        except:
            try:
                slices.sort(key=lambda x: int(x.InstanceNumber))
            except:
                pass

        # Get pixel spacing
        try:
            pixel_spacing = slices[0].PixelSpacing
            slice_thickness = float(slices[0].SliceThickness) if hasattr(slices[0], 'SliceThickness') else 1.0
            spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]), slice_thickness)
        except:
            spacing = (1.0, 1.0, 1.0)

        # Stack slices
        volume = np.stack([s.pixel_array for s in slices], axis=0)

        # Apply rescale slope/intercept for HU conversion
        try:
            slope = float(slices[0].RescaleSlope)
            intercept = float(slices[0].RescaleIntercept)
            volume = volume * slope + intercept
        except:
            pass

        metadata = {
            'spacing': spacing,
            'origin': (0, 0, 0),
            'direction': (1, 0, 0, 0, 1, 0, 0, 0, 1),
            'size': volume.shape[::-1]
        }

        return volume.astype(np.float32), metadata


class MaskLoader:
    """Load NIfTI masks"""

    @staticmethod
    def load_nifti_mask(mask_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Load NIfTI mask file

        Args:
            mask_path: Path to NIfTI file (.nii or .nii.gz)

        Returns:
            Tuple of (3D mask array, metadata dict)
        """
        nii = nib.load(mask_path)
        mask = nii.get_fdata().astype(np.uint8)

        # Get affine for spacing information
        affine = nii.affine
        spacing = tuple(np.abs(np.diag(affine)[:3]))

        metadata = {
            'spacing': spacing,
            'affine': affine,
            'shape': mask.shape
        }

        return mask, metadata


def find_data_paths(base_path: str, config) -> List[Dict]:
    """
    Find all valid data samples with images and masks

    Args:
        base_path: Base directory containing subject folders
        config: Configuration object

    Returns:
        List of dicts with 'subject_id', 'image_dir', 'skin_mask', 'abdominal_wall_mask' paths
    """
    samples = []

    # Find all subject folders
    subject_pattern = os.path.join(base_path, "*")
    subject_dirs = sorted(glob.glob(subject_pattern))

    for subject_dir in subject_dirs:
        if not os.path.isdir(subject_dir):
            continue

        subject_id = os.path.basename(subject_dir)

        # Find image directory
        image_dir = os.path.join(subject_dir, config.data.image_subpath)
        if not os.path.isdir(image_dir):
            logger.warning(f"Image directory not found for {subject_id}: {image_dir}")
            continue

        # Find masks - try primary path first
        skin_mask = None
        abdominal_wall_mask = None

        # Primary mask path
        primary_mask_dir = os.path.join(subject_dir, config.data.primary_mask_subpath)
        skin_primary = os.path.join(primary_mask_dir, config.data.skin_mask_name)
        abdominal_primary = os.path.join(primary_mask_dir, config.data.abdominal_wall_mask_name)

        # Alternative mask path
        alt_mask_dir = os.path.join(subject_dir, config.data.alt_mask_subpath)
        skin_alt = os.path.join(alt_mask_dir, config.data.skin_mask_name)
        abdominal_alt = os.path.join(alt_mask_dir, config.data.abdominal_wall_mask_name)
        # Handle typo in alternative path
        abdominal_alt_typo = os.path.join(alt_mask_dir, config.data.alt_abdominal_wall_mask_name)

        # Check skin mask
        if os.path.exists(skin_primary):
            skin_mask = skin_primary
        elif os.path.exists(skin_alt):
            skin_mask = skin_alt

        # Check abdominal wall mask
        if os.path.exists(abdominal_primary):
            abdominal_wall_mask = abdominal_primary
        elif os.path.exists(abdominal_alt):
            abdominal_wall_mask = abdominal_alt
        elif os.path.exists(abdominal_alt_typo):
            abdominal_wall_mask = abdominal_alt_typo

        # Only include if both masks are found
        if skin_mask and abdominal_wall_mask:
            samples.append({
                'subject_id': subject_id,
                'image_dir': image_dir,
                'skin_mask': skin_mask,
                'abdominal_wall_mask': abdominal_wall_mask
            })
            logger.info(f"Found valid sample: {subject_id}")
        else:
            logger.warning(f"Missing masks for {subject_id}. Skin: {skin_mask is not None}, Abdominal: {abdominal_wall_mask is not None}")

    logger.info(f"Total valid samples found: {len(samples)}")
    return samples


def prepare_data_dict(samples: List[Dict]) -> List[Dict]:
    """
    Prepare data dictionaries for MONAI transforms

    Args:
        samples: List of sample dicts

    Returns:
        List of dicts with 'image', 'label' keys
    """
    data_dicts = []

    for sample in samples:
        data_dicts.append({
            'subject_id': sample['subject_id'],
            'image_dir': sample['image_dir'],
            'skin_mask': sample['skin_mask'],
            'abdominal_wall_mask': sample['abdominal_wall_mask']
        })

    return data_dicts


class SkinWallDataset(Dataset):
    """
    Custom dataset for Skin and Abdominal Wall segmentation
    """

    def __init__(
        self,
        data_list: List[Dict],
        config,
        mode: str = 'train',
        transform=None,
        cache: bool = True
    ):
        """
        Args:
            data_list: List of data dictionaries
            config: Configuration object
            mode: 'train', 'val', or 'test'
            transform: Optional custom transforms
            cache: Whether to cache loaded data
        """
        self.data_list = data_list
        self.config = config
        self.mode = mode
        self.transform = transform
        self.cache = cache
        self.cached_data = {}

        logger.info(f"Initialized {mode} dataset with {len(data_list)} samples")

    def __len__(self) -> int:
        return len(self.data_list)

    def _load_sample(self, idx: int) -> Dict:
        """Load and preprocess a single sample"""
        sample = self.data_list[idx]

        # Load DICOM images
        try:
            image, image_meta = DICOMLoader.load_dicom_series(sample['image_dir'])
        except Exception as e:
            logger.error(f"Error loading DICOM for {sample['subject_id']}: {e}")
            raise

        # Load masks
        try:
            skin_mask, _ = MaskLoader.load_nifti_mask(sample['skin_mask'])
            abdominal_mask, _ = MaskLoader.load_nifti_mask(sample['abdominal_wall_mask'])
        except Exception as e:
            logger.error(f"Error loading masks for {sample['subject_id']}: {e}")
            raise

        # Ensure masks have same shape as image
        # Masks might need to be transposed or resampled
        if skin_mask.shape != image.shape:
            # Try to match dimensions
            skin_mask = self._align_mask_to_image(skin_mask, image.shape)
        if abdominal_mask.shape != image.shape:
            abdominal_mask = self._align_mask_to_image(abdominal_mask, image.shape)

        # Combine masks: 0=background, 1=skin, 2=abdominal_wall
        combined_mask = np.zeros_like(image, dtype=np.uint8)
        combined_mask[skin_mask > 0] = 1
        combined_mask[abdominal_mask > 0] = 2

        return {
            'image': image,
            'label': combined_mask,
            'spacing': image_meta['spacing'],
            'subject_id': sample['subject_id']
        }

    def _align_mask_to_image(self, mask: np.ndarray, target_shape: Tuple) -> np.ndarray:
        """Align mask to image shape through resampling"""
        if mask.shape == target_shape:
            return mask

        # Calculate zoom factors
        zoom_factors = [t / s for t, s in zip(target_shape, mask.shape)]

        # Use nearest neighbor for masks
        resampled_mask = zoom(mask.astype(np.float32), zoom_factors, order=0)

        return resampled_mask.astype(np.uint8)

    def __getitem__(self, idx: int) -> Dict:
        # Check cache
        if self.cache and idx in self.cached_data:
            data = self.cached_data[idx].copy()
        else:
            data = self._load_sample(idx)
            if self.cache:
                self.cached_data[idx] = data.copy()

        # Preprocess
        data = self._preprocess(data)

        # Apply transforms
        if self.transform:
            data = self.transform(data)

        return data

    def _preprocess(self, data: Dict) -> Dict:
        """Apply preprocessing"""
        image = data['image']
        label = data['label']

        # Clip intensity values
        clip_min, clip_max = self.config.preprocess.clip_values
        image = np.clip(image, clip_min, clip_max)

        # Normalize
        if self.config.preprocess.normalize_method == 'zscore':
            mean = image.mean()
            std = image.std()
            image = (image - mean) / (std + 1e-8)
        elif self.config.preprocess.normalize_method == 'minmax':
            image = (image - clip_min) / (clip_max - clip_min)

        # Add channel dimension
        image = image[np.newaxis, ...].astype(np.float32)  # (1, D, H, W)
        label = label[np.newaxis, ...].astype(np.int64)  # (1, D, H, W)

        data['image'] = image
        data['label'] = label

        return data


def get_transforms(config, mode: str = 'train'):
    """
    Get MONAI transforms for training/validation/test

    Args:
        config: Configuration object
        mode: 'train', 'val', or 'test'

    Returns:
        Composed transforms
    """
    patch_size = config.preprocess.patch_size

    if mode == 'train':
        transforms = Compose([
            # Ensure tensor type
            EnsureTyped(keys=['image', 'label']),

            # Random cropping with positive/negative sampling
            RandCropByPosNegLabeld(
                keys=['image', 'label'],
                label_key='label',
                spatial_size=patch_size,
                pos=2,  # 2 positive samples
                neg=1,  # 1 negative sample
                num_samples=4,  # 4 patches per volume
                image_key='image',
                image_threshold=0,
            ),

            # Spatial augmentations
            RandFlipd(keys=['image', 'label'], prob=config.augment.random_flip_prob, spatial_axis=0),
            RandFlipd(keys=['image', 'label'], prob=config.augment.random_flip_prob, spatial_axis=1),
            RandFlipd(keys=['image', 'label'], prob=config.augment.random_flip_prob, spatial_axis=2),

            RandRotate90d(keys=['image', 'label'], prob=config.augment.random_rotate_prob, max_k=3),

            # Affine transformations
            RandAffined(
                keys=['image', 'label'],
                mode=['bilinear', 'nearest'],
                prob=config.augment.random_scale_prob,
                scale_range=[0.1, 0.1, 0.1],
                rotate_range=[0.26, 0.26, 0.26],  # ~15 degrees
                padding_mode='zeros'
            ),

            # Intensity augmentations
            RandShiftIntensityd(
                keys=['image'],
                offsets=config.augment.intensity_shift_range[1],
                prob=config.augment.random_intensity_shift_prob
            ),

            RandScaleIntensityd(
                keys=['image'],
                factors=config.augment.intensity_scale_range[1] - 1.0,
                prob=config.augment.random_intensity_scale_prob
            ),

            RandGaussianNoised(
                keys=['image'],
                prob=config.augment.gaussian_noise_prob,
                std=config.augment.gaussian_noise_std
            ),

            RandGaussianSmoothd(
                keys=['image'],
                prob=config.augment.gaussian_blur_prob,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0)
            ),

            # Ensure correct types
            EnsureTyped(keys=['image', 'label'], dtype=torch.float32),
        ])

    else:  # val or test
        transforms = Compose([
            EnsureTyped(keys=['image', 'label']),

            # Center crop or pad to patch size for validation
            SpatialPadd(keys=['image', 'label'], spatial_size=patch_size),

            EnsureTyped(keys=['image', 'label'], dtype=torch.float32),
        ])

    return transforms


def create_dataloaders(config, num_workers: int = 8):
    """
    Create train, validation, and test dataloaders

    Args:
        config: Configuration object
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Find all data samples
    samples = find_data_paths(config.data.base_path, config)

    if len(samples) == 0:
        raise ValueError("No valid data samples found!")

    # Split data
    train_samples, temp_samples = train_test_split(
        samples,
        test_size=(config.data.val_ratio + config.data.test_ratio),
        random_state=config.data.seed
    )

    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=config.data.test_ratio / (config.data.val_ratio + config.data.test_ratio),
        random_state=config.data.seed
    )

    logger.info(f"Data split: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")

    # Create datasets
    train_transforms = get_transforms(config, mode='train')
    val_transforms = get_transforms(config, mode='val')

    train_dataset = SkinWallDataset(
        train_samples, config, mode='train', transform=train_transforms
    )
    val_dataset = SkinWallDataset(
        val_samples, config, mode='val', transform=val_transforms
    )
    test_dataset = SkinWallDataset(
        test_samples, config, mode='test', transform=val_transforms
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Full volume for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    return train_loader, val_loader, test_loader


def custom_collate_fn(batch):
    """Custom collate function to handle variable-sized patches"""
    # Filter out None values
    batch = [b for b in batch if b is not None]

    if len(batch) == 0:
        return None

    # Check if batch contains lists (from RandCropByPosNegLabeld)
    if isinstance(batch[0].get('image'), list):
        # Flatten the list of patches
        flattened = []
        for item in batch:
            for i in range(len(item['image'])):
                flattened.append({
                    'image': item['image'][i],
                    'label': item['label'][i],
                    'subject_id': item['subject_id']
                })
        batch = flattened

    # Stack tensors
    images = torch.stack([b['image'] if isinstance(b['image'], torch.Tensor)
                         else torch.from_numpy(b['image']) for b in batch])
    labels = torch.stack([b['label'] if isinstance(b['label'], torch.Tensor)
                         else torch.from_numpy(b['label']) for b in batch])

    return {
        'image': images,
        'label': labels,
        'subject_id': [b['subject_id'] for b in batch]
    }


# Alternative: MONAI-native dataset with caching
def create_monai_dataloaders(config, num_workers: int = 8):
    """
    Create dataloaders using MONAI's CacheDataset for better performance

    Args:
        config: Configuration object
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from monai.data import CacheDataset, DataLoader

    # Find and prepare data
    samples = find_data_paths(config.data.base_path, config)

    if len(samples) == 0:
        raise ValueError("No valid data samples found!")

    # Split data
    train_samples, temp_samples = train_test_split(
        samples,
        test_size=(config.data.val_ratio + config.data.test_ratio),
        random_state=config.data.seed
    )

    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=config.data.test_ratio / (config.data.val_ratio + config.data.test_ratio),
        random_state=config.data.seed
    )

    # Prepare data dicts
    train_files = prepare_data_dict(train_samples)
    val_files = prepare_data_dict(val_samples)
    test_files = prepare_data_dict(test_samples)

    # Get transforms
    train_transforms = get_transforms(config, mode='train')
    val_transforms = get_transforms(config, mode='val')

    # Create datasets with caching
    train_ds = SkinWallDataset(train_files, config, mode='train', transform=train_transforms)
    val_ds = SkinWallDataset(val_files, config, mode='val', transform=val_transforms)
    test_ds = SkinWallDataset(test_files, config, mode='test', transform=val_transforms)

    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from configs.config import get_config

    config = get_config()

    print("Finding data samples...")
    samples = find_data_paths(config.data.base_path, config)
    print(f"Found {len(samples)} samples")

    if samples:
        print("\nTesting data loading...")
        dataset = SkinWallDataset(samples[:1], config, mode='val')
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Label shape: {sample['label'].shape}")
        print(f"Unique labels: {np.unique(sample['label'])}")
