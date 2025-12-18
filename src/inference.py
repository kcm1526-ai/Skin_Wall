"""
Inference script for Skin and Abdominal Wall Segmentation
Supports sliding window inference and test-time augmentation
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import nibabel as nib
from scipy import ndimage

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import Config, get_config
from src.dataset import DICOMLoader, MaskLoader
from src.models import create_model
from src.losses import DiceMetric

# MONAI imports
try:
    from monai.inferers import sliding_window_inference
    from monai.transforms import AsDiscrete
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print("Warning: MONAI not available. Using basic inference.")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Predictor:
    """
    Predictor class for inference on new data
    """

    def __init__(
        self,
        checkpoint_path: str,
        config: Optional[Config] = None,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            config: Configuration (loaded from checkpoint if not provided)
            device: Device to use for inference
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load checkpoint
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Get config
        if config is None:
            config = self.checkpoint.get('config', get_config())
        self.config = config

        # Load model
        self._load_model()

        logger.info(f"Model loaded from {checkpoint_path}")
        logger.info(f"Device: {self.device}")

    def _load_model(self):
        """Load model from checkpoint"""
        self.model = create_model(
            model_type=self.config.model.model_type,
            in_channels=self.config.model.in_channels,
            out_channels=self.config.model.out_channels,
            img_size=self.config.preprocess.patch_size,
            use_deep_supervision=False  # Disable for inference
        ).to(self.device)

        # Load weights
        state_dict = self.checkpoint['model_state_dict']

        # Handle DataParallel wrapper
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for inference

        Args:
            image: Input image array of shape (D, H, W)

        Returns:
            Preprocessed tensor of shape (1, 1, D, H, W)
        """
        # Clip intensity
        clip_min, clip_max = self.config.preprocess.clip_values
        image = np.clip(image, clip_min, clip_max)

        # Normalize
        if self.config.preprocess.normalize_method == 'zscore':
            mean = image.mean()
            std = image.std()
            image = (image - mean) / (std + 1e-8)
        elif self.config.preprocess.normalize_method == 'minmax':
            image = (image - clip_min) / (clip_max - clip_min)

        # Add batch and channel dimensions
        image = image[np.newaxis, np.newaxis, ...]

        return torch.from_numpy(image.astype(np.float32)).to(self.device)

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        use_tta: bool = True,
        return_probabilities: bool = False
    ) -> np.ndarray:
        """
        Run prediction on an image

        Args:
            image: Input image array of shape (D, H, W)
            use_tta: Whether to use test-time augmentation
            return_probabilities: Whether to return probability maps

        Returns:
            Predicted segmentation mask or probability maps
        """
        # Preprocess
        x = self.preprocess(image)

        # Run inference
        if MONAI_AVAILABLE:
            # Sliding window inference
            output = sliding_window_inference(
                x,
                roi_size=self.config.preprocess.patch_size,
                sw_batch_size=self.config.inference.sw_batch_size,
                predictor=self.model,
                overlap=self.config.inference.sw_overlap,
                mode='gaussian'
            )
        else:
            output = self._basic_inference(x)

        # Test-time augmentation
        if use_tta:
            output = self._apply_tta(x, output)

        # Post-process
        if return_probabilities:
            probs = F.softmax(output, dim=1).cpu().numpy()[0]
            return probs
        else:
            pred = output.argmax(dim=1).cpu().numpy()[0]
            if self.config.inference.use_postprocessing:
                pred = self.postprocess(pred)
            return pred

    def _basic_inference(self, x: torch.Tensor) -> torch.Tensor:
        """Basic inference without sliding window"""
        # Pad to patch size if needed
        patch_size = self.config.preprocess.patch_size
        pad_d = (patch_size[0] - x.shape[2] % patch_size[0]) % patch_size[0]
        pad_h = (patch_size[1] - x.shape[3] % patch_size[1]) % patch_size[1]
        pad_w = (patch_size[2] - x.shape[4] % patch_size[2]) % patch_size[2]

        x_padded = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

        output = self.model(x_padded)

        # Handle deep supervision output
        if isinstance(output, tuple):
            output = output[0]

        # Remove padding
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            output = output[:, :, :x.shape[2], :x.shape[3], :x.shape[4]]

        return output

    def _apply_tta(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Apply test-time augmentation"""
        outputs = [output]

        # Flip augmentations
        for axis in self.config.inference.tta_flips:
            x_flipped = torch.flip(x, [axis])

            if MONAI_AVAILABLE:
                out_flipped = sliding_window_inference(
                    x_flipped,
                    roi_size=self.config.preprocess.patch_size,
                    sw_batch_size=self.config.inference.sw_batch_size,
                    predictor=self.model,
                    overlap=self.config.inference.sw_overlap,
                    mode='gaussian'
                )
            else:
                out_flipped = self._basic_inference(x_flipped)

            # Flip back
            out_flipped = torch.flip(out_flipped, [axis])
            outputs.append(out_flipped)

        # Average predictions
        output = torch.stack(outputs).mean(dim=0)
        return output

    def postprocess(self, pred: np.ndarray) -> np.ndarray:
        """
        Post-process prediction

        Args:
            pred: Predicted segmentation mask

        Returns:
            Post-processed mask
        """
        # Keep only largest connected component for each class
        processed = np.zeros_like(pred)

        for class_id in range(1, self.config.data.num_classes):
            class_mask = (pred == class_id).astype(np.uint8)

            # Find connected components
            labeled, num_features = ndimage.label(class_mask)

            if num_features > 0:
                # Get component sizes
                component_sizes = ndimage.sum(class_mask, labeled, range(1, num_features + 1))

                # Keep components larger than threshold
                for i, size in enumerate(component_sizes):
                    if size >= self.config.inference.min_component_size:
                        processed[labeled == (i + 1)] = class_id

        return processed

    def predict_from_dicom(
        self,
        dicom_dir: str,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Run prediction on DICOM series

        Args:
            dicom_dir: Directory containing DICOM files
            output_path: Optional path to save predictions

        Returns:
            Dictionary with predictions and metadata
        """
        logger.info(f"Loading DICOM from {dicom_dir}")

        # Load DICOM
        image, metadata = DICOMLoader.load_dicom_series(dicom_dir)
        logger.info(f"Image shape: {image.shape}")

        # Predict
        pred = self.predict(image, use_tta=self.config.inference.use_tta)

        result = {
            'prediction': pred,
            'metadata': metadata,
            'image_shape': image.shape
        }

        # Save if output path provided
        if output_path:
            self.save_prediction(pred, metadata, output_path)

        return result

    def save_prediction(
        self,
        pred: np.ndarray,
        metadata: Dict,
        output_path: str,
        separate_classes: bool = True
    ):
        """
        Save prediction as NIfTI files

        Args:
            pred: Predicted segmentation mask
            metadata: Image metadata (spacing, etc.)
            output_path: Output directory or file path
            separate_classes: Whether to save each class separately
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create affine from metadata
        spacing = metadata.get('spacing', (1.0, 1.0, 1.0))
        affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])

        if separate_classes:
            # Save each class separately
            class_names = self.config.data.class_names

            for class_id in range(1, self.config.data.num_classes):
                class_mask = (pred == class_id).astype(np.uint8)
                class_name = class_names[class_id] if class_id < len(class_names) else f'class_{class_id}'

                nii = nib.Nifti1Image(class_mask, affine)
                nib.save(nii, output_dir / f'{class_name}.nii.gz')

                logger.info(f"Saved {class_name} mask")
        else:
            # Save combined mask
            nii = nib.Nifti1Image(pred.astype(np.uint8), affine)
            nib.save(nii, output_dir / 'segmentation.nii.gz')

        logger.info(f"Predictions saved to {output_dir}")


class BatchPredictor:
    """
    Batch predictor for running inference on multiple subjects
    """

    def __init__(
        self,
        checkpoint_path: str,
        config: Optional[Config] = None
    ):
        self.predictor = Predictor(checkpoint_path, config)
        self.config = self.predictor.config

    def predict_dataset(
        self,
        data_dir: str,
        output_dir: str,
        evaluate: bool = True
    ) -> Optional[Dict]:
        """
        Run prediction on a dataset

        Args:
            data_dir: Base directory containing subject folders
            output_dir: Output directory for predictions
            evaluate: Whether to compute metrics (requires ground truth)

        Returns:
            Evaluation metrics if evaluate=True
        """
        from src.dataset import find_data_paths

        # Find all subjects
        samples = find_data_paths(data_dir, self.config)
        logger.info(f"Found {len(samples)} subjects")

        if evaluate:
            dice_metric = DiceMetric(
                num_classes=self.config.data.num_classes,
                include_background=False
            )

        results = []

        for sample in tqdm(samples, desc='Processing subjects'):
            subject_id = sample['subject_id']
            logger.info(f"Processing {subject_id}")

            try:
                # Predict
                result = self.predictor.predict_from_dicom(
                    sample['image_dir'],
                    output_path=os.path.join(output_dir, subject_id)
                )

                # Evaluate if ground truth available
                if evaluate:
                    # Load ground truth
                    skin_mask, _ = MaskLoader.load_nifti_mask(sample['skin_mask'])
                    abdominal_mask, _ = MaskLoader.load_nifti_mask(sample['abdominal_wall_mask'])

                    # Combine masks
                    gt = np.zeros_like(result['prediction'])
                    gt[skin_mask > 0] = 1
                    gt[abdominal_mask > 0] = 2

                    # Compute dice
                    pred_tensor = torch.from_numpy(result['prediction']).unsqueeze(0)
                    gt_tensor = torch.from_numpy(gt).unsqueeze(0)
                    dice_metric.update(pred_tensor, gt_tensor)

                    results.append({
                        'subject_id': subject_id,
                        'status': 'success'
                    })

            except Exception as e:
                logger.error(f"Error processing {subject_id}: {e}")
                results.append({
                    'subject_id': subject_id,
                    'status': 'error',
                    'error': str(e)
                })

        # Save results summary
        summary_path = os.path.join(output_dir, 'results_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)

        if evaluate:
            metrics = dice_metric.compute()
            logger.info(f"Overall metrics: {metrics}")

            metrics_path = os.path.join(output_dir, 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)

            return metrics

        return None


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run inference')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Input DICOM directory or base data directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--batch', action='store_true',
                        help='Run batch prediction on multiple subjects')
    parser.add_argument('--evaluate', action='store_true',
                        help='Compute metrics (requires ground truth)')
    parser.add_argument('--no-tta', action='store_true',
                        help='Disable test-time augmentation')
    parser.add_argument('--gpu', type=str, default='2',
                        help='GPU device ID (default: 2)')

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.batch:
        # Batch prediction
        predictor = BatchPredictor(args.checkpoint)
        predictor.predict_dataset(
            args.input,
            args.output,
            evaluate=args.evaluate
        )
    else:
        # Single subject prediction
        predictor = Predictor(args.checkpoint)

        # Override TTA setting
        if args.no_tta:
            predictor.config.inference.use_tta = False

        predictor.predict_from_dicom(args.input, args.output)


if __name__ == '__main__':
    main()
