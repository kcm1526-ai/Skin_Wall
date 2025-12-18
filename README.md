# Skin and Abdominal Wall Segmentation

Deep learning models for 3D segmentation of skin and abdominal wall from CT/MRI images.

## Features

- **Multiple State-of-the-Art Architectures**:
  - 3D UNet (baseline)
  - Residual 3D UNet with SE blocks
  - Attention 3D UNet
  - UNETR (Vision Transformer + UNet decoder)
  - Swin UNETR (State-of-the-art for medical imaging)

- **Advanced Training Techniques**:
  - Deep supervision for better gradient flow
  - Combined loss (Dice + Cross-Entropy + Focal)
  - Mixed precision training (AMP)
  - Gradient accumulation for larger effective batch sizes
  - Learning rate warmup with cosine annealing
  - Extensive data augmentation

- **Robust Inference**:
  - Sliding window inference for full volumes
  - Test-time augmentation (TTA)
  - Post-processing with connected component analysis

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Data Structure

The expected data structure is:

```
/raid/users/ai_kcm_0/skin_and_wall/01011_SEV_sale/
├── SUBJECT_001/
│   ├── 01_DICOM/
│   │   └── PP/
│   │       ├── *.dcm (or files without extension)
│   │       └── Mask/
│   │           ├── Skin.nii.gz
│   │           └── Abdominal_wall.nii.gz
│   └── 03_Vein/  (alternative mask location)
│       ├── Skin.nii.gz
│       └── Abdominal_wall.nii.gz  (or Abdonimal_wall.nii.gz)
├── SUBJECT_002/
│   └── ...
```

## Training

### Quick Start

```bash
# Train with Swin UNETR (recommended)
python -m src.train --model swin_unetr --epochs 500 --gpu 0

# Train with UNETR
python -m src.train --model unetr --epochs 500 --gpu 0

# Train with Attention UNet
python -m src.train --model attention_unet3d --epochs 500 --gpu 0
```

### Advanced Training Options

```bash
python -m src.train \
    --model swin_unetr \
    --epochs 500 \
    --batch-size 2 \
    --lr 1e-4 \
    --gpu 0,1 \
    --exp-name my_experiment \
    --output-dir ./outputs
```

### Using Shell Script

```bash
chmod +x scripts/train.sh
./scripts/train.sh --model swin_unetr --epochs 500 --gpu 0
```

## Inference

### Single Subject

```bash
python -m src.inference \
    --checkpoint outputs/best_model.pth \
    --input /path/to/dicom_folder \
    --output /path/to/output
```

### Batch Prediction

```bash
python -m src.inference \
    --checkpoint outputs/best_model.pth \
    --input /path/to/data_directory \
    --output /path/to/output \
    --batch \
    --evaluate
```

## Configuration

Configuration can be modified in `configs/config.py`. Key settings include:

### Data Configuration
```python
data:
  base_path: "/raid/users/ai_kcm_0/skin_and_wall/01011_SEV_sale"
  num_classes: 3  # Background, Skin, Abdominal Wall
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
```

### Model Configuration
```python
model:
  model_type: "swin_unetr"  # Options: unet3d, resunet3d, attention_unet3d, unetr, swin_unetr
  use_deep_supervision: true
  dropout_rate: 0.1
```

### Training Configuration
```python
train:
  batch_size: 2
  num_epochs: 500
  learning_rate: 1e-4
  loss_type: "dice_ce_focal"
  use_amp: true  # Mixed precision
  accumulation_steps: 4  # Gradient accumulation
```

## Model Architectures

### Swin UNETR (Recommended)
State-of-the-art architecture using Swin Transformer as encoder. Best for achieving high Dice scores.

```python
from src.models import create_model

model = create_model(
    model_type='swin_unetr',
    in_channels=1,
    out_channels=3,
    img_size=(96, 96, 96),
    feature_size=48
)
```

### UNETR
Uses Vision Transformer encoder with UNet-style decoder.

### Attention UNet
3D UNet with attention gates in skip connections.

## Tips for Achieving High Dice Score (>0.9)

1. **Use Swin UNETR or UNETR**: Transformer-based models generally perform better on medical imaging tasks.

2. **Enable Deep Supervision**: Helps with gradient flow and provides auxiliary supervision.

3. **Use Combined Loss**: `dice_ce_focal` loss combines Dice, Cross-Entropy, and Focal losses for robust training.

4. **Increase Training Epochs**: Medical imaging tasks often benefit from longer training (500+ epochs).

5. **Use Data Augmentation**: Extensive augmentation helps prevent overfitting.

6. **Enable Test-Time Augmentation**: Averages predictions over augmented versions for better accuracy.

7. **Use Mixed Precision Training**: Allows for larger batch sizes and faster training.

8. **Post-processing**: Remove small connected components to clean up predictions.

## Evaluation Metrics

The following metrics are computed:
- **Dice Score**: Primary metric for overlap
- **IoU (Jaccard Index)**: Intersection over Union
- **Surface Dice**: Accounts for boundary accuracy
- **Hausdorff Distance (95%)**: Maximum surface-to-surface distance
- **Precision/Recall**: For class-specific analysis

## Troubleshooting

### Out of Memory

- Reduce `batch_size` in config
- Increase `accumulation_steps` to maintain effective batch size
- Reduce `patch_size` if necessary

### Slow Training

- Enable `use_amp` for mixed precision training
- Reduce `num_workers` if CPU bound
- Use `SmartCacheDataset` for faster data loading

### Poor Dice Score

- Increase training epochs
- Try different learning rates (1e-4 to 1e-5)
- Use `swin_unetr` model
- Check data loading and preprocessing

## Project Structure

```
Skin_Wall/
├── configs/
│   ├── __init__.py
│   └── config.py           # Configuration settings
├── src/
│   ├── __init__.py
│   ├── dataset.py          # Data loading and preprocessing
│   ├── losses.py           # Loss functions
│   ├── train.py            # Training script
│   ├── inference.py        # Inference script
│   ├── utils.py            # Utility functions
│   └── models/
│       ├── __init__.py
│       ├── model_factory.py
│       ├── unet3d.py       # 3D UNet variants
│       ├── unetr.py        # UNETR model
│       └── swin_unetr.py   # Swin UNETR model
├── scripts/
│   └── train.sh            # Training shell script
├── requirements.txt
└── README.md
```

## Citation

If you use this code, please cite the relevant papers:

```bibtex
@article{hatamizadeh2021swin,
  title={Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images},
  author={Hatamizadeh, Ali and others},
  journal={arXiv preprint arXiv:2201.01266},
  year={2021}
}

@article{hatamizadeh2022unetr,
  title={UNETR: Transformers for 3D Medical Image Segmentation},
  author={Hatamizadeh, Ali and others},
  booktitle={WACV},
  year={2022}
}
```

## License

This project is for research and educational purposes.
