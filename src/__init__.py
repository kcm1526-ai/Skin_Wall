"""
Skin and Abdominal Wall Segmentation Package
"""

from .dataset import SkinWallDataset, create_dataloaders
from .losses import get_loss_function, DiceMetric
from .models import create_model, get_model_names
from .train import Trainer
from .inference import Predictor, BatchPredictor
from .utils import compute_all_metrics, set_seed

__version__ = '1.0.0'
__author__ = 'AI Medical Imaging Team'

__all__ = [
    'SkinWallDataset',
    'create_dataloaders',
    'get_loss_function',
    'DiceMetric',
    'create_model',
    'get_model_names',
    'Trainer',
    'Predictor',
    'BatchPredictor',
    'compute_all_metrics',
    'set_seed'
]
