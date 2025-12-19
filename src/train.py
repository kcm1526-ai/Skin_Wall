"""
Training script for Skin and Abdominal Wall Segmentation
Implements best practices for achieving high Dice scores
"""

import os
import sys

# Set default GPU devices BEFORE importing torch
# This must be done before any CUDA initialization
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5'

import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import Config, get_config
from src.dataset import create_dataloaders, find_data_paths
from src.models import create_model, get_model_names
from src.losses import get_loss_function, DiceMetric, compute_class_weights

# MONAI imports for sliding window inference
try:
    from monai.inferers import sliding_window_inference
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print("Warning: MONAI not available. Sliding window inference will be limited.")


def setup_logging(output_dir: str, exp_name: str) -> logging.Logger:
    """Setup logging"""
    log_dir = os.path.join(output_dir, exp_name, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for medical image segmentation
    """

    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)

        # Setup directories
        self.exp_dir = os.path.join(config.train.output_dir, config.train.exp_name)
        self.checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        self.tensorboard_dir = os.path.join(self.exp_dir, 'tensorboard')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)

        # TensorBoard writer
        self.writer = SummaryWriter(self.tensorboard_dir)

        # Initialize components
        self._setup_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_loss()

        # Training state
        self.start_epoch = 0
        self.best_dice = 0.0
        self.global_step = 0
        self.patience_counter = 0

        # Mixed precision
        self.scaler = GradScaler() if config.train.use_amp else None

        # Resume if specified
        if config.train.resume_checkpoint:
            self._resume_training(config.train.resume_checkpoint)

    def _setup_data(self):
        """Setup data loaders"""
        self.logger.info("Setting up data loaders...")

        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            self.config,
            num_workers=self.config.train.num_workers
        )

        self.logger.info(f"Train batches: {len(self.train_loader)}")
        self.logger.info(f"Val batches: {len(self.val_loader)}")
        self.logger.info(f"Test batches: {len(self.test_loader)}")

    def _setup_model(self):
        """Setup model"""
        self.logger.info(f"Setting up model: {self.config.model.model_type}")

        self.model = create_model(
            model_type=self.config.model.model_type,
            in_channels=self.config.model.in_channels,
            out_channels=self.config.model.out_channels,
            img_size=self.config.preprocess.patch_size,
            config=self.config,
            use_deep_supervision=self.config.model.use_deep_supervision,
            dropout=self.config.model.dropout_rate
        ).to(self.device)

        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")

    def _setup_optimizer(self):
        """Setup optimizer and scheduler"""
        self.logger.info("Setting up optimizer and scheduler...")

        # Optimizer
        if self.config.train.optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.train.learning_rate,
                weight_decay=self.config.train.weight_decay
            )
        elif self.config.train.optimizer.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.train.learning_rate,
                weight_decay=self.config.train.weight_decay
            )
        elif self.config.train.optimizer.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.train.learning_rate,
                momentum=0.9,
                weight_decay=self.config.train.weight_decay,
                nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.train.optimizer}")

        # Scheduler
        if self.config.train.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.train.num_epochs,
                eta_min=self.config.train.min_lr
            )
        elif self.config.train.scheduler == 'cosine_warmup':
            from torch.optim.lr_scheduler import LambdaLR

            def warmup_cosine(epoch):
                if epoch < self.config.train.warmup_epochs:
                    return epoch / self.config.train.warmup_epochs
                progress = (epoch - self.config.train.warmup_epochs) / (
                    self.config.train.num_epochs - self.config.train.warmup_epochs
                )
                return 0.5 * (1 + np.cos(np.pi * progress))

            self.scheduler = LambdaLR(self.optimizer, warmup_cosine)
        elif self.config.train.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config.train.scheduler == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                min_lr=self.config.train.min_lr
            )
        else:
            self.scheduler = None

    def _setup_loss(self):
        """Setup loss function"""
        self.logger.info("Setting up loss function...")

        # Compute class weights if needed
        if self.config.train.use_class_weights and self.config.train.class_weights is None:
            self.logger.info("Computing class weights from training data...")
            self.config.train.class_weights = compute_class_weights(
                self.train_loader,
                self.config.data.num_classes,
                self.device
            ).tolist()
            self.logger.info(f"Class weights: {self.config.train.class_weights}")

        self.loss_fn = get_loss_function(self.config)

        # Metrics
        self.dice_metric = DiceMetric(
            num_classes=self.config.data.num_classes,
            include_background=False
        )

    def _resume_training(self, checkpoint_path: str):
        """Resume training from checkpoint"""
        self.logger.info(f"Resuming from checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load training state
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_dice = checkpoint.get('best_dice', 0.0)
        self.global_step = checkpoint.get('global_step', 0)

        self.logger.info(f"Resumed from epoch {self.start_epoch}, best dice: {self.best_dice:.4f}")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint"""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict() if isinstance(
                self.model, nn.DataParallel
            ) else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_dice': self.best_dice,
            'global_step': self.global_step,
            'config': self.config
        }

        if self.scheduler:
            state['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_epoch_{epoch:04d}.pth'
        )
        torch.save(state, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(state, best_path)
            self.logger.info(f"Saved best model with dice: {self.best_dice:.4f}")

        # Clean up old checkpoints
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Keep only the latest N checkpoints"""
        checkpoints = sorted(Path(self.checkpoint_dir).glob('checkpoint_epoch_*.pth'))
        if len(checkpoints) > self.config.train.keep_n_checkpoints:
            for ckpt in checkpoints[:-self.config.train.keep_n_checkpoints]:
                ckpt.unlink()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_losses = {}
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}', leave=False)
        accumulation_counter = 0

        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue

            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward pass with mixed precision
            if self.config.train.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss, loss_dict = self.loss_fn(outputs, labels)
                    loss = loss / self.config.train.accumulation_steps

                # Backward pass
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(images)
                loss, loss_dict = self.loss_fn(outputs, labels)
                loss = loss / self.config.train.accumulation_steps
                loss.backward()

            accumulation_counter += 1

            # Gradient accumulation
            if accumulation_counter == self.config.train.accumulation_steps:
                if self.config.train.use_amp:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.train.grad_clip_max_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.train.grad_clip_max_norm
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()
                accumulation_counter = 0
                self.global_step += 1

            epoch_loss += loss.item() * self.config.train.accumulation_steps

            # Update epoch losses
            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item() * self.config.train.accumulation_steps:.4f}'})

            # Log to tensorboard
            if self.global_step % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item() * self.config.train.accumulation_steps, self.global_step)
                for k, v in loss_dict.items():
                    self.writer.add_scalar(f'train/{k}', v, self.global_step)

        # Calculate averages
        epoch_loss /= num_batches
        for k in epoch_losses:
            epoch_losses[k] /= num_batches

        return {'loss': epoch_loss, **epoch_losses}

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        self.dice_metric.reset()

        val_loss = 0.0
        num_batches = len(self.val_loader)

        pbar = tqdm(self.val_loader, desc='Validation', leave=False)

        for batch in pbar:
            if batch is None:
                continue

            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # Sliding window inference for full volumes
            if MONAI_AVAILABLE and images.shape[2:] != self.config.preprocess.patch_size:
                outputs = sliding_window_inference(
                    images,
                    roi_size=self.config.preprocess.patch_size,
                    sw_batch_size=self.config.train.sw_batch_size,
                    predictor=self.model,
                    overlap=self.config.train.sw_overlap
                )
            else:
                outputs = self.model(images)

            # Handle deep supervision output
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # Compute loss
            loss, _ = self.loss_fn(outputs, labels) if hasattr(self.loss_fn, 'losses') else (
                nn.functional.cross_entropy(outputs, labels.squeeze(1).long()), {}
            )
            val_loss += loss.item()

            # Update metrics
            pred = outputs.argmax(dim=1)
            self.dice_metric.update(pred, labels.squeeze(1))

        # Compute metrics
        val_loss /= num_batches
        dice_results = self.dice_metric.compute()

        # Log to tensorboard
        self.writer.add_scalar('val/loss', val_loss, epoch)
        for k, v in dice_results.items():
            self.writer.add_scalar(f'val/{k}', v, epoch)

        return {'loss': val_loss, **dice_results}

    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        self.logger.info(f"Model: {self.config.model.model_type}")
        self.logger.info(f"Epochs: {self.config.train.num_epochs}")
        self.logger.info(f"Batch size: {self.config.train.batch_size}")
        self.logger.info(f"Learning rate: {self.config.train.learning_rate}")

        for epoch in range(self.start_epoch, self.config.train.num_epochs):
            epoch_start_time = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            if (epoch + 1) % self.config.train.val_every_n_epochs == 0:
                val_metrics = self.validate(epoch)

                # Log metrics
                self.logger.info(
                    f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Mean Dice: {val_metrics.get('mean_dice', 0):.4f}"
                )

                # Check for best model
                current_dice = val_metrics.get('mean_dice', 0)
                is_best = current_dice > self.best_dice

                if is_best:
                    self.best_dice = current_dice
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                # Save checkpoint
                if (epoch + 1) % self.config.train.save_every_n_epochs == 0 or is_best:
                    self.save_checkpoint(epoch, is_best)

                # Early stopping
                if self.patience_counter >= self.config.train.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break

                # Update scheduler
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(current_dice)
                    else:
                        self.scheduler.step()

            else:
                # Just update scheduler without validation
                if self.scheduler and not isinstance(
                    self.scheduler, optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step()

            epoch_time = time.time() - epoch_start_time
            self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")

            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('train/learning_rate', current_lr, epoch)

        # Final test evaluation
        self.logger.info("Training completed. Running final test evaluation...")
        self._load_best_model()
        test_metrics = self._test()
        self.logger.info(f"Final Test Metrics: {test_metrics}")

        self.writer.close()

        return self.best_dice

    def _load_best_model(self):
        """Load best model checkpoint"""
        best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=self.device)
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("Loaded best model")

    @torch.no_grad()
    def _test(self) -> Dict[str, float]:
        """Run test evaluation"""
        self.model.eval()
        self.dice_metric.reset()

        for batch in tqdm(self.test_loader, desc='Testing'):
            if batch is None:
                continue

            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # Sliding window inference
            if MONAI_AVAILABLE:
                outputs = sliding_window_inference(
                    images,
                    roi_size=self.config.preprocess.patch_size,
                    sw_batch_size=self.config.train.sw_batch_size,
                    predictor=self.model,
                    overlap=self.config.train.sw_overlap
                )
            else:
                outputs = self.model(images)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            pred = outputs.argmax(dim=1)
            self.dice_metric.update(pred, labels.squeeze(1))

        return self.dice_metric.compute()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train segmentation model')

    parser.add_argument('--model', type=str, default='swin_unetr',
                        choices=get_model_names(),
                        help='Model architecture')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=str, default='2,3,4,5',
                        help='GPU device IDs (default: 2,3,4,5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Note: CUDA_VISIBLE_DEVICES is set at the top of this file before torch import
    # If --gpu is specified differently, user should set CUDA_VISIBLE_DEVICES env var before running
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Load config
    config = get_config(args.model)

    # Override config with command line arguments
    config.model.model_type = args.model
    if args.exp_name:
        config.train.exp_name = args.exp_name
    else:
        config.train.exp_name = f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.output_dir:
        config.train.output_dir = args.output_dir
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.epochs:
        config.train.num_epochs = args.epochs
    if args.lr:
        config.train.learning_rate = args.lr
    if args.resume:
        config.train.resume_checkpoint = args.resume

    # Setup logging
    logger = setup_logging(config.train.output_dir, config.train.exp_name)
    logger.info(f"Configuration: {config}")

    # Create trainer and train
    trainer = Trainer(config, device)
    best_dice = trainer.train()

    logger.info(f"Training completed. Best Dice: {best_dice:.4f}")


if __name__ == '__main__':
    main()
