#!/usr/bin/env python3
"""
Training script for Generalizable Sparse-View 3D Object Reconstruction
using 3D Gaussian Splatting with Transformer architecture.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import wandb
from pathlib import Path
import logging
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.transformer_encoder import SparseViewReconstructionModel
from data.dataset import create_dataloader
from utils.losses import TotalLoss, compute_metrics
from utils.visualization import save_images, save_gaussian_visualization


def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(config: Dict[str, Any]) -> torch.device:
    """Setup device (CPU/GPU) based on configuration."""
    if config['hardware']['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.set_device(0)  # Use first GPU
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU")
    
    return device


def create_model(config: Dict[str, Any], device: torch.device) -> SparseViewReconstructionModel:
    """Create and initialize the model."""
    model_config = config['model']
    
    model = SparseViewReconstructionModel(
        num_gaussians=model_config['num_gaussians'],
        feature_dim=model_config['feature_dim'],
        transformer_layers=model_config['transformer_layers'],
        num_heads=model_config['num_heads'],
        dropout=model_config['dropout'],
        near_plane=config['rendering']['near_plane'],
        far_plane=config['rendering']['far_plane']
    )
    
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"Model created with {total_params:,} total parameters")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    
    return model


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """Create optimizer for training."""
    training_config = config['training']
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any]) -> optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler."""
    training_config = config['training']
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=training_config['scheduler_step_size'],
        gamma=training_config['scheduler_gamma']
    )
    
    return scheduler


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer,
                criterion: TotalLoss, device: torch.device, epoch: int,
                writer: SummaryWriter, config: Dict[str, Any]) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_metrics = {}
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        images = batch['images'].to(device)  # (B, num_views, C, H, W)
        camera_matrices = batch['camera_matrices'].to(device)  # (B, num_views, 3, 3)
        camera_poses = batch['camera_poses'].to(device)  # (B, num_views, 4, 4)
        
        batch_size = images.shape[0]
        num_views = images.shape[1]
        image_size = (images.shape[3], images.shape[4])
        
        # Forward pass
        optimizer.zero_grad()
        
        # Use first view as target for now (in practice, you might want to use a novel view)
        target_images = images[:, 0]  # (B, C, H, W)
        input_images = images[:, 1:]  # (B, num_views-1, C, H, W)
        target_camera_matrix = camera_matrices[:, 0]  # (B, 3, 3)
        target_camera_pose = camera_poses[:, 0]  # (B, 4, 4)
        
        # Forward pass through model
        rendered_images = model(input_images, target_camera_matrix, target_camera_pose, image_size)
        
        # Get Gaussian parameters for loss computation
        features = model.encode_features(input_images)
        gaussian_params = model.predict_gaussians(features)
        
        # Compute loss
        losses = criterion(rendered_images, target_images, gaussian_params)
        loss = losses['total']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if config['training']['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
        
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            metrics = compute_metrics(rendered_images, target_images)
        
        # Update running totals
        total_loss += loss.item()
        for key, value in metrics.items():
            if key not in total_metrics:
                total_metrics[key] = 0.0
            total_metrics[key] += value.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'PSNR': f"{metrics['psnr']:.2f}",
            'SSIM': f"{metrics['ssim']:.3f}"
        })
        
        # Log to tensorboard
        if batch_idx % 10 == 0:
            step = epoch * num_batches + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), step)
            writer.add_scalar('Train/PSNR', metrics['psnr'], step)
            writer.add_scalar('Train/SSIM', metrics['ssim'], step)
            
            # Log individual loss components
            for loss_name, loss_value in losses.items():
                writer.add_scalar(f'Train/{loss_name}_loss', loss_value.item(), step)
        
        # Save sample images periodically
        if batch_idx % 100 == 0:
            save_images(
                rendered_images[:4], target_images[:4],
                os.path.join(config['logging']['log_dir'], f'epoch_{epoch}_batch_{batch_idx}.png')
            )
    
    # Compute averages
    avg_loss = total_loss / num_batches
    avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
    
    return {'loss': avg_loss, **avg_metrics}


def validate_epoch(model: nn.Module, dataloader: DataLoader, criterion: TotalLoss,
                  device: torch.device, epoch: int, writer: SummaryWriter,
                  config: Dict[str, Any]) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    
    total_loss = 0.0
    total_metrics = {}
    num_batches = len(dataloader)
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Validation Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            images = batch['images'].to(device)
            camera_matrices = batch['camera_matrices'].to(device)
            camera_poses = batch['camera_poses'].to(device)
            
            batch_size = images.shape[0]
            num_views = images.shape[1]
            image_size = (images.shape[3], images.shape[4])
            
            # Use first view as target
            target_images = images[:, 0]
            input_images = images[:, 1:]
            target_camera_matrix = camera_matrices[:, 0]
            target_camera_pose = camera_poses[:, 0]
            
            # Forward pass
            rendered_images = model(input_images, target_camera_matrix, target_camera_pose, image_size)
            
            # Get Gaussian parameters
            features = model.encode_features(input_images)
            gaussian_params = model.predict_gaussians(features)
            
            # Compute loss
            losses = criterion(rendered_images, target_images, gaussian_params)
            loss = losses['total']
            
            # Compute metrics
            metrics = compute_metrics(rendered_images, target_images)
            
            # Update running totals
            total_loss += loss.item()
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                total_metrics[key] += value.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'PSNR': f"{metrics['psnr']:.2f}",
                'SSIM': f"{metrics['ssim']:.3f}"
            })
            
            # Save sample images
            if batch_idx == 0:
                save_images(
                    rendered_images[:4], target_images[:4],
                    os.path.join(config['logging']['log_dir'], f'val_epoch_{epoch}.png')
                )
                
                # Save Gaussian visualization
                save_gaussian_visualization(
                    gaussian_params,
                    os.path.join(config['logging']['log_dir'], f'gaussians_epoch_{epoch}.ply')
                )
    
    # Compute averages
    avg_loss = total_loss / num_batches
    avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
    
    # Log to tensorboard
    step = epoch * num_batches
    writer.add_scalar('Val/Loss', avg_loss, step)
    writer.add_scalar('Val/PSNR', avg_metrics['psnr'], step)
    writer.add_scalar('Val/SSIM', avg_metrics['ssim'], step)
    
    return {'loss': avg_loss, **avg_metrics}


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler,
                   epoch: int, metrics: Dict[str, float], config: Dict[str, Any], is_best: bool = False):
    """Save model checkpoint."""
    checkpoint_dir = os.path.join(config['logging']['log_dir'], 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'config': config
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_checkpoint.pth')
        torch.save(checkpoint, best_path)
        logging.info(f"Saved best checkpoint with PSNR: {metrics['psnr']:.2f}")
    
    # Keep only recent checkpoints
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')])
    if len(checkpoints) > 5:  # Keep only 5 most recent
        for checkpoint_file in checkpoints[:-5]:
            os.remove(os.path.join(checkpoint_dir, checkpoint_file))


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train 3D Gaussian Splatting Model')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--use_synthetic', action='store_true',
                       help='Use synthetic dataset for testing')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config['logging']['log_dir'])
    logger.info("Starting training...")
    
    # Setup device
    device = setup_device(config)
    
    # Initialize wandb if enabled
    if config['logging']['wandb']:
        wandb.init(
            project="gaussian-splatting-3d",
            config=config,
            name=f"sparse_view_reconstruction_{config['model']['num_gaussians']}gaussians"
        )
    
    # Create model
    model = create_model(config, device)
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        data_dir=config['data']['train_dir'],
        batch_size=config['data']['batch_size'],
        num_views=config['data']['num_views'],
        image_size=tuple(config['data']['image_size']),
        num_workers=config['data']['num_workers'],
        split='train',
        use_synthetic=args.use_synthetic
    )
    
    val_dataloader = create_dataloader(
        data_dir=config['data']['val_dir'],
        batch_size=config['data']['batch_size'],
        num_views=config['data']['num_views'],
        image_size=tuple(config['data']['image_size']),
        num_workers=config['data']['num_workers'],
        split='val',
        use_synthetic=args.use_synthetic
    )
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Create loss function
    criterion = TotalLoss(config['loss'])
    criterion = criterion.to(device)
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=config['logging']['log_dir'])
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_psnr = 0.0
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint['metrics']['psnr']
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    num_epochs = config['training']['epochs']
    
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Starting epoch {epoch}/{num_epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, train_dataloader, optimizer, criterion, device, epoch, writer, config
        )
        
        # Validate
        if epoch % config['logging']['val_freq'] == 0:
            val_metrics = validate_epoch(
                model, val_dataloader, criterion, device, epoch, writer, config
            )
            
            # Log metrics
            logger.info(f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}, "
                       f"Val Loss: {val_metrics['loss']:.4f}, "
                       f"Val PSNR: {val_metrics['psnr']:.2f}")
            
            # Save checkpoint
            is_best = val_metrics['psnr'] > best_psnr
            if is_best:
                best_psnr = val_metrics['psnr']
            
            if epoch % config['logging']['save_freq'] == 0 or is_best:
                save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, config, is_best)
        
        # Update learning rate
        scheduler.step()
        
        # Log learning rate
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('Train/Learning_Rate', current_lr, epoch)
        
        # Log to wandb
        if config['logging']['wandb']:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'val_psnr': val_metrics['psnr'],
                'val_ssim': val_metrics['ssim'],
                'learning_rate': current_lr
            })
    
    # Save final checkpoint
    save_checkpoint(model, optimizer, scheduler, num_epochs - 1, val_metrics, config)
    
    logger.info("Training completed!")
    writer.close()
    
    if config['logging']['wandb']:
        wandb.finish()


if __name__ == '__main__':
    main()
