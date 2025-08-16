#!/usr/bin/env python3
"""
Evaluation script for Generalizable Sparse-View 3D Object Reconstruction
using 3D Gaussian Splatting with Transformer architecture.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import json
from pathlib import Path
import logging
from typing import Dict, Any, List
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.transformer_encoder import SparseViewReconstructionModel
from data.dataset import create_dataloader
from utils.losses import TotalLoss, compute_metrics
from utils.visualization import save_images, create_training_plots


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path: str, config: Dict[str, Any], device: torch.device) -> SparseViewReconstructionModel:
    """Load trained model from checkpoint."""
    # Create model
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
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logging.info(f"Loaded model from {checkpoint_path}")
    logging.info(f"Checkpoint epoch: {checkpoint['epoch']}")
    logging.info(f"Checkpoint metrics: {checkpoint['metrics']}")
    
    return model


def evaluate_model(model: SparseViewReconstructionModel, dataloader: DataLoader,
                  criterion: TotalLoss, device: torch.device) -> Dict[str, List[float]]:
    """Evaluate model on test dataset."""
    model.eval()
    
    all_metrics = {
        'psnr': [],
        'ssim': [],
        'l1_error': [],
        'l2_error': [],
        'reconstruction_loss': [],
        'total_loss': []
    }
    
    all_losses = {
        'reconstruction': [],
        'ssim': [],
        'depth': [],
        'smoothness': [],
        'sparsity': [],
        'regularization': []
    }
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        
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
            
            # Compute metrics
            metrics = compute_metrics(rendered_images, target_images)
            
            # Store results
            for key, value in metrics.items():
                all_metrics[key].append(value.item())
            
            for key, value in losses.items():
                all_losses[key].append(value.item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'PSNR': f"{metrics['psnr']:.2f}",
                'SSIM': f"{metrics['ssim']:.3f}",
                'Loss': f"{losses['total']:.4f}"
            })
    
    return all_metrics, all_losses


def compute_statistics(metrics: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """Compute statistics for evaluation metrics."""
    statistics = {}
    
    for metric_name, values in metrics.items():
        values = np.array(values)
        statistics[metric_name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values))
        }
    
    return statistics


def save_evaluation_results(metrics: Dict[str, List[float]], losses: Dict[str, List[float]],
                           statistics: Dict[str, Dict[str, float]], output_dir: str):
    """Save evaluation results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw metrics
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save raw losses
    losses_path = os.path.join(output_dir, 'evaluation_losses.json')
    with open(losses_path, 'w') as f:
        json.dump(losses, f, indent=2)
    
    # Save statistics
    stats_path = os.path.join(output_dir, 'evaluation_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(statistics, f, indent=2)
    
    logging.info(f"Saved evaluation results to {output_dir}")


def create_evaluation_plots(metrics: Dict[str, List[float]], losses: Dict[str, List[float]],
                           output_dir: str):
    """Create evaluation plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Metrics distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    metric_names = ['psnr', 'ssim', 'l1_error', 'l2_error']
    titles = ['PSNR Distribution', 'SSIM Distribution', 'L1 Error Distribution', 'L2 Error Distribution']
    
    for i, (metric_name, title) in enumerate(zip(metric_names, titles)):
        values = metrics[metric_name]
        axes[i].hist(values, bins=30, alpha=0.7, edgecolor='black')
        axes[i].set_title(title)
        axes[i].set_xlabel(metric_name.upper())
        axes[i].set_ylabel('Frequency')
        axes[i].axvline(np.mean(values), color='red', linestyle='--', label=f'Mean: {np.mean(values):.3f}')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Loss distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    loss_names = list(losses.keys())
    for i, loss_name in enumerate(loss_names):
        values = losses[loss_name]
        axes[i].hist(values, bins=30, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'{loss_name.replace("_", " ").title()} Distribution')
        axes[i].set_xlabel('Loss Value')
        axes[i].set_ylabel('Frequency')
        axes[i].axvline(np.mean(values), color='red', linestyle='--', label=f'Mean: {np.mean(values):.4f}')
        axes[i].legend()
    
    # Remove extra subplot
    if len(loss_names) < 6:
        axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'losses_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Correlation matrix
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Metrics correlation
    metrics_df = {k: v for k, v in metrics.items() if k in ['psnr', 'ssim', 'l1_error', 'l2_error']}
    metrics_corr = np.corrcoef([metrics_df[k] for k in metrics_df.keys()])
    
    im1 = axes[0].imshow(metrics_corr, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0].set_title('Metrics Correlation Matrix')
    axes[0].set_xticks(range(len(metrics_df)))
    axes[0].set_yticks(range(len(metrics_df)))
    axes[0].set_xticklabels(list(metrics_df.keys()), rotation=45)
    axes[0].set_yticklabels(list(metrics_df.keys()))
    
    # Add correlation values
    for i in range(len(metrics_df)):
        for j in range(len(metrics_df)):
            axes[0].text(j, i, f'{metrics_corr[i, j]:.2f}', 
                        ha='center', va='center', color='black', fontweight='bold')
    
    plt.colorbar(im1, ax=axes[0])
    
    # Losses correlation
    losses_corr = np.corrcoef([losses[k] for k in losses.keys()])
    
    im2 = axes[1].imshow(losses_corr, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1].set_title('Losses Correlation Matrix')
    axes[1].set_xticks(range(len(losses)))
    axes[1].set_yticks(range(len(losses)))
    axes[1].set_xticklabels(list(losses.keys()), rotation=45)
    axes[1].set_yticklabels(list(losses.keys()))
    
    # Add correlation values
    for i in range(len(losses)):
        for j in range(len(losses)):
            axes[1].text(j, i, f'{losses_corr[i, j]:.2f}', 
                        ha='center', va='center', color='black', fontweight='bold')
    
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved evaluation plots to {output_dir}")


def print_evaluation_summary(statistics: Dict[str, Dict[str, float]]):
    """Print evaluation summary."""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    # Print metrics
    print("\nMETRICS:")
    print("-" * 40)
    for metric_name, stats in statistics.items():
        if metric_name in ['psnr', 'ssim', 'l1_error', 'l2_error']:
            print(f"{metric_name.upper():<15}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"{'':<15}  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    # Print losses
    print("\nLOSSES:")
    print("-" * 40)
    for loss_name, stats in statistics.items():
        if loss_name in ['reconstruction_loss', 'total_loss']:
            print(f"{loss_name.replace('_', ' ').title():<20}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    print("\n" + "="*60)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate 3D Gaussian Splatting Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--test_dir', type=str, required=True,
                       help='Directory containing test data')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--use_synthetic', action='store_true',
                       help='Use synthetic dataset for testing')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting evaluation...")
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, config, device)
    
    # Create test dataloader
    test_dataloader = create_dataloader(
        data_dir=args.test_dir,
        batch_size=args.batch_size,
        num_views=config['data']['num_views'],
        image_size=tuple(config['data']['image_size']),
        num_workers=config['data']['num_workers'],
        split='test',
        use_synthetic=args.use_synthetic
    )
    
    # Create loss function
    criterion = TotalLoss(config['loss'])
    criterion = criterion.to(device)
    
    # Evaluate model
    logger.info("Evaluating model...")
    metrics, losses = evaluate_model(model, test_dataloader, criterion, device)
    
    # Compute statistics
    all_data = {**metrics, **losses}
    statistics = compute_statistics(all_data)
    
    # Print summary
    print_evaluation_summary(statistics)
    
    # Save results
    logger.info("Saving evaluation results...")
    save_evaluation_results(metrics, losses, statistics, args.output_dir)
    
    # Create plots
    logger.info("Creating evaluation plots...")
    create_evaluation_plots(metrics, losses, args.output_dir)
    
    logger.info("Evaluation completed!")


if __name__ == '__main__':
    main()
