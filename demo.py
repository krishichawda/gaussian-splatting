#!/usr/bin/env python3
"""
Demo script for Generalizable Sparse-View 3D Object Reconstruction
using 3D Gaussian Splatting with Transformer architecture.

This script demonstrates the complete pipeline with synthetic data.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.transformer_encoder import SparseViewReconstructionModel
from data.dataset import SyntheticDataset
from utils.visualization import save_images, create_3d_scatter_plot, create_gaussian_heatmap
from utils.losses import TotalLoss, compute_metrics


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def create_demo_config():
    """Create a demo configuration for quick testing."""
    config = {
        'model': {
            'num_gaussians': 1000,  # Smaller for demo
            'feature_dim': 128,
            'transformer_layers': 3,
            'num_heads': 4,
            'dropout': 0.1,
            'use_positional_encoding': True
        },
        'rendering': {
            'near_plane': 0.1,
            'far_plane': 10.0
        },
        'loss': {
            'reconstruction_weight': 1.0,
            'ssim_weight': 0.1,
            'depth_weight': 0.1,
            'smoothness_weight': 0.01,
            'sparsity_weight': 0.001,
            'regularization_weight': 0.01
        }
    }
    return config


def run_demo():
    """Run the complete demo pipeline."""
    logger = setup_logging()
    logger.info("Starting 3D Gaussian Splatting Demo...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create demo configuration
    config = create_demo_config()
    
    # Create synthetic dataset
    logger.info("Creating synthetic dataset...")
    dataset = SyntheticDataset(
        num_samples=10,
        num_views=4,
        image_size=(128, 128)  # Smaller for demo
    )
    
    # Create model
    logger.info("Creating model...")
    model = SparseViewReconstructionModel(
        num_gaussians=config['model']['num_gaussians'],
        feature_dim=config['model']['feature_dim'],
        transformer_layers=config['model']['transformer_layers'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        near_plane=config['rendering']['near_plane'],
        far_plane=config['rendering']['far_plane']
    )
    model = model.to(device)
    
    # Create loss function
    criterion = TotalLoss(config['loss'])
    criterion = criterion.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Training loop (simplified for demo)
    logger.info("Starting training...")
    num_epochs = 5
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx in range(5):  # Just a few batches for demo
            # Get sample from dataset
            sample = dataset[batch_idx]
            images = sample['images'].unsqueeze(0).to(device)  # Add batch dimension
            camera_matrices = sample['camera_matrices'].unsqueeze(0).to(device)
            camera_poses = sample['camera_poses'].unsqueeze(0).to(device)
            
            # Use first view as target
            target_images = images[:, 0]
            input_images = images[:, 1:]
            target_camera_matrix = camera_matrices[:, 0]
            target_camera_pose = camera_poses[:, 0]
            
            # Forward pass
            optimizer.zero_grad()
            rendered_images = model(input_images, target_camera_matrix, target_camera_pose, (128, 128))
            
            # Get Gaussian parameters for loss
            features = model.encode_features(input_images)
            gaussian_params = model.predict_gaussians(features)
            
            # Compute loss
            losses = criterion(rendered_images, target_images, gaussian_params)
            loss = losses['total']
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Compute metrics
            with torch.no_grad():
                metrics = compute_metrics(rendered_images, target_images)
            
            if batch_idx % 2 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.4f}, "
                           f"PSNR = {metrics['psnr']:.2f}, SSIM = {metrics['ssim']:.3f}")
        
        avg_loss = total_loss / 5
        logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
    
    # Evaluation
    logger.info("Running evaluation...")
    model.eval()
    
    with torch.no_grad():
        # Get a test sample
        test_sample = dataset[0]
        images = test_sample['images'].unsqueeze(0).to(device)
        camera_matrices = test_sample['camera_matrices'].unsqueeze(0).to(device)
        camera_poses = test_sample['camera_poses'].unsqueeze(0).to(device)
        
        target_images = images[:, 0]
        input_images = images[:, 1:]
        target_camera_matrix = camera_matrices[:, 0]
        target_camera_pose = camera_poses[:, 0]
        
        # Forward pass
        rendered_images = model(input_images, target_camera_matrix, target_camera_pose, (128, 128))
        
        # Get Gaussian parameters
        features = model.encode_features(input_images)
        gaussian_params = model.predict_gaussians(features)
        
        # Compute final metrics
        metrics = compute_metrics(rendered_images, target_images)
        logger.info(f"Final metrics - PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.3f}")
    
    # Save results
    logger.info("Saving demo results...")
    os.makedirs('demo_results', exist_ok=True)
    
    # Save rendered vs target comparison
    save_images(
        rendered_images[:4], target_images[:4],
        'demo_results/rendered_vs_target.png',
        num_images=4
    )
    
    # Save 3D scatter plot
    create_3d_scatter_plot(
        gaussian_params,
        'demo_results/gaussian_positions.png',
        max_points=500
    )
    
    # Save 2D heatmap
    create_gaussian_heatmap(
        gaussian_params,
        'demo_results/gaussian_density.png',
        resolution=50
    )
    
    # Create a simple visualization of the training progress
    logger.info("Creating training visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original input images
    input_imgs = input_images[0].cpu()  # Remove batch dimension
    for i in range(min(3, input_imgs.shape[0])):
        img = input_imgs[i]
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = torch.clamp(img * std + mean, 0, 1)
        axes[i].imshow(img.permute(1, 2, 0))
        axes[i].set_title(f'Input View {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_results/input_views.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    with open('demo_results/model_summary.txt', 'w') as f:
        f.write("3D Gaussian Splatting Model Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n")
        f.write(f"Number of Gaussians: {config['model']['num_gaussians']}\n")
        f.write(f"Feature dimension: {config['model']['feature_dim']}\n")
        f.write(f"Transformer layers: {config['model']['transformer_layers']}\n")
        f.write(f"Number of heads: {config['model']['num_heads']}\n")
        f.write(f"Final PSNR: {metrics['psnr']:.2f}\n")
        f.write(f"Final SSIM: {metrics['ssim']:.3f}\n")
    
    logger.info("Demo completed! Check 'demo_results' directory for outputs.")
    logger.info(f"Final PSNR: {metrics['psnr']:.2f}")
    logger.info(f"Final SSIM: {metrics['ssim']:.3f}")


if __name__ == '__main__':
    run_demo()
