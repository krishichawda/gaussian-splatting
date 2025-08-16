#!/usr/bin/env python3
"""
Inference script for Generalizable Sparse-View 3D Object Reconstruction
using 3D Gaussian Splatting with Transformer architecture.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.transformer_encoder import SparseViewReconstructionModel
from utils.visualization import (
    save_images, save_gaussian_visualization, save_gaussian_ply,
    create_3d_scatter_plot, create_gaussian_heatmap
)


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


def load_and_preprocess_images(image_paths: List[str], image_size: Tuple[int, int]) -> torch.Tensor:
    """Load and preprocess images for inference."""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        images.append(image)
    
    return torch.stack(images)  # (num_views, C, H, W)


def create_camera_parameters(num_views: int, image_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create default camera parameters for unposed images."""
    # Create camera matrix
    fov = 60.0
    focal_length = image_size[0] / (2 * np.tan(np.radians(fov) / 2))
    camera_matrix = torch.tensor([
        [focal_length, 0, image_size[0] / 2],
        [0, focal_length, image_size[1] / 2],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    # Create camera poses in a circle around the object
    camera_matrices = []
    camera_poses = []
    
    for i in range(num_views):
        angle = 2 * np.pi * i / num_views
        distance = 2.0
        
        # Camera position
        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        z = 1.0
        
        # Look at origin
        look_at = torch.zeros(3, dtype=torch.float32)
        up = torch.tensor([0, 0, 1], dtype=torch.float32)
        position = torch.tensor([x, y, z], dtype=torch.float32)
        
        # Create camera coordinate system
        z_axis = F.normalize(look_at - position, dim=0)
        x_axis = F.normalize(torch.cross(up, z_axis), dim=0)
        y_axis = torch.cross(z_axis, x_axis)
        
        # Create rotation matrix
        rotation = torch.stack([x_axis, y_axis, z_axis], dim=1)
        
        # Create camera pose matrix
        camera_pose = torch.eye(4, dtype=torch.float32)
        camera_pose[:3, :3] = rotation
        camera_pose[:3, 3] = position
        
        camera_matrices.append(camera_matrix)
        camera_poses.append(camera_pose)
    
    return torch.stack(camera_matrices), torch.stack(camera_poses)


def reconstruct_3d_object(model: SparseViewReconstructionModel, images: torch.Tensor,
                         camera_matrices: torch.Tensor, camera_poses: torch.Tensor,
                         device: torch.device) -> Dict[str, torch.Tensor]:
    """Reconstruct 3D object from input images."""
    with torch.no_grad():
        # Move data to device
        images = images.to(device)
        camera_matrices = camera_matrices.to(device)
        camera_poses = camera_poses.to(device)
        
        # Add batch dimension
        images = images.unsqueeze(0)  # (1, num_views, C, H, W)
        camera_matrices = camera_matrices.unsqueeze(0)  # (1, num_views, 3, 3)
        camera_poses = camera_poses.unsqueeze(0)  # (1, num_views, 4, 4)
        
        image_size = (images.shape[3], images.shape[4])
        
        # Forward pass
        rendered_image = model(images, camera_matrices[:, 0], camera_poses[:, 0], image_size)
        
        # Get Gaussian parameters
        features = model.encode_features(images)
        gaussian_params = model.predict_gaussians(features)
        
        return {
            'rendered_image': rendered_image[0],  # Remove batch dimension
            'gaussian_params': gaussian_params,
            'features': features[0]  # Remove batch dimension
        }


def save_results(results: Dict[str, torch.Tensor], output_dir: str, scene_name: str):
    """Save reconstruction results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save rendered image
    rendered_image = results['rendered_image']
    rendered_path = os.path.join(output_dir, f'{scene_name}_rendered.png')
    
    # Denormalize and save
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rendered_denorm = torch.clamp(rendered_image * std + mean, 0, 1)
    
    rendered_pil = transforms.ToPILImage()(rendered_denorm)
    rendered_pil.save(rendered_path)
    logging.info(f"Saved rendered image to {rendered_path}")
    
    # Save Gaussian visualization
    gaussian_params = results['gaussian_params']
    
    # PLY file
    ply_path = os.path.join(output_dir, f'{scene_name}_gaussians.ply')
    save_gaussian_ply(gaussian_params, ply_path)
    
    # Point cloud visualization
    pcd_path = os.path.join(output_dir, f'{scene_name}_pointcloud.ply')
    save_gaussian_visualization(gaussian_params, pcd_path)
    
    # 3D scatter plot
    scatter_path = os.path.join(output_dir, f'{scene_name}_scatter.png')
    create_3d_scatter_plot(gaussian_params, scatter_path)
    
    # 2D heatmap
    heatmap_path = os.path.join(output_dir, f'{scene_name}_heatmap.png')
    create_gaussian_heatmap(gaussian_params, heatmap_path)
    
    # Save Gaussian parameters as JSON
    params_path = os.path.join(output_dir, f'{scene_name}_params.json')
    save_gaussian_params_json(gaussian_params, params_path)
    
    logging.info(f"Saved all results to {output_dir}")


def save_gaussian_params_json(gaussian_params: Dict[str, torch.Tensor], save_path: str):
    """Save Gaussian parameters as JSON file."""
    params_dict = {}
    
    for key, tensor in gaussian_params.items():
        if key == 'features':
            # Skip features as they're too large
            continue
        
        # Convert to numpy and then to list
        params_dict[key] = tensor.detach().cpu().numpy().tolist()
    
    with open(save_path, 'w') as f:
        json.dump(params_dict, f, indent=2)
    
    logging.info(f"Saved Gaussian parameters to {save_path}")


def render_novel_views(model: SparseViewReconstructionModel, gaussian_params: Dict[str, torch.Tensor],
                      num_views: int, image_size: Tuple[int, int], device: torch.device,
                      output_dir: str, scene_name: str):
    """Render novel views from the reconstructed 3D Gaussians."""
    # Create novel camera poses
    camera_matrices, camera_poses = create_camera_parameters(num_views, image_size)
    
    # Move to device
    camera_matrices = camera_matrices.to(device)
    camera_poses = camera_poses.to(device)
    
    # Render novel views
    novel_views = []
    
    for i in range(num_views):
        with torch.no_grad():
            rendered = model.renderer(gaussian_params, camera_matrices[i], camera_poses[i], image_size)
            novel_views.append(rendered)
    
    # Save novel views
    novel_views = torch.stack(novel_views)
    
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    novel_views_denorm = torch.clamp(novel_views * std + mean, 0, 1)
    
    # Save as grid
    grid_path = os.path.join(output_dir, f'{scene_name}_novel_views.png')
    save_novel_views_grid(novel_views_denorm, grid_path)
    
    # Save individual views
    for i, view in enumerate(novel_views_denorm):
        view_path = os.path.join(output_dir, f'{scene_name}_novel_view_{i:02d}.png')
        view_pil = transforms.ToPILImage()(view)
        view_pil.save(view_path)
    
    logging.info(f"Saved {num_views} novel views to {output_dir}")


def save_novel_views_grid(novel_views: torch.Tensor, save_path: str):
    """Save novel views as a grid image."""
    import torchvision.utils as vutils
    
    # Create grid
    grid = vutils.make_grid(novel_views, nrow=4, padding=2, normalize=False)
    
    # Save
    vutils.save_image(grid, save_path)
    logging.info(f"Saved novel views grid to {save_path}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Inference for 3D Gaussian Splatting Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Directory to save results')
    parser.add_argument('--num_views', type=int, default=4,
                       help='Number of input views to use')
    parser.add_argument('--novel_views', type=int, default=8,
                       help='Number of novel views to render')
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256],
                       help='Image size (width, height)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting inference...")
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, config, device)
    
    # Find input images
    input_dir = Path(args.input_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f'*{ext}'))
        image_files.extend(input_dir.glob(f'*{ext.upper()}'))
    
    if len(image_files) < args.num_views:
        logger.error(f"Not enough images found. Need at least {args.num_views}, found {len(image_files)}")
        return
    
    # Sort and take first num_views
    image_files = sorted(image_files)[:args.num_views]
    logger.info(f"Using {len(image_files)} images: {[f.name for f in image_files]}")
    
    # Load and preprocess images
    image_paths = [str(f) for f in image_files]
    images = load_and_preprocess_images(image_paths, tuple(args.image_size))
    
    # Create camera parameters
    camera_matrices, camera_poses = create_camera_parameters(args.num_views, tuple(args.image_size))
    
    # Reconstruct 3D object
    logger.info("Reconstructing 3D object...")
    results = reconstruct_3d_object(model, images, camera_matrices, camera_poses, device)
    
    # Save results
    scene_name = input_dir.name
    save_results(results, args.output_dir, scene_name)
    
    # Render novel views
    logger.info("Rendering novel views...")
    render_novel_views(
        model, results['gaussian_params'], args.novel_views,
        tuple(args.image_size), device, args.output_dir, scene_name
    )
    
    logger.info("Inference completed!")


if __name__ == '__main__':
    main()
