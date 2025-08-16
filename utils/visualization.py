import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from typing import Dict, Tuple, Optional
import open3d as o3d
from plyfile import PlyData, PlyElement


def save_images(rendered: torch.Tensor, target: torch.Tensor, save_path: str,
                num_images: int = 4, denormalize: bool = True):
    """Save rendered and target images side by side."""
    if denormalize:
        # Denormalize images (assuming ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        rendered = rendered * std + mean
        target = target * std + mean
    
    # Clamp values to [0, 1]
    rendered = torch.clamp(rendered, 0, 1)
    target = torch.clamp(target, 0, 1)
    
    # Take first num_images
    rendered = rendered[:num_images]
    target = target[:num_images]
    
    # Create comparison grid
    comparison = torch.cat([rendered, target], dim=0)
    grid = vutils.make_grid(comparison, nrow=num_images, padding=2, normalize=False)
    
    # Save image
    vutils.save_image(grid, save_path)
    print(f"Saved comparison image to {save_path}")


def save_gaussian_visualization(gaussian_params: Dict, save_path: str, 
                               max_points: int = 10000):
    """Save 3D Gaussian visualization as PLY file."""
    positions = gaussian_params['positions'].detach().cpu().numpy()
    scales = gaussian_params['scales'].detach().cpu().numpy()
    rotations = gaussian_params['rotations'].detach().cpu().numpy()
    opacities = gaussian_params['opacities'].detach().cpu().numpy()
    
    # Sample points if too many
    if len(positions) > max_points:
        indices = np.random.choice(len(positions), max_points, replace=False)
        positions = positions[indices]
        scales = scales[indices]
        rotations = rotations[indices]
        opacities = opacities[indices]
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    
    # Set colors based on opacity
    colors = np.zeros((len(positions), 3))
    colors[:, 0] = opacities.flatten()  # Red channel for opacity
    colors[:, 1] = 1 - opacities.flatten()  # Green channel for inverse opacity
    colors[:, 2] = 0.5  # Blue channel constant
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Save as PLY
    o3d.io.write_point_cloud(save_path, pcd)
    print(f"Saved Gaussian visualization to {save_path}")


def save_gaussian_ply(gaussian_params: Dict, save_path: str):
    """Save Gaussian parameters as PLY file with custom properties."""
    positions = gaussian_params['positions'].detach().cpu().numpy()
    scales = gaussian_params['scales'].detach().cpu().numpy()
    rotations = gaussian_params['rotations'].detach().cpu().numpy()
    opacities = gaussian_params['opacities'].detach().cpu().numpy()
    
    # Create structured array for PLY
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('scale_x', 'f4'), ('scale_y', 'f4'), ('scale_z', 'f4'),
        ('rot_w', 'f4'), ('rot_x', 'f4'), ('rot_y', 'f4'), ('rot_z', 'f4'),
        ('opacity', 'f4')
    ]
    
    data = np.zeros(len(positions), dtype=dtype)
    data['x'] = positions[:, 0]
    data['y'] = positions[:, 1]
    data['z'] = positions[:, 2]
    data['scale_x'] = scales[:, 0]
    data['scale_y'] = scales[:, 1]
    data['scale_z'] = scales[:, 2]
    data['rot_w'] = rotations[:, 0]
    data['rot_x'] = rotations[:, 1]
    data['rot_y'] = rotations[:, 2]
    data['rot_z'] = rotations[:, 3]
    data['opacity'] = opacities.flatten()
    
    # Create PLY element
    vertex_element = PlyElement.describe(data, 'vertex')
    
    # Create PLY data and save
    ply_data = PlyData([vertex_element])
    ply_data.write(save_path)
    print(f"Saved Gaussian PLY to {save_path}")


def create_3d_scatter_plot(gaussian_params: Dict, save_path: str, 
                          max_points: int = 5000):
    """Create 3D scatter plot of Gaussian positions."""
    positions = gaussian_params['positions'].detach().cpu().numpy()
    opacities = gaussian_params['opacities'].detach().cpu().numpy().flatten()
    
    # Sample points if too many
    if len(positions) > max_points:
        indices = np.random.choice(len(positions), max_points, replace=False)
        positions = positions[indices]
        opacities = opacities[indices]
    
    # Create 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                        c=opacities, cmap='viridis', alpha=0.6, s=10)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Gaussian Positions')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Opacity')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 3D scatter plot to {save_path}")


def visualize_gaussian_ellipsoids(gaussian_params: Dict, save_path: str,
                                 max_ellipsoids: int = 100):
    """Visualize Gaussian ellipsoids using Open3D."""
    positions = gaussian_params['positions'].detach().cpu().numpy()
    scales = gaussian_params['scales'].detach().cpu().numpy()
    rotations = gaussian_params['rotations'].detach().cpu().numpy()
    opacities = gaussian_params['opacities'].detach().cpu().numpy().flatten()
    
    # Sample ellipsoids if too many
    if len(positions) > max_ellipsoids:
        indices = np.random.choice(len(positions), max_ellipsoids, replace=False)
        positions = positions[indices]
        scales = scales[indices]
        rotations = rotations[indices]
        opacities = opacities[indices]
    
    # Create visualization geometries
    geometries = []
    
    for i in range(len(positions)):
        # Create ellipsoid
        ellipsoid = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=10)
        
        # Scale
        ellipsoid.scale(scales[i, 0], scales[i, 1], scales[i, 2])
        
        # Rotate (simplified - using quaternion to rotation matrix)
        # This is a simplified rotation - in practice you'd want proper quaternion handling
        ellipsoid.rotate(o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz([0, 0, 0]))
        
        # Translate
        ellipsoid.translate(positions[i])
        
        # Set color based on opacity
        color = [opacities[i], 1 - opacities[i], 0.5]
        ellipsoid.paint_uniform_color(color)
        
        geometries.append(ellipsoid)
    
    # Combine all geometries
    combined_mesh = o3d.geometry.TriangleMesh()
    for geom in geometries:
        combined_mesh += geom
    
    # Save
    o3d.io.write_triangle_mesh(save_path, combined_mesh)
    print(f"Saved Gaussian ellipsoids to {save_path}")


def create_training_plots(train_losses: list, val_losses: list, train_psnr: list, 
                         val_psnr: list, save_dir: str):
    """Create training plots and save them."""
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # PSNR plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_psnr, 'b-', label='Training PSNR')
    plt.plot(epochs, val_psnr, 'r-', label='Validation PSNR')
    plt.title('Training and Validation PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'psnr_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_render_comparison(rendered: torch.Tensor, target: torch.Tensor, 
                          save_path: str, title: str = "Rendered vs Target"):
    """Save a detailed comparison between rendered and target images."""
    if rendered.dim() == 3:
        rendered = rendered.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)
    
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rendered = torch.clamp(rendered * std + mean, 0, 1)
    target = torch.clamp(target * std + mean, 0, 1)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Rendered image
    axes[0].imshow(rendered[0].permute(1, 2, 0).cpu().numpy())
    axes[0].set_title('Rendered')
    axes[0].axis('off')
    
    # Target image
    axes[1].imshow(target[0].permute(1, 2, 0).cpu().numpy())
    axes[1].set_title('Target')
    axes[1].axis('off')
    
    # Difference
    diff = torch.abs(rendered[0] - target[0])
    axes[2].imshow(diff.permute(1, 2, 0).cpu().numpy(), cmap='hot')
    axes[2].set_title('Difference')
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved render comparison to {save_path}")


def create_gaussian_heatmap(gaussian_params: Dict, save_path: str, 
                           resolution: int = 100):
    """Create a 2D heatmap of Gaussian density."""
    positions = gaussian_params['positions'].detach().cpu().numpy()
    opacities = gaussian_params['opacities'].detach().cpu().numpy().flatten()
    
    # Create 2D histogram
    x = positions[:, 0]
    y = positions[:, 1]
    weights = opacities
    
    # Create bins
    x_bins = np.linspace(x.min(), x.max(), resolution)
    y_bins = np.linspace(y.min(), y.max(), resolution)
    
    # Create histogram
    heatmap, x_edges, y_edges = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=weights)
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap.T, origin='lower', cmap='viridis', 
               extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    plt.colorbar(label='Gaussian Density')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Gaussian Density Heatmap')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Gaussian heatmap to {save_path}")
