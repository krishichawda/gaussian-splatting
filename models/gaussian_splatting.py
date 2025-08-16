import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math


class Gaussian3D(nn.Module):
    """3D Gaussian representation with learnable parameters."""
    
    def __init__(self, num_gaussians: int, feature_dim: int = 256):
        super().__init__()
        self.num_gaussians = num_gaussians
        
        # Learnable parameters for each Gaussian
        self.positions = nn.Parameter(torch.randn(num_gaussians, 3) * 0.1)
        self.scales = nn.Parameter(torch.ones(num_gaussians, 3) * 0.1)
        self.rotations = nn.Parameter(torch.randn(num_gaussians, 4))  # Quaternions
        self.opacities = nn.Parameter(torch.sigmoid(torch.randn(num_gaussians, 1)))
        self.features = nn.Parameter(torch.randn(num_gaussians, feature_dim) * 0.1)
        
        # Initialize rotations as unit quaternions
        with torch.no_grad():
            self.rotations.data = F.normalize(self.rotations.data, dim=-1)
    
    def forward(self):
        """Return all Gaussian parameters."""
        return {
            'positions': self.positions,
            'scales': torch.exp(self.scales),  # Ensure positive scales
            'rotations': F.normalize(self.rotations, dim=-1),  # Normalize quaternions
            'opacities': torch.sigmoid(self.opacities),  # Ensure opacities in [0, 1]
            'features': self.features
        }


class GaussianRenderer(nn.Module):
    """3D Gaussian Splatting renderer."""
    
    def __init__(self, feature_dim: int = 256, near_plane: float = 0.1, far_plane: float = 10.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.near_plane = near_plane
        self.far_plane = far_plane
        
        # Feature to RGB mapping
        self.feature_to_rgb = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )
    
    def quaternion_to_rotation_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """Convert quaternions to rotation matrices."""
        # q: (N, 4) -> (N, 3, 3)
        q = F.normalize(q, dim=-1)
        w, x, y, z = q.unbind(-1)
        
        return torch.stack([
            1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y,
            2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x,
            2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y
        ], dim=-1).view(-1, 3, 3)
    
    def compute_covariance_matrix(self, scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
        """Compute 3D covariance matrix for each Gaussian."""
        # scales: (N, 3), rotations: (N, 4)
        R = self.quaternion_to_rotation_matrix(rotations)  # (N, 3, 3)
        S = torch.diag_embed(scales)  # (N, 3, 3)
        
        # Covariance = R * S * S^T * R^T
        cov = R @ S @ S.transpose(-2, -1) @ R.transpose(-2, -1)
        return cov
    
    def project_gaussians(self, positions: torch.Tensor, covariances: torch.Tensor, 
                         camera_matrix: torch.Tensor, camera_pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project 3D Gaussians to 2D image plane."""
        # Transform positions to camera space
        R = camera_pose[:3, :3]  # (3, 3)
        t = camera_pose[:3, 3]   # (3,)
        
        # Transform positions
        positions_cam = (R @ positions.T).T + t  # (N, 3)
        
        # Project to 2D
        positions_2d = positions_cam[:, :2] / positions_cam[:, 2:3]  # (N, 2)
        
        # Transform covariance matrices
        J = torch.zeros(positions.shape[0], 2, 3, device=positions.device)
        J[:, 0, 0] = 1.0 / positions_cam[:, 2]
        J[:, 1, 1] = 1.0 / positions_cam[:, 2]
        J[:, 0, 2] = -positions_cam[:, 0] / (positions_cam[:, 2] ** 2)
        J[:, 1, 2] = -positions_cam[:, 1] / (positions_cam[:, 2] ** 2)
        
        # Project covariance: cov_2d = J * cov_3d * J^T
        cov_2d = J @ covariances @ J.transpose(-2, -1)
        
        return positions_2d, cov_2d
    
    def render_gaussians(self, positions_2d: torch.Tensor, covariances_2d: torch.Tensor,
                        opacities: torch.Tensor, colors: torch.Tensor,
                        image_size: Tuple[int, int]) -> torch.Tensor:
        """Render 2D Gaussians to image."""
        H, W = image_size
        device = positions_2d.device
        
        # Create pixel coordinates
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        pixels = torch.stack([x_coords, y_coords], dim=-1).view(-1, 2)  # (H*W, 2)
        
        # Compute Gaussian values for each pixel
        rendered = torch.zeros(H*W, 3, device=device)
        alpha_sum = torch.zeros(H*W, 1, device=device)
        
        # Sort Gaussians by depth (back to front)
        depths = positions_2d[:, 2] if positions_2d.shape[1] > 2 else torch.zeros(positions_2d.shape[0], device=device)
        sorted_indices = torch.argsort(depths, descending=True)
        
        for idx in sorted_indices:
            pos = positions_2d[idx]
            cov = covariances_2d[idx]
            opacity = opacities[idx]
            color = colors[idx]
            
            # Compute distance from pixel to Gaussian center
            diff = pixels - pos.unsqueeze(0)  # (H*W, 2)
            
            # Compute Gaussian value
            inv_cov = torch.inverse(cov + torch.eye(2, device=device) * 1e-6)
            exponent = -0.5 * torch.sum(diff @ inv_cov * diff, dim=-1)
            gaussian_val = torch.exp(exponent) / (2 * math.pi * torch.sqrt(torch.det(cov) + 1e-6))
            
            # Alpha blending
            alpha = opacity * gaussian_val.unsqueeze(-1)
            rendered += alpha * color.unsqueeze(0) * (1 - alpha_sum)
            alpha_sum += alpha * (1 - alpha_sum)
            
            # Early termination if fully opaque
            if torch.all(alpha_sum > 0.99):
                break
        
        return rendered.view(H, W, 3)
    
    def forward(self, gaussian_params: dict, camera_matrix: torch.Tensor, 
                camera_pose: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """Forward pass of the Gaussian renderer."""
        positions = gaussian_params['positions']
        scales = gaussian_params['scales']
        rotations = gaussian_params['rotations']
        opacities = gaussian_params['opacities']
        features = gaussian_params['features']
        
        # Convert features to colors
        colors = self.feature_to_rgb(features)
        
        # Compute covariance matrices
        covariances = self.compute_covariance_matrix(scales, rotations)
        
        # Project to 2D
        positions_2d, covariances_2d = self.project_gaussians(
            positions, covariances, camera_matrix, camera_pose
        )
        
        # Render
        rendered_image = self.render_gaussians(
            positions_2d, covariances_2d, opacities, colors, image_size
        )
        
        return rendered_image


class GaussianSplattingModel(nn.Module):
    """Complete 3D Gaussian Splatting model."""
    
    def __init__(self, num_gaussians: int = 10000, feature_dim: int = 256, 
                 near_plane: float = 0.1, far_plane: float = 10.0):
        super().__init__()
        self.gaussians = Gaussian3D(num_gaussians, feature_dim)
        self.renderer = GaussianRenderer(feature_dim, near_plane, far_plane)
    
    def forward(self, camera_matrix: torch.Tensor, camera_pose: torch.Tensor, 
                image_size: Tuple[int, int]) -> torch.Tensor:
        """Forward pass."""
        gaussian_params = self.gaussians()
        rendered_image = self.renderer(gaussian_params, camera_matrix, camera_pose, image_size)
        return rendered_image
    
    def get_gaussian_params(self):
        """Get current Gaussian parameters."""
        return self.gaussians()
