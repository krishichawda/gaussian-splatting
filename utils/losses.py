import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import math


class ReconstructionLoss(nn.Module):
    """Reconstruction loss for comparing rendered and target images."""
    
    def __init__(self, loss_type: str = 'l1', reduction: str = 'mean'):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        
        if loss_type == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif loss_type == 'l2':
            self.criterion = nn.MSELoss(reduction=reduction)
        elif loss_type == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss(reduction=reduction)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, rendered: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Compute reconstruction loss."""
        if mask is not None:
            # Apply mask if provided
            rendered = rendered * mask
            target = target * mask
        
        return self.criterion(rendered, target)


class SSIMLoss(nn.Module):
    """SSIM (Structural Similarity Index) loss."""
    
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        
        # Create Gaussian window
        self.register_buffer('window', self._create_window(window_size, sigma))
    
    def _create_window(self, window_size: int, sigma: float) -> torch.Tensor:
        """Create Gaussian window for SSIM computation."""
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
            return gauss/gauss.sum()
        
        _1D_window = gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(3, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss."""
        # SSIM parameters
        C1 = 0.01**2
        C2 = 0.03**2
        
        mu1 = F.conv2d(rendered, self.window, padding=self.window_size//2, groups=3)
        mu2 = F.conv2d(target, self.window, padding=self.window_size//2, groups=3)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(rendered * rendered, self.window, padding=self.window_size//2, groups=3) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.window, padding=self.window_size//2, groups=3) - mu2_sq
        sigma12 = F.conv2d(rendered * target, self.window, padding=self.window_size//2, groups=3) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim_map.mean()


class DepthLoss(nn.Module):
    """Depth consistency loss for 3D reconstruction."""
    
    def __init__(self, loss_type: str = 'l1'):
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif loss_type == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown depth loss type: {loss_type}")
    
    def forward(self, rendered_depth: torch.Tensor, target_depth: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        """Compute depth consistency loss."""
        if mask is not None:
            rendered_depth = rendered_depth * mask
            target_depth = target_depth * mask
        
        return self.criterion(rendered_depth, target_depth)


class SmoothnessLoss(nn.Module):
    """Smoothness loss to encourage spatial coherence."""
    
    def __init__(self, loss_type: str = 'l1'):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Compute smoothness loss using gradients."""
        # Compute gradients
        grad_x = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])
        grad_y = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])
        
        if self.loss_type == 'l1':
            loss = grad_x.mean() + grad_y.mean()
        elif self.loss_type == 'l2':
            loss = (grad_x**2).mean() + (grad_y**2).mean()
        else:
            raise ValueError(f"Unknown smoothness loss type: {self.loss_type}")
        
        return loss


class SparsityLoss(nn.Module):
    """Sparsity loss to encourage sparse Gaussian distributions."""
    
    def __init__(self, loss_type: str = 'l1'):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(self, gaussian_params: Dict) -> torch.Tensor:
        """Compute sparsity loss on Gaussian parameters."""
        positions = gaussian_params['positions']
        opacities = gaussian_params['opacities']
        
        # Encourage sparsity in positions and opacities
        if self.loss_type == 'l1':
            pos_sparsity = torch.abs(positions).mean()
            opacity_sparsity = torch.abs(opacities).mean()
        elif self.loss_type == 'l2':
            pos_sparsity = (positions**2).mean()
            opacity_sparsity = (opacities**2).mean()
        else:
            raise ValueError(f"Unknown sparsity loss type: {self.loss_type}")
        
        return pos_sparsity + opacity_sparsity


class GaussianRegularizationLoss(nn.Module):
    """Regularization loss for Gaussian parameters."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, gaussian_params: Dict) -> torch.Tensor:
        """Compute regularization loss for Gaussian parameters."""
        positions = gaussian_params['positions']
        scales = gaussian_params['scales']
        rotations = gaussian_params['rotations']
        opacities = gaussian_params['opacities']
        
        # Position regularization (encourage Gaussians to stay within bounds)
        pos_norm = torch.norm(positions, dim=-1)
        pos_reg = torch.mean(torch.relu(pos_norm - 2.0))  # Penalize positions outside unit sphere
        
        # Scale regularization (encourage reasonable scales)
        scale_reg = torch.mean(torch.relu(scales - 1.0))  # Penalize large scales
        
        # Rotation regularization (encourage unit quaternions)
        rot_norm = torch.norm(rotations, dim=-1)
        rot_reg = torch.mean((rot_norm - 1.0)**2)  # Penalize non-unit quaternions
        
        # Opacity regularization (encourage binary opacities)
        opacity_reg = torch.mean(opacities * (1 - opacities))  # Penalize non-binary opacities
        
        return pos_reg + scale_reg + rot_reg + opacity_reg


class TotalLoss(nn.Module):
    """Combined loss function for training."""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Initialize individual loss components
        self.reconstruction_loss = ReconstructionLoss(
            loss_type=config.get('reconstruction_type', 'l1')
        )
        self.ssim_loss = SSIMLoss()
        self.depth_loss = DepthLoss(
            loss_type=config.get('depth_type', 'l1')
        )
        self.smoothness_loss = SmoothnessLoss(
            loss_type=config.get('smoothness_type', 'l1')
        )
        self.sparsity_loss = SparsityLoss(
            loss_type=config.get('sparsity_type', 'l1')
        )
        self.regularization_loss = GaussianRegularizationLoss()
        
        # Loss weights
        self.reconstruction_weight = config.get('reconstruction_weight', 1.0)
        self.ssim_weight = config.get('ssim_weight', 0.1)
        self.depth_weight = config.get('depth_weight', 0.1)
        self.smoothness_weight = config.get('smoothness_weight', 0.01)
        self.sparsity_weight = config.get('sparsity_weight', 0.001)
        self.regularization_weight = config.get('regularization_weight', 0.01)
    
    def forward(self, rendered: torch.Tensor, target: torch.Tensor, 
                gaussian_params: Dict, rendered_depth: torch.Tensor = None,
                target_depth: torch.Tensor = None, mask: torch.Tensor = None) -> Dict:
        """Compute total loss."""
        losses = {}
        
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(rendered, target, mask)
        losses['reconstruction'] = recon_loss
        
        # SSIM loss
        ssim_loss = self.ssim_loss(rendered, target)
        losses['ssim'] = ssim_loss
        
        # Depth loss (if depth information is available)
        if rendered_depth is not None and target_depth is not None:
            depth_loss = self.depth_loss(rendered_depth, target_depth, mask)
            losses['depth'] = depth_loss
        else:
            depth_loss = torch.tensor(0.0, device=rendered.device)
            losses['depth'] = depth_loss
        
        # Smoothness loss
        smoothness_loss = self.smoothness_loss(rendered)
        losses['smoothness'] = smoothness_loss
        
        # Sparsity loss
        sparsity_loss = self.sparsity_loss(gaussian_params)
        losses['sparsity'] = sparsity_loss
        
        # Regularization loss
        reg_loss = self.regularization_loss(gaussian_params)
        losses['regularization'] = reg_loss
        
        # Total loss
        total_loss = (
            self.reconstruction_weight * recon_loss +
            self.ssim_weight * ssim_loss +
            self.depth_weight * depth_loss +
            self.smoothness_weight * smoothness_loss +
            self.sparsity_weight * sparsity_loss +
            self.regularization_weight * reg_loss
        )
        
        losses['total'] = total_loss
        
        return losses


def compute_psnr(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = F.mse_loss(rendered, target)
    if mse == 0:
        return torch.tensor(float('inf'))
    max_val = torch.max(target)
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr


def compute_ssim(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Structural Similarity Index."""
    ssim_loss = SSIMLoss()
    return 1 - ssim_loss(rendered, target)


def compute_metrics(rendered: torch.Tensor, target: torch.Tensor) -> Dict:
    """Compute various image quality metrics."""
    metrics = {}
    
    # PSNR
    metrics['psnr'] = compute_psnr(rendered, target)
    
    # SSIM
    metrics['ssim'] = compute_ssim(rendered, target)
    
    # L1 error
    metrics['l1_error'] = F.l1_loss(rendered, target)
    
    # L2 error
    metrics['l2_error'] = F.mse_loss(rendered, target)
    
    return metrics
