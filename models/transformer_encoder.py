import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Tuple, Optional
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of multi-head attention."""
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(context)
        return output


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward network."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of transformer block."""
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class CNNBackbone(nn.Module):
    """CNN backbone for feature extraction from images."""
    
    def __init__(self, in_channels: int = 3, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        
        # ResNet-like architecture
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, feature_dim, 2, stride=2)
        
        # Global average pooling and projection
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(feature_dim, feature_dim)
    
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1):
        """Create a layer of residual blocks."""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.projection(x)
        
        return x


class ResidualBlock(nn.Module):
    """Residual block for CNN backbone."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        
        return out


class TransformerEncoder(nn.Module):
    """Transformer-based encoder for multi-view feature processing."""
    
    def __init__(self, feature_dim: int = 256, num_layers: int = 6, num_heads: int = 8,
                 d_ff: int = 1024, dropout: float = 0.1, use_positional_encoding: bool = True):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_positional_encoding = use_positional_encoding
        
        # CNN backbone for feature extraction
        self.cnn_backbone = CNNBackbone(in_channels=3, feature_dim=feature_dim)
        
        # Positional encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(feature_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(feature_dim, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Process multi-view images through transformer encoder."""
        batch_size, num_views, channels, height, width = images.shape
        
        # Extract features from each view
        features = []
        for i in range(num_views):
            view_features = self.cnn_backbone(images[:, i])  # (B, feature_dim)
            features.append(view_features)
        
        # Stack features: (num_views, batch_size, feature_dim)
        features = torch.stack(features, dim=0)
        
        # Add positional encoding if enabled
        if self.use_positional_encoding:
            features = self.pos_encoding(features)
        
        # Apply transformer layers
        for transformer_layer in self.transformer_layers:
            features = transformer_layer(features)
        
        # Average across views and project
        features = features.mean(dim=0)  # (batch_size, feature_dim)
        features = self.output_projection(features)
        features = self.dropout(features)
        
        return features


class GaussianPredictor(nn.Module):
    """Predict 3D Gaussian parameters from encoded features."""
    
    def __init__(self, feature_dim: int = 256, num_gaussians: int = 10000):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_gaussians = num_gaussians
        
        # Feature processing
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Predict Gaussian parameters
        self.position_predictor = nn.Linear(256, num_gaussians * 3)
        self.scale_predictor = nn.Linear(256, num_gaussians * 3)
        self.rotation_predictor = nn.Linear(256, num_gaussians * 4)
        self.opacity_predictor = nn.Linear(256, num_gaussians * 1)
        self.feature_predictor = nn.Linear(256, num_gaussians * feature_dim)
    
    def forward(self, features: torch.Tensor) -> dict:
        """Predict Gaussian parameters from encoded features."""
        processed_features = self.feature_mlp(features)
        
        # Predict parameters
        positions = self.position_predictor(processed_features).view(-1, self.num_gaussians, 3)
        scales = self.scale_predictor(processed_features).view(-1, self.num_gaussians, 3)
        rotations = self.rotation_predictor(processed_features).view(-1, self.num_gaussians, 4)
        opacities = self.opacity_predictor(processed_features).view(-1, self.num_gaussians, 1)
        gaussian_features = self.feature_predictor(processed_features).view(-1, self.num_gaussians, self.feature_dim)
        
        return {
            'positions': positions,
            'scales': torch.exp(scales),  # Ensure positive scales
            'rotations': F.normalize(rotations, dim=-1),  # Normalize quaternions
            'opacities': torch.sigmoid(opacities),  # Ensure opacities in [0, 1]
            'features': gaussian_features
        }


class SparseViewReconstructionModel(nn.Module):
    """Complete sparse-view 3D reconstruction model with transformer encoder."""
    
    def __init__(self, num_gaussians: int = 10000, feature_dim: int = 256,
                 transformer_layers: int = 6, num_heads: int = 8, dropout: float = 0.1,
                 near_plane: float = 0.1, far_plane: float = 10.0):
        super().__init__()
        
        # Transformer encoder
        self.encoder = TransformerEncoder(
            feature_dim=feature_dim,
            num_layers=transformer_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Gaussian predictor
        self.gaussian_predictor = GaussianPredictor(feature_dim, num_gaussians)
        
        # Gaussian renderer
        self.renderer = GaussianRenderer(feature_dim, near_plane, far_plane)
    
    def forward(self, images: torch.Tensor, camera_matrix: torch.Tensor,
                camera_pose: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """Forward pass: encode images, predict Gaussians, and render."""
        # Encode multi-view images
        features = self.encoder(images)
        
        # Predict Gaussian parameters
        gaussian_params = self.gaussian_predictor(features)
        
        # Render from predicted Gaussians
        rendered_image = self.renderer(gaussian_params, camera_matrix, camera_pose, image_size)
        
        return rendered_image
    
    def encode_features(self, images: torch.Tensor) -> torch.Tensor:
        """Encode multi-view images to features."""
        return self.encoder(images)
    
    def predict_gaussians(self, features: torch.Tensor) -> dict:
        """Predict Gaussian parameters from features."""
        return self.gaussian_predictor(features)
