#!/usr/bin/env python3
"""
Test script to verify that all components of the 3D Gaussian Splatting project
are working correctly.
"""

import sys
import torch
import numpy as np
import logging

# Add project root to path
sys.path.append('.')

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from models.transformer_encoder import SparseViewReconstructionModel
        from models.gaussian_splatting import GaussianSplattingModel
        from data.dataset import MultiViewDataset, SyntheticDataset
        from utils.losses import TotalLoss, compute_metrics
        from utils.visualization import save_images, create_3d_scatter_plot
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_model_creation():
    """Test that models can be created."""
    print("\nTesting model creation...")
    
    try:
        # Test transformer model
        model = SparseViewReconstructionModel(
            num_gaussians=100,
            feature_dim=64,
            transformer_layers=2,
            num_heads=4,
            dropout=0.1
        )
        print("✓ Transformer model created successfully")
        
        # Test basic Gaussian model
        gaussian_model = GaussianSplattingModel(
            num_gaussians=100,
            feature_dim=64
        )
        print("✓ Gaussian model created successfully")
        
        return True
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        return False

def test_forward_pass():
    """Test that models can perform forward pass."""
    print("\nTesting forward pass...")
    
    try:
        # Create model
        model = SparseViewReconstructionModel(
            num_gaussians=100,
            feature_dim=64,
            transformer_layers=2,
            num_heads=4,
            dropout=0.1
        )
        
        # Create dummy data
        batch_size = 2
        num_views = 3
        image_size = (64, 64)
        
        images = torch.randn(batch_size, num_views, 3, *image_size)
        camera_matrix = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
        camera_pose = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Forward pass
        with torch.no_grad():
            output = model(images, camera_matrix, camera_pose, image_size)
        
        print(f"✓ Forward pass successful. Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"✗ Forward pass error: {e}")
        return False

def test_dataset():
    """Test that datasets can be created and used."""
    print("\nTesting dataset...")
    
    try:
        # Test synthetic dataset
        dataset = SyntheticDataset(
            num_samples=5,
            num_views=4,
            image_size=(64, 64)
        )
        
        # Get a sample
        sample = dataset[0]
        print(f"✓ Synthetic dataset created. Sample keys: {list(sample.keys())}")
        print(f"  Images shape: {sample['images'].shape}")
        print(f"  Camera matrices shape: {sample['camera_matrices'].shape}")
        print(f"  Camera poses shape: {sample['camera_poses'].shape}")
        
        return True
    except Exception as e:
        print(f"✗ Dataset error: {e}")
        return False

def test_loss_functions():
    """Test that loss functions work."""
    print("\nTesting loss functions...")
    
    try:
        # Create loss function
        loss_config = {
            'reconstruction_weight': 1.0,
            'ssim_weight': 0.1,
            'depth_weight': 0.1,
            'smoothness_weight': 0.01,
            'sparsity_weight': 0.001,
            'regularization_weight': 0.01
        }
        criterion = TotalLoss(loss_config)
        
        # Create dummy data
        rendered = torch.randn(2, 3, 64, 64)
        target = torch.randn(2, 3, 64, 64)
        gaussian_params = {
            'positions': torch.randn(2, 100, 3),
            'scales': torch.randn(2, 100, 3),
            'rotations': torch.randn(2, 100, 4),
            'opacities': torch.randn(2, 100, 1),
            'features': torch.randn(2, 100, 64)
        }
        
        # Compute loss
        losses = criterion(rendered, target, gaussian_params)
        print(f"✓ Loss computation successful. Total loss: {losses['total']:.4f}")
        
        # Test metrics
        metrics = compute_metrics(rendered, target)
        print(f"✓ Metrics computation successful. PSNR: {metrics['psnr']:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Loss function error: {e}")
        return False

def test_device_compatibility():
    """Test that models work on available devices."""
    print("\nTesting device compatibility...")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Create model on device
        model = SparseViewReconstructionModel(
            num_gaussians=50,
            feature_dim=32,
            transformer_layers=1,
            num_heads=2,
            dropout=0.1
        ).to(device)
        
        # Create data on device
        images = torch.randn(1, 3, 3, 32, 32).to(device)
        camera_matrix = torch.eye(3).unsqueeze(0).to(device)
        camera_pose = torch.eye(4).unsqueeze(0).to(device)
        
        # Forward pass
        with torch.no_grad():
            output = model(images, camera_matrix, camera_pose, (32, 32))
        
        print(f"✓ Device compatibility test passed. Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"✗ Device compatibility error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("3D Gaussian Splatting - Installation Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_model_creation,
        test_forward_pass,
        test_dataset,
        test_loss_functions,
        test_device_compatibility
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Installation is successful.")
        print("\nYou can now run:")
        print("  python demo.py                    # Run demo with synthetic data")
        print("  python train.py --use_synthetic   # Train with synthetic data")
        print("  python inference.py --help        # See inference options")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    
    print("=" * 60)

if __name__ == '__main__':
    main()
