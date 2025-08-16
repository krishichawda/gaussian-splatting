import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import json
from typing import List, Tuple, Dict, Optional
import cv2
from pathlib import Path


class MultiViewDataset(Dataset):
    """Dataset for multi-view 3D reconstruction."""
    
    def __init__(self, data_dir: str, num_views: int = 4, image_size: Tuple[int, int] = (256, 256),
                 transform: Optional[transforms.Compose] = None, split: str = 'train'):
        self.data_dir = Path(data_dir)
        self.num_views = num_views
        self.image_size = image_size
        self.split = split
        
        # Default transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # Load scene information
        self.scenes = self._load_scenes()
        
    def _load_scenes(self) -> List[Dict]:
        """Load scene information from data directory."""
        scenes = []
        
        # Look for scene directories
        for scene_dir in self.data_dir.iterdir():
            if scene_dir.is_dir():
                scene_info = self._load_scene_info(scene_dir)
                if scene_info is not None:
                    scenes.append(scene_info)
        
        return scenes
    
    def _load_scene_info(self, scene_dir: Path) -> Optional[Dict]:
        """Load information for a single scene."""
        # Look for images and camera information
        image_files = list(scene_dir.glob("*.jpg")) + list(scene_dir.glob("*.png"))
        
        if len(image_files) < self.num_views:
            return None
        
        # Try to load camera information
        camera_file = scene_dir / "cameras.json"
        if camera_file.exists():
            with open(camera_file, 'r') as f:
                camera_data = json.load(f)
        else:
            # Generate default camera parameters
            camera_data = self._generate_default_cameras(len(image_files))
        
        return {
            'scene_dir': scene_dir,
            'image_files': image_files,
            'camera_data': camera_data
        }
    
    def _generate_default_cameras(self, num_images: int) -> Dict:
        """Generate default camera parameters for unposed images."""
        cameras = {}
        
        # Generate cameras in a circle around the object
        for i in range(num_images):
            angle = 2 * np.pi * i / num_images
            distance = 2.0
            
            # Camera position
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            z = 1.0
            
            # Look at origin
            look_at = np.array([0, 0, 0])
            up = np.array([0, 0, 1])
            
            # Create camera matrix
            camera_matrix = np.array([
                [1.0, 0.0, 256/2],
                [0.0, 1.0, 256/2],
                [0.0, 0.0, 1.0]
            ])
            
            # Create camera pose (simplified)
            cameras[f"camera_{i}"] = {
                'camera_matrix': camera_matrix.tolist(),
                'position': [x, y, z],
                'look_at': look_at.tolist(),
                'up': up.tolist()
            }
        
        return cameras
    
    def __len__(self) -> int:
        return len(self.scenes)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample from the dataset."""
        scene = self.scenes[idx]
        
        # Randomly select views
        num_available = len(scene['image_files'])
        if num_available >= self.num_views:
            selected_indices = np.random.choice(num_available, self.num_views, replace=False)
        else:
            # If not enough views, sample with replacement
            selected_indices = np.random.choice(num_available, self.num_views, replace=True)
        
        # Load images and camera parameters
        images = []
        camera_matrices = []
        camera_poses = []
        
        for i, img_idx in enumerate(selected_indices):
            # Load image
            img_path = scene['image_files'][img_idx]
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            images.append(image)
            
            # Load camera parameters
            camera_key = f"camera_{img_idx}"
            if camera_key in scene['camera_data']:
                camera_info = scene['camera_data'][camera_key]
                camera_matrix = torch.tensor(camera_info['camera_matrix'], dtype=torch.float32)
                camera_pose = self._create_camera_pose(camera_info)
            else:
                # Use default camera
                camera_matrix = torch.eye(3, dtype=torch.float32)
                camera_pose = torch.eye(4, dtype=torch.float32)
            
            camera_matrices.append(camera_matrix)
            camera_poses.append(camera_pose)
        
        # Stack tensors
        images = torch.stack(images)  # (num_views, C, H, W)
        camera_matrices = torch.stack(camera_matrices)  # (num_views, 3, 3)
        camera_poses = torch.stack(camera_poses)  # (num_views, 4, 4)
        
        return {
            'images': images,
            'camera_matrices': camera_matrices,
            'camera_poses': camera_poses,
            'scene_id': scene['scene_dir'].name
        }
    
    def _create_camera_pose(self, camera_info: Dict) -> torch.Tensor:
        """Create camera pose matrix from camera information."""
        position = torch.tensor(camera_info['position'], dtype=torch.float32)
        look_at = torch.tensor(camera_info['look_at'], dtype=torch.float32)
        up = torch.tensor(camera_info['up'], dtype=torch.float32)
        
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
        
        return camera_pose


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing with simple geometric shapes."""
    
    def __init__(self, num_samples: int = 1000, num_views: int = 4, 
                 image_size: Tuple[int, int] = (256, 256)):
        self.num_samples = num_samples
        self.num_views = num_views
        self.image_size = image_size
        
        # Generate synthetic data
        self.data = self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> List[Dict]:
        """Generate synthetic multi-view data."""
        data = []
        
        for i in range(self.num_samples):
            # Generate random camera positions around a sphere
            camera_positions = []
            camera_matrices = []
            camera_poses = []
            images = []
            
            for j in range(self.num_views):
                # Random camera position on a sphere
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, np.pi)
                radius = 3.0
                
                x = radius * np.sin(phi) * np.cos(theta)
                y = radius * np.sin(phi) * np.sin(theta)
                z = radius * np.cos(phi)
                
                position = torch.tensor([x, y, z], dtype=torch.float32)
                
                # Look at origin
                look_at = torch.zeros(3, dtype=torch.float32)
                up = torch.tensor([0, 0, 1], dtype=torch.float32)
                
                # Create camera pose
                z_axis = F.normalize(look_at - position, dim=0)
                x_axis = F.normalize(torch.cross(up, z_axis), dim=0)
                y_axis = torch.cross(z_axis, x_axis)
                
                rotation = torch.stack([x_axis, y_axis, z_axis], dim=1)
                camera_pose = torch.eye(4, dtype=torch.float32)
                camera_pose[:3, :3] = rotation
                camera_pose[:3, 3] = position
                
                # Create camera matrix
                fov = 60.0
                focal_length = self.image_size[0] / (2 * np.tan(np.radians(fov) / 2))
                camera_matrix = torch.tensor([
                    [focal_length, 0, self.image_size[0] / 2],
                    [0, focal_length, self.image_size[1] / 2],
                    [0, 0, 1]
                ], dtype=torch.float32)
                
                # Generate synthetic image (simple pattern)
                image = self._generate_synthetic_image(camera_pose)
                
                camera_positions.append(position)
                camera_matrices.append(camera_matrix)
                camera_poses.append(camera_pose)
                images.append(image)
            
            data.append({
                'images': torch.stack(images),
                'camera_matrices': torch.stack(camera_matrices),
                'camera_poses': torch.stack(camera_poses),
                'scene_id': f'synthetic_{i}'
            })
        
        return data
    
    def _generate_synthetic_image(self, camera_pose: torch.Tensor) -> torch.Tensor:
        """Generate a synthetic image based on camera pose."""
        H, W = self.image_size
        
        # Create a simple pattern
        image = torch.zeros(3, H, W)
        
        # Add some geometric patterns
        for i in range(H):
            for j in range(W):
                # Normalized coordinates
                u = (j - W/2) / (W/2)
                v = (i - H/2) / (H/2)
                
                # Simple color pattern
                r = 0.5 + 0.5 * torch.sin(u * 10 + camera_pose[0, 3])
                g = 0.5 + 0.5 * torch.cos(v * 10 + camera_pose[1, 3])
                b = 0.5 + 0.5 * torch.sin((u + v) * 5 + camera_pose[2, 3])
                
                image[0, i, j] = r
                image[1, i, j] = g
                image[2, i, j] = b
        
        return image
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]


def create_dataloader(data_dir: str, batch_size: int = 4, num_views: int = 4,
                     image_size: Tuple[int, int] = (256, 256), num_workers: int = 4,
                     split: str = 'train', use_synthetic: bool = False) -> DataLoader:
    """Create a dataloader for the dataset."""
    
    if use_synthetic:
        dataset = SyntheticDataset(
            num_samples=1000 if split == 'train' else 100,
            num_views=num_views,
            image_size=image_size
        )
    else:
        dataset = MultiViewDataset(
            data_dir=data_dir,
            num_views=num_views,
            image_size=image_size,
            split=split
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for batching."""
    # Stack all tensors
    images = torch.stack([item['images'] for item in batch])
    camera_matrices = torch.stack([item['camera_matrices'] for item in batch])
    camera_poses = torch.stack([item['camera_poses'] for item in batch])
    scene_ids = [item['scene_id'] for item in batch]
    
    return {
        'images': images,  # (B, num_views, C, H, W)
        'camera_matrices': camera_matrices,  # (B, num_views, 3, 3)
        'camera_poses': camera_poses,  # (B, num_views, 4, 4)
        'scene_ids': scene_ids
    }
