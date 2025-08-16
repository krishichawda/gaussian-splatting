# Quick Start Guide

This guide will help you get started with the Generalizable Sparse-View 3D Object Reconstruction using 3D Gaussian Splatting.

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test installation:**
   ```bash
   python test_installation.py
   ```

## Quick Demo

Run the demo with synthetic data to see the system in action:

```bash
python demo.py
```

This will:
- Create a synthetic dataset
- Train a small model for 5 epochs
- Generate visualizations
- Save results to `demo_results/`

## Training

### With Synthetic Data (for testing)
```bash
python train.py --use_synthetic --config configs/train_config.yaml
```

### With Real Data
1. Prepare your data in the following structure:
   ```
   data/
   ├── train/
   │   ├── scene1/
   │   │   ├── image1.jpg
   │   │   ├── image2.jpg
   │   │   ├── image3.jpg
   │   │   ├── image4.jpg
   │   │   └── cameras.json (optional)
   │   └── scene2/
   │       └── ...
   ├── val/
   └── test/
   ```

2. Run training:
   ```bash
   python train.py --config configs/train_config.yaml
   ```

## Inference

Reconstruct 3D objects from new images:

```bash
python inference.py \
    --checkpoint logs/checkpoints/best_checkpoint.pth \
    --input_dir path/to/your/images \
    --output_dir outputs \
    --num_views 4
```

## Evaluation

Evaluate a trained model:

```bash
python evaluate.py \
    --checkpoint logs/checkpoints/best_checkpoint.pth \
    --test_dir data/test \
    --output_dir evaluation_results
```

## Configuration

Edit `configs/train_config.yaml` to customize:

- **Model parameters**: Number of Gaussians, feature dimensions, transformer layers
- **Training parameters**: Learning rate, batch size, number of epochs
- **Loss weights**: Reconstruction, SSIM, depth, smoothness, sparsity
- **Hardware settings**: Device, mixed precision, number of GPUs

## Key Features

- **Sparse-View Reconstruction**: Works with only 4 input views
- **Transformer Architecture**: Uses attention mechanisms for better feature processing
- **3D Gaussian Splatting**: Efficient rendering using 3D Gaussian primitives
- **Unposed Inputs**: Handles unposed camera inputs
- **Real-time Rendering**: Fast inference for interactive applications

## Output Files

### Training
- `logs/`: Training logs and TensorBoard files
- `logs/checkpoints/`: Model checkpoints
- `logs/epoch_*.png`: Sample rendered images during training

### Inference
- `outputs/*_rendered.png`: Rendered images
- `outputs/*_gaussians.ply`: 3D Gaussian parameters
- `outputs/*_pointcloud.ply`: Point cloud visualization
- `outputs/*_scatter.png`: 3D scatter plot
- `outputs/*_heatmap.png`: 2D density heatmap
- `outputs/*_novel_views.png`: Novel view renderings

### Evaluation
- `evaluation_results/metrics_distributions.png`: Metrics distributions
- `evaluation_results/losses_distributions.png`: Loss distributions
- `evaluation_results/correlation_matrices.png`: Correlation analysis
- `evaluation_results/evaluation_statistics.json`: Statistical summary

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or number of Gaussians
2. **Import errors**: Make sure all dependencies are installed
3. **Slow training**: Use GPU acceleration and reduce image size

### Performance Tips

- Use GPU acceleration for faster training
- Start with synthetic data to verify setup
- Adjust number of Gaussians based on your needs
- Use smaller image sizes for faster training

## Next Steps

1. **Customize the model**: Modify architecture in `models/`
2. **Add new loss functions**: Extend `utils/losses.py`
3. **Support new data formats**: Extend `data/dataset.py`
4. **Improve visualization**: Enhance `utils/visualization.py`

## Citation

If you use this code, please cite:

```bibtex
@article{gaussian_splatting_2024,
  title={Generalizable Sparse-View 3D Object Reconstruction with 3D Gaussian Splatting},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```
