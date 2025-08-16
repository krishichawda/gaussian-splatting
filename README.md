# Generalizable Sparse-View 3D Object Reconstruction

A PyTorch implementation of a single-pass 3D Gaussian Splatting network for sparse-view (4-view) 3D object reconstruction from unposed inputs, using transformer blocks to improve generalization and scalability.

## Features

- **Sparse-View Reconstruction**: Reconstruct 3D objects from only 4 input views
- **Transformer Architecture**: Uses transformer blocks for better feature extraction and generalization
- **3D Gaussian Splatting**: Efficient rendering using 3D Gaussian primitives
- **Unposed Inputs**: Works with unposed camera inputs
- **Real-time Rendering**: Fast inference for interactive applications

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd gaussian-splatting
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install CUDA for GPU acceleration

## Usage

### Training

```bash
python train.py --config configs/train_config.yaml
```

### Inference

```bash
python inference.py --checkpoint path/to/checkpoint.pth --input_dir path/to/images
```

### Evaluation

```bash
python evaluate.py --checkpoint path/to/checkpoint.pth --test_dir path/to/test_data
```

## Project Structure

```
├── configs/                 # Configuration files
├── data/                   # Data loading and preprocessing
├── models/                 # Network architectures
├── utils/                  # Utility functions
├── train.py               # Training script
├── inference.py           # Inference script
├── evaluate.py            # Evaluation script
└── requirements.txt       # Dependencies
```

## Model Architecture

The model consists of:
1. **Feature Encoder**: CNN backbone for extracting image features
2. **Transformer Blocks**: Multi-head attention for feature refinement
3. **3D Gaussian Predictor**: Predicts 3D Gaussian parameters
4. **Renderer**: 3D Gaussian Splatting renderer


## License

MIT License
