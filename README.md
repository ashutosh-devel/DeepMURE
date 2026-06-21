# SpeckleFormer: Hierarchical Transformer-based Speckle Denoising for SAR Images

SpeckleFormer is a hierarchical U-Net architecture designed for the removal of speckle noise from Synthetic Aperture Radar (SAR) images. By integrating multi-scale spatial feature extraction with frequency-domain gating, SpeckleFormer effectively preserves structural details while suppressing multiplicative speckle noise.

## 🚀 Features

- **Multi-Scale Spatial Learning**: Utilizes Spatial Token Blocks (STB) with varying kernel sizes and dilated convolutions to capture features across different scales.
- **Frequency-Domain Gating**: Employs Frequency Token Blocks (FTB) that use Fast Fourier Transforms (FFT) and smooth gain estimation to filter noise in the spectral domain.
- **Adaptive Feature Fusion**: A Channel Attention Block (CAB) using a Q-K-V formulation dynamically fuses spatial and frequency representations.
- **Residual Refinement**: Incorporates a Residual Feed-Forward Network (RFFN) and a specialized shallow branch (TPM) to ensure high-fidelity reconstruction.
- **Model Scalability**: Provides three variants to balance performance and computational cost:
  - `SpeckleFormer_Small` (~1.0M params)
  - `SpeckleFormer_Medium` (~7.3M params) - *Recommended*
  - `SpeckleFormer_Large` (~12.0M params)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch
- torchvision
- NumPy
- Matplotlib
- SciPy
- Pillow

### Setup
```bash
git clone https://github.com/your-username/SpeckleFormer.git
cd SpeckleFormer
pip install -r requirements.txt # If provided, otherwise install prerequisites manually
```

## 📖 Usage

### Training
To train the model, use `train.py`. You can configure the dataset paths, number of looks, and model hyperparameters within the script.

```bash
python train.py
```

### Evaluation (Inference)
To denoise images using a pre-trained checkpoint, use `evaluate.py`.

```bash
python evaluate.py --checkpoint path/to/checkpoint.pth --input_dir path/to/noisy_images --output_dir results
```

**Arguments:**
- `--checkpoint`: Path to the `.pth` model weights.
- `--input_dir`: Folder containing corrupted `.png` or `.tif` images.
- `--output_dir`: Folder where denoised images will be saved (default: `results`).
- `--device`: Device to run inference on (`cuda` or `cpu`).

## 🏗️ Architecture Overview

The SpeckleFormer architecture follows a hierarchical U-Net structure:
1. **Encoder**: Progressively downsamples the image through stages of `SFDB` (SpeckleFormer Dual Blocks) and `Downsample` layers.
2. **Bottleneck**: High-level latent representation processed by multiple `SFDB` layers.
3. **Decoder**: Upsamples features using `Upsample` layers and fuses them with encoder skip-connections via `CAB` and `RFFN`.
4. **Reconstruction**: Final output is a combination of the decoder's reconstruction and a shallow-path projection (TPM).

## 📊 Results
(Add your results table, PSNR/SSIM metrics, and qualitative comparisons here)

## 📜 Citation
If you find this work useful in your research, please cite:
```bibtex
@article{speckleformer2026,
  title={SpeckleFormer: Hierarchical Transformer-based Speckle Denoising for SAR Images},
  author={Your Name and Co-authors},
  journal={Your Journal/Conference},
  year={2026}
}
```
