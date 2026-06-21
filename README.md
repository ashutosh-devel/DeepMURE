# DeepMURE & SpeckleFormer: Self-Supervised SAR Despeckling

This repository contains the official PyTorch implementation for the paper: **"DeepMURE: Deep Self-Supervised Despeckling via Monte Carlo Multiplicative Unbiased Risk Estimation"** (IEEE Transactions on Image Processing, 2026).

It includes both **DeepMURE**, a mathematically grounded self-supervised learning framework that trains denoisers without ground-truth images, and **SpeckleFormer**, a specialized hierarchical spatial-frequency transformer architecture designed specifically for Synthetic Aperture Radar (SAR) imagery.

## 🚀 Features

### 1. DeepMURE Training Framework
- **Ground-Truth Free Training**: Trains directly on noisy SAR intensity images without requiring clean references, multi-temporal stacks, or complex SLC data.
- **Monte Carlo Risk Estimation**: Utilizes Hutchinson-type stochastic trace estimators (Jacobian and Hessian corrections) to approximate the Oracle Mean Squared Error (MSE) stably and efficiently.
- **Statistical Regularization**: Incorporates Total Variation (TV) and a dynamically estimated Gamma KL-divergence loss to ensure the extracted speckle residuals strictly adhere to the theoretical Gamma distribution of fully developed speckle.

### 2. SpeckleFormer Architecture
- **Multi-Scale Spatial Learning**: Utilizes Spatial Token Blocks (STB) featuring standard 3×3, 5×5, and dilated convolutions to capture both local speckle variations and broader structural textures.
- **Frequency-Domain Gating**: Employs Frequency Token Blocks (FTB) that decompose features in the Fourier domain into low, mid, and high-frequency bands, applying a learnable gating mechanism to suppress wideband noise while preserving structure.
- **Adaptive Feature Fusion**: A Channel Attention Block (CAB) uses a Q-K-V formulation (via 3×3 depth-wise convolutions) to dynamically fuse spatial and frequency representations along the channel dimension.
- **Target Preservation Module (TPM)**: A specialized, learnable shallow-path dynamic thresholding mechanism operating in parallel to the main backbone. It prevents the over-smoothing of high-intensity deterministic point targets (e.g., urban buildings).
- **Model Scalability**: Provides three variants to balance performance and computational cost:
  - `SpeckleFormer_Small` (~1.0M params)
  - `SpeckleFormer_Medium` (~7.31M params) - *Default configuration evaluated in the paper*
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
📖 Usage
Training with DeepMURE (Demo script)
To train the model in a self-supervised manner, use estimation_demo.py. You can configure the dataset paths, the equivalent number of looks (L), and model hyperparameters within the script.
# Basic run
python estimation_demo.py

# Run with DeepMURE mode for single-look (L=1) Sentinel-1 Hybrid data:
python estimation_demo.py --mode deepmure --img_h 256 --img_w 256 --arch speckleformer_medium --looks 1 --batch_size 2 --val_batch_size 2
Evaluation (Inference)
To denoise images using a pre-trained checkpoint, use evaluate.py. The inference time for a 256×256 patch using SpeckleFormer_Medium is approximately ~35 ms on an NVIDIA A100 GPU.
python evaluate.py --checkpoint path/to/checkpoint.pth --input_dir path/to/noisy_images --output_dir results
Arguments:
--checkpoint: Path to the .pth model weights.
--input_dir: Folder containing corrupted .png or .tif images.
--output_dir: Folder where denoised images will be saved (default: results).
--device: Device to run inference on (cuda or cpu).
🏗️ Architecture Overview
The SpeckleFormer architecture follows a hierarchical U-Net structure featuring the novel Spatial-Frequency Despeckling Block (SFDB):
Shallow Feature Extraction: Initial 3×3 convolutions extract foundational features (F 
shallow
​
 ).
Encoder: Progressively downsamples the image through stages of SFDB blocks, reducing spatial resolution to capture wide-context latent representations.
Decoder: Upsamples features and concatenates them with encoder skip-connections, refining them via CAB (channel attention) and RFFN (gated residual feed-forward networks).
Target Preservation & Reconstruction: The final output is formed by combining the decoder's volumetric reconstruction with the radiometrically protected features from the parallel TPM branch.
📊 Results Summary
Our DeepMURE + SpeckleFormer pipeline sets a new state-of-the-art for self-supervised SAR despeckling, successfully bridging the gap to fully-supervised Oracle performance:
Sentinel-1 Hybrid Dataset (L=1): Achieves 27.09 dB PSNR and 0.7656 SSIM, outperforming competing self-supervised baselines like Speckle2Void and MERLIN by over 1.6 dB.
Sentinel-1 Real Dataset (GRD): Yields a Target-to-Clutter Ratio (TCR) of 9.8 and an Edge Preservation Degree (EPD-ROA) of 0.89, demonstrating superior geometric fidelity.
ICEYE SLC Dataset: Provides sub-meter high-resolution noise suppression with an Equivalent Number of Looks (ENL) of 11.7 while strictly maintaining a low Gamma KL-Divergence of 0.17.
Downstream Applications: Demonstrates an overall +4.00% Mean IoU improvement in SAR semantic segmentation tasks (e.g., FUSAR-Map) after preprocessing raw inputs with our pipeline.
📜 Citation
If you find this code or our DeepMURE framework useful in your research, please cite our paper:
@article{gupta2026deepmure,
  title={DeepMURE: Deep Self-Supervised Despeckling via Monte Carlo Multiplicative Unbiased Risk Estimation},
  author={Gupta, Ashutosh and Seelamantula, Chandra Sekhar and Blu, Thierry and Yelgoe, Tanish and Dube, Nitant and Raman, Shanmuganathan},
  journal={IEEE Transactions on Image Processing},
  year={2026}
}