"""
PyTorch implementation of denoising neural network architectures.

This module provides various CNN architectures for image denoising,
including U-Net variants and DnCNN (Denoising Convolutional Neural Network).

Models included:
- Unet: Standard U-Net architecture
- SimpleUnet: Lightweight U-Net variant
- DnCNN: Denoising CNN (17 layers)
- SmallDnCNN: Smaller DnCNN variant (20 layers)
- LargeDnCNN: Larger DnCNN variant (28 layers)
- LargeDnCNNWithTanh: Large DnCNN with tanh output
- IDCNN: Identity-based denoising CNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union


# ============================================================================
# HELPER MODULES
# ============================================================================

class ConvBlock(nn.Module):
    """
    Basic convolutional block with two Conv-ReLU pairs.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        use_batchnorm: Whether to use batch normalization
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batchnorm: bool = False
    ):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        ])
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownBlock(nn.Module):
    """
    Encoder block: ConvBlock followed by optional MaxPooling.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        use_maxpool: Whether to apply max pooling
        use_batchnorm: Whether to use batch normalization
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_maxpool: bool = True,
        use_batchnorm: bool = False
    ):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, use_batchnorm)
        self.use_maxpool = use_maxpool
        if use_maxpool:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        conv_out = self.conv(x)
        if self.use_maxpool:
            pooled = self.pool(conv_out)
            return pooled, conv_out  # Return both pooled and skip connection
        return conv_out


class UpBlock(nn.Module):
    """
    Decoder block: Upsample, concatenate skip connection, then ConvBlock.
    
    Args:
        in_channels: Number of input channels (from previous layer)
        skip_channels: Number of channels from skip connection
        out_channels: Number of output channels
        use_batchnorm: Whether to use batch normalization
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_batchnorm: bool = False
    ):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels, use_batchnorm)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        
        # Handle size mismatch due to odd dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([x, skip], dim=1)  # Concatenate along channel dimension
        return self.conv(x)


# ============================================================================
# U-NET MODELS
# ============================================================================

class Unet(nn.Module):
    """
    Standard U-Net architecture for image denoising.
    
    Architecture:
        - 4 encoder blocks with max pooling
        - 1 bottleneck block
        - 4 decoder blocks with skip connections
        - Final 1x1 convolution for output
    
    Args:
        input_channels: Number of input image channels (default: 1 for grayscale)
        output_channels: Number of output channels (default: 1)
        features: List of feature sizes for each level
        use_batchnorm: Whether to use batch normalization
        dropout: Dropout probability (currently unused)
    
    Example:
        >>> model = Unet(input_channels=1, output_channels=1)
        >>> x = torch.randn(1, 1, 256, 256)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 1, 256, 256])
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        features: Optional[List[int]] = None,
        use_batchnorm: bool = False,
        dropout: float = 0.9
    ):
        super().__init__()
        
        if features is None:
            #features = [64, 128, 256, 512, 1024]
            features = [16, 32,64, 128, 256]
        
        self.features = features
        
        # Encoder path
        self.down1 = DownBlock(input_channels, features[0], use_maxpool=True, use_batchnorm=use_batchnorm)
        self.down2 = DownBlock(features[0], features[1], use_maxpool=True, use_batchnorm=use_batchnorm)
        self.down3 = DownBlock(features[1], features[2], use_maxpool=True, use_batchnorm=use_batchnorm)
        self.down4 = DownBlock(features[2], features[3], use_maxpool=True, use_batchnorm=use_batchnorm)
        
        # Bottleneck
        self.bottleneck = DownBlock(features[3], features[4], use_maxpool=False, use_batchnorm=use_batchnorm)
        
        # Decoder path
        self.up1 = UpBlock(features[4], features[3], features[3], use_batchnorm=use_batchnorm)
        self.up2 = UpBlock(features[3], features[2], features[2], use_batchnorm=use_batchnorm)
        self.up3 = UpBlock(features[2], features[1], features[1], use_batchnorm=use_batchnorm)
        self.up4 = UpBlock(features[1], features[0], features[0], use_batchnorm=use_batchnorm)
        
        # Output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(features[0], output_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1, skip1 = self.down1(x)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)
        x4, skip4 = self.down4(x3)
        
        # Bottleneck
        x5 = self.bottleneck(x4)
        
        # Decoder
        x = self.up1(x5, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)
        
        # Output
        return self.output_conv(x)

# ============================================================================
# DnCNN MODELS
# ============================================================================

class DnCNN(nn.Module):
    """
    Denoising Convolutional Neural Network (DnCNN).
    
    Standard DnCNN architecture with 17 convolutional layers.
    Uses Conv-ReLU blocks without batch normalization.
    
    Reference:
        Zhang et al., "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN
        for Image Denoising", IEEE TIP, 2017.
    
    Args:
        input_channels: Number of input channels
        output_channels: Number of output channels
        num_layers: Number of convolutional layers (default: 17)
        features: Number of feature channels (default: 64)
        use_batchnorm: Whether to use batch normalization
        activation: Output activation ('tanh', 'relu', 'sigmoid', or None)
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        num_layers: int = 17,
        features: int = 64,
        use_batchnorm: bool = False,
        activation: Optional[str] = None
    ):
        super().__init__()
        
        self.num_layers = num_layers
        
        # Build layers
        layers = []
        
        # First layer: Conv + ReLU
        layers.append(nn.Conv2d(input_channels, features, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        
        # Middle layers: Conv + (BN) + LeakyReLU
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size=3, padding=1, bias=True))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(features))
            layers.append(nn.LeakyReLU(inplace=True))
        
        # Last layer: Conv + ReLU
        layers.append(nn.Conv2d(features, output_channels, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        
        self.dncnn = nn.Sequential(*layers)
        
        # Output activation
        self.activation = activation
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dncnn(x)
        
        if self.activation == 'tanh':
            return torch.tanh(out)
        elif self.activation == 'sigmoid':
            return torch.sigmoid(out)
        elif self.activation == 'relu':
            return F.relu(out)
        else:
            return out


class IDCNN(nn.Module):
    """
    Identity-based Denoising CNN.
    
    Estimates noise and divides input by it for denoising.
    Output = tanh(input / estimated_noise)
    
    Args:
        input_channels: Number of input channels
        output_channels: Number of output channels
        num_layers: Number of convolutional layers (default: 17)
        features: Number of feature channels (default: 64)
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        num_layers: int = 17,
        features: int = 64
    ):
        super().__init__()
        
        layers = []
        
        # First layer
        layers.append(nn.Conv2d(input_channels, features, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        # Middle layers
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.ReLU(inplace=True))
        
        # Last layer
        layers.append(nn.Conv2d(features, output_channels, kernel_size=3, padding=1, bias=False))
        
        self.cnn = nn.Sequential(*layers)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Estimate noise/factor
        noise_estimate = self.cnn(x)
        noise_estimate = F.relu(noise_estimate) + 1e-6  # Ensure positive
        
        # Divide input by estimated noise
        denoised = x / noise_estimate
        
        return torch.tanh(denoised)


# ============================================================================
# FACTORY FUNCTIONS (for compatibility with original API)
# ============================================================================

def unet(input_size: Tuple[int, int, int] = (256, 256, 1), dropout: float = 0.9) -> Unet:
    """
    Factory function for Unet (matches original TensorFlow API).
    
    Args:
        input_size: Tuple of (H, W, C) - note TensorFlow uses channels-last
        dropout: Dropout probability (currently unused)
        
    Returns:
        Unet model instance
    """
    # Convert from TF format (H, W, C) to number of channels
    input_channels = input_size[-1] if len(input_size) == 3 else 1
    return Unet(input_channels=input_channels, output_channels=input_channels)

def dncnn(input_size: Tuple[int, int, int]) -> DnCNN:
    """Factory function for DnCNN (17 layers)."""
    input_channels = input_size[-1] if len(input_size) == 3 else 1
    return DnCNN(input_channels=input_channels, output_channels=input_channels, num_layers=17)


def id_cnn(input_size: Tuple[int, int, int]) -> IDCNN:
    """Factory function for IDCNN."""
    input_channels = input_size[-1] if len(input_size) == 3 else 1
    return IDCNN(input_channels=input_channels, output_channels=input_channels)


# ============================================================================
# MODEL SUMMARY UTILITIES
# ============================================================================

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_summary(model: nn.Module, input_size: Tuple[int, ...] = (1, 1, 256, 256)):
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, H, W)
    """
    print("=" * 70)
    print(f"Model: {model.__class__.__name__}")
    print("=" * 70)
    
    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Non-trainable parameters: {total - trainable:,}")
    
    # Test forward pass
    device = next(model.parameters()).device
    x = torch.randn(*input_size, device=device)
    
    with torch.no_grad():
        y = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print("=" * 70)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing Denoising Models")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Test input
    batch_size = 2
    height, width = 256, 256
    channels = 1
    x = torch.randn(batch_size, channels, height, width, device=device)
    
    # Test each model
    models_to_test = [
        ("Unet", Unet(input_channels=1)),
        ("DnCNN (17 layers)", DnCNN(input_channels=1, num_layers=17)),
        ("SmallDnCNN (20 layers)", SmallDnCNN(input_channels=1, num_layers=20)),
        ("LargeDnCNN (28 layers)", LargeDnCNN(input_channels=1, num_layers=28)),
        ("LargeDnCNNWithTanh (34 layers)", LargeDnCNNWithTanh(input_channels=1, num_layers=34)),
        ("IDCNN", IDCNN(input_channels=1)),
    ]
    
    for name, model in models_to_test:
        model = model.to(device)
        
        # Forward pass
        with torch.no_grad():
            y = model(x)
        
        total, trainable = count_parameters(model)
        
        print(f"{name}:")
        print(f"  Parameters: {total:,}")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {y.shape}")
        print(f"  Output range: [{y.min().item():.4f}, {y.max().item():.4f}]")
        print()
    
    # Test factory functions
    print("\nTesting factory functions:")
    print("-" * 40)
    
    input_size = (256, 256, 1)  # TensorFlow format (H, W, C)
    
    factory_models = [
        ("Unet_factory", unet(input_size)),
        ("simple_unet_model", simple_unet_model(input_size)),
        ("dncnn", dncnn(input_size)),
        ("small_dncnn", small_dncnn(input_size)),
        ("large_dncnn", large_dncnn(input_size)),
    ]
    
    for name, model in factory_models:
        model = model.to(device)
        with torch.no_grad():
            y = model(x)
        print(f"{name}: OK (output shape: {y.shape})")
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)