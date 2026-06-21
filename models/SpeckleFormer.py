import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import matplotlib.pyplot as plt

# ==========================================
# IMPLEMENT MAIN BLOCKS IN THE PAPER
# ==========================================

class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm for 2D spatial inputs."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class STB(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = LayerNorm2d(dim)
        self.conv1x1_in = nn.Conv2d(dim, dim, 1)
        
        self.conv3x3 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv5x5 = nn.Conv2d(dim, dim, 5, padding=2)
        self.conv3x3_dil = nn.Conv2d(dim, dim, 3, padding=2, dilation=2)
        self.conv1x1_out = nn.Conv2d(dim * 3, dim, 1)

    def forward(self, x):
        x_norm = self.conv1x1_in(self.ln(x))
        p1 = self.conv3x3(x_norm)
        p2 = self.conv5x5(x_norm)
        p3 = self.conv3x3_dil(x_norm)
        out = torch.cat([p1, p2, p3], dim=1)
        return F.gelu(self.conv1x1_out(out))

#compared to paper, we use more bands for smoother interpolation of gating functions here
class FTB(nn.Module):
    """
    Frequency Token Block - Smooth Sheet Version.
    Uses low-resolution gain estimation + bilinear interpolation to ensure a smooth frequency response and zero ringing.
    """
    def __init__(self, dim, gain_res=(8, 8)):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.gain_res = gain_res # The "resolution" of the smooth sheet
        
        # This network operates on the low-res "thumbnail" of the spectrum
        self.gain_net = nn.Sequential(
            nn.Conv2d(dim, dim, 1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, dim, 1),
            nn.Tanh() #leaky + tanh option reduces ringing around residual high intensity targets
        )
        
        #was used earlier
        # self.gain_net = nn.Sequential(
        #     nn.Conv2d(dim, dim, 1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(dim, dim, 1),
        #     nn.Sigmoid() #leaky + tanh option reduces ringing around residual high intensity targets
        # )
        
        self.conv_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_proj = self.conv1(x)
        X_fft = torch.fft.rfft2(x_proj, norm='ortho')
        # Magnitude shape: (B, C, H, W//2 + 1)
        mag = torch.abs(X_fft)
        mag_low = F.interpolate(mag, size=self.gain_res, mode='bilinear', align_corners=False)
        gain_low = self.gain_net(mag_low)
        gain_smooth = F.interpolate(
            gain_low, 
            size=(H, X_fft.shape[-1]), 
            mode='bilinear', 
            align_corners=False
        )
        final_gain = 1.0 + 0.2 * gain_smooth #additional step:limiting the maximum gain from the gating.
        Z_mod = X_fft * final_gain
        x_recon = torch.fft.irfft2(Z_mod, s=(H, W), norm='ortho')
        
        return self.conv_out(x_recon)

class CAB(nn.Module):
    """Channel Attention Block (CAB) with a lightweight Q‑K‑V formulation."""

    def __init__(self, dim: int, reduction: int = 4):
        """
        Args:
            dim: number of input channels (after the spatial‑freq concat).
            reduction: reduction factor for the hidden dimension in the
                       optional squeeze‑excitation part (default 4).
        """
        super().__init__()

        # Fuse the two branches (spatial + frequency)
        self.fuse = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)

        # Q, K, V projections – 1×1 conv keeps spatial size unchanged
        self.q_conv = nn.Conv2d(dim, dim, kernel_size=3, bias=False,padding=1)  #C1  1->3
        self.k_conv = nn.Conv2d(dim, dim, kernel_size=3, bias=False,padding=1)
        self.v_conv = nn.Conv2d(dim, dim, kernel_size=3, bias=False,padding=1)

        # Learnable scaling factor (as in many transformer blocks)
        self.gamma = nn.Parameter(torch.zeros(1))

        # Final projection back to `dim` channels
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x_spatial: torch.Tensor, x_freq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_spatial: (B, C, H, W) – spatial branch output
            x_freq:    (B, C, H, W) – frequency‑branch output
        Returns:
            Tensor of shape (B, C, H, W)
        """
        # ------------------------------------------------------------------
        # 1️⃣ Fuse the two branches
        # ------------------------------------------------------------------
        x = self.fuse(torch.cat([x_spatial, x_freq], dim=1))   # (B, C, H, W)

        # ------------------------------------------------------------------
        # 2️⃣ Q, K, V projections
        # ------------------------------------------------------------------
        q = self.q_conv(x)   # (B, C, H, W)
        k = self.k_conv(x)   # (B, C, H, W)
        v = self.v_conv(x)   # (B, C, H, W)

        B, C, H, W = q.shape
        N = H * W

        # ------------------------------------------------------------------
        # 3️⃣ Channel‑wise similarity (dot‑product)
        # ------------------------------------------------------------------
        q_flat = q.view(B, C, N)          # (B, C, N)
        k_flat = k.view(B, C, N)          # (B, C, N)

        # (B, C) – one similarity score per channel per batch element
        similarity = (q_flat * k_flat).sum(dim=2)   # or torch.einsum('bcn,bcn->bc', ...)

        # Normalise to [0,1] (sigmoid works well; you could also use softmax)
        attn = torch.sigmoid(similarity).view(B, C, 1, 1)   # (B, C, 1, 1)

        # ------------------------------------------------------------------
        # 4️⃣ Weight V, add residual, scale, and project
        # ------------------------------------------------------------------
        weighted_v = v * attn                       # (B, C, H, W)
        out = self.gamma * weighted_v + x           # residual + learnable scale
        out = self.proj(out)                        # final 1×1 conv
        return out

class RFFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = LayerNorm2d(dim)
        self.expand = nn.Conv2d(dim, dim * 2, 1)
        self.conv3x3 = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        x_r = self.expand(self.ln(x))
        x_linear, x_gated = x_r.chunk(2, dim=1)
        x_refined = x_linear * torch.sigmoid(x_gated)
        return self.conv3x3(x_refined) + x

class SFDB(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = LayerNorm2d(dim)
        self.stb = STB(dim)
        self.ftb = FTB(dim)
        self.cab = CAB(dim)
        self.rffn = RFFN(dim)

    def forward(self, x):
        x_norm = self.ln(x)
        x_spatial = self.stb(x_norm)
        x_freq = self.ftb(x_norm)
        x_fused = self.cab(x_spatial, x_freq)
        return self.rffn(x_fused)  + x

class TPM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 1)
        self.conv2 = nn.Conv2d(3, 1, 1)
        #self.alpha = nn.Parameter(torch.zeros(0.9)) #enable bias instead!
        self.alpha = 0.0 #enable bias instead!

    def forward(self, f_shallow):
        x = F.relu(self.conv1(f_shallow))
        x = self.conv2(x)
        return F.relu(x) #+ f_shallow

# --- Standard U-Net Channel Scaling Helpers ---
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Standard downsampling doubles the channels
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.down(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            # 1. Upscale spatially using bilinear interpolation (No artifacts)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            # 2. Use a standard Conv to refine the result and change channels
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return self.up(x)

# ==========================================
# SPECKLEFORMER MAIN DEFINITION
# ==========================================
class SpeckleFormer(nn.Module):
    """Hierarchical U-Net SpeckleFormer Architecture (Standard Channel Scaling)."""
    def __init__(self, dim=16):
        super().__init__()
        # Initial features
        self.shallow = nn.Conv2d(1, 3, 3, padding=1)
        self.embed = nn.Conv2d(3, dim, 3, padding=1)
        
        # Encoder Level 1 (dim)
        self.enc1 = nn.Sequential(*[SFDB(dim) for _ in range(4)])
        self.down1 = Downsample(dim, dim * 2)
        
        # Encoder Level 2 (dim * 2)
        self.enc2 = nn.Sequential(*[SFDB(dim * 2) for _ in range(4)])
        self.down2 = Downsample(dim * 2, dim * 4)
        
        # Latent/Bottleneck Level 3 (dim * 4)
        self.latent = nn.Sequential(*[SFDB(dim * 4) for _ in range(6)])
        
        # Decoder Level 2 (dim * 2)
        self.up1 = Upsample(dim * 4, dim * 2)
        # Concat channels: upsampled (dim * 2) + skip (dim * 2) = dim * 4
        self.reduce1 = nn.Conv2d(dim * 4, dim * 2, 1)
        self.dec1 = nn.Sequential(*[SFDB(dim * 2) for _ in range(4)])
        
        # Decoder Level 1 (dim)
        self.up2 = Upsample(dim * 2, dim)
        # Concat channels: upsampled (dim) + skip (dim) = dim * 2
        self.reduce2 = nn.Conv2d(dim * 2, dim, 1)
        self.dec2 = nn.Sequential(*[SFDB(dim) for _ in range(4)])
        
        # Finalization
        self.unembed = nn.Conv2d(dim, 1, 3, padding=1)
        self.tpm = TPM()

    def forward(self, y):
        # Feature Extraction
        f_shallow = self.shallow(y)
        x0 = self.embed(f_shallow)
        
        # Encoder
        x1 = self.enc1(x0)            # Level 1 features
        x1_down = self.down1(x1)
        
        x2 = self.enc2(x1_down)       # Level 2 features
        x2_down = self.down2(x2)
        
        # Bottleneck
        x_latent = self.latent(x2_down) #this is the bottleneck layer
        
        # Decoder
        x_up1 = self.up1(x_latent)
        x_cat1 = torch.cat([x_up1, x2], dim=1) # Skip connection
        x_dec1 = self.dec1(self.reduce1(x_cat1))
        
        x_up2 = self.up2(x_dec1)
        x_cat2 = torch.cat([x_up2, x1], dim=1) # Skip connection
        x_dec2 = self.dec2(self.reduce2(x_cat2))
        
        # Reconstruction
        recon = F.gelu(self.unembed(x_dec2))
        tpm_out = self.tpm(f_shallow)
        
        return recon + tpm_out #y - (recon + tpm_out)

# ==========================================
# CODE THE PARAMETER COUNTER & VARIANTS
# ==========================================
# Chaannels now double at each depth level (C -> 2C -> 4C),


def SpeckleFormer_Small():
    # Targets ~1.0M parameters
    return SpeckleFormer(dim=11)

def SpeckleFormer_Medium():
    # Targets ~7.31M parameters
    return SpeckleFormer(dim=24) #this is the variant we used in the paper

def SpeckleFormer_Large():
    # Targets ~12.0M parameters
    return SpeckleFormer(dim=38)

if __name__ == "__main__":
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    model_small = SpeckleFormer_Small()
    model_med = SpeckleFormer_Medium()
    model_large = SpeckleFormer_Large()
    
    print(f"SpeckleFormer Small  : {count_parameters(model_small):.2f} M params")
    print(f"SpeckleFormer Medium : {count_parameters(model_med):.2f} M params")
    print(f"SpeckleFormer Large  : {count_parameters(model_large):.2f} M params")