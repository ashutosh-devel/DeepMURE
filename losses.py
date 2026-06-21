"""
PyTorch implementation of MURE (Monte Carlo Unbiased Risk Estimator) and MSE losses.
Optimized for production and multi-GPU setups.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Any

class MSELoss(nn.Module):
    """Standard MSE Loss wrapper."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse(prediction, target)

class TVLoss(nn.Module):
    """Total Variation Loss to encourage spatial smoothness."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        h_x = x.size(2)
        w_x = x.size(3)

        count_h = (h_x - 1) * w_x
        count_w = h_x * (w_x - 1)

        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()

        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

class DeepMURELoss(nn.Module):
    """
    Monte Carlo Unbiased Risk Estimator (MURE) Loss for SAR image despeckling.
    Estimates true MSE using only the noisy observation via finite-difference 
    approximations of Jacobian and Hessian terms.
    """
    def __init__(
        self,
        model: nn.Module,
        n_looks: int = 1,
        num_perturbations: int = 1,
        eps_factor: float = 1.0,
        img_size: Optional[Tuple[int, ...]] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.model = model
        self.num_perturbations = num_perturbations
        self.eps_factor = eps_factor
        self.n_looks = n_looks
        self.img_size = img_size
        self.device = device if device is not None else torch.device('cpu')
        
        # History tracking arrays
        self.histories: List[List[float]] = []
        self.true_histories: List[List[float]] = []
        
        # Internal states
        self.iteration: Optional[int] = None
        self.total_iterations: Optional[int] = None
        self.eps = 1e-2
        
        # Operational flags
        self.debugmode = False
        self.fix_perts = True
        self.use_hessian = True
        self.square_mode = False
        
        # Fixed noise buffers to avoid excessive device allocations
        self._cached_B_J: Optional[torch.Tensor] = None
        self._cached_B_H: Optional[torch.Tensor] = None
    
    def to(self, *args, **kwargs):
        """Override to keep internal device variables up to date."""
        device = torch.device(args[0]) if args else kwargs.get('device', self.device)
        if device is not None:
            self.device = device
            if self._cached_B_J is not None:
                self._cached_B_J = self._cached_B_J.to(device)
            if self._cached_B_H is not None:
                self._cached_B_H = self._cached_B_H.to(device)
        return super().to(*args, **kwargs)

    def _get_normal_noise(self, mean: float = 0.0, std_dev: float = 1.0) -> torch.Tensor:
        return torch.randn(self.img_size, device=self.device) * std_dev + mean
    
    def _get_triangular_noise(self, a: float = -2.0, m: float = 1.0, b: float = 1.0) -> torch.Tensor:
        noise = np.random.triangular(left=a, mode=m, right=b, size=self.img_size)
        return torch.from_numpy(noise).float().to(self.device)
    
    def perturb_input_image(
        self,
        noisy: torch.Tensor,
        noise: torch.Tensor,
        epsilon: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        YB = noisy * noise
        positive = (noisy + epsilon * YB).unsqueeze(0)
        negative = (noisy - epsilon * YB).unsqueeze(0)
        return positive, negative, YB
    
    def _estimate_HF_Y3(self, Y: torch.Tensor, FY: torch.Tensor) -> torch.Tensor:
        Y = Y.detach()
        epsilons = np.linspace(7e-2, 9e-2, self.num_perturbations)
        avg_estimate_HF = torch.tensor(0.0, device=self.device)
        noise_std = 1.0
        
        if self.fix_perts:
            if self._cached_B_H is None or self._cached_B_H.shape != self.img_size:
                self._cached_B_H = self._get_triangular_noise() * noise_std
            B_H = self._cached_B_H
        
        avg_third_moment = 0.0
        
        for p in range(self.num_perturbations):
            eps = epsilons[p]
            if not self.fix_perts:
                B_H = self._get_triangular_noise() * noise_std
            
            B_H = B_H - B_H.mean()
            third_moment = (B_H ** 3.0).mean()
            avg_third_moment += third_moment.item()
            
            H_pert_pos, H_pert_neg, YBH = self.perturb_input_image(Y, B_H, eps)
            # with torch.no_grad():
            f_H_pert_pos = self.model(H_pert_pos)
            f_H_pert_neg = self.model(H_pert_neg)
        
            Hf = (f_H_pert_pos + f_H_pert_neg - 2 * FY) / (eps ** 2)
            avg_estimate_HF = avg_estimate_HF + (YBH * Hf).sum()
            
            if self.debugmode:
                self._save_debug_tensors({
                    'YBH': YBH, 'Hf': Hf, 'H_positive': H_pert_pos, 'H_negative': H_pert_neg,
                    'fH_positive': f_H_pert_pos, 'fH_negative': f_H_pert_neg
                }, prefix='H_')
        
        avg_third_moment /= self.num_perturbations
        avg_estimate_HF = avg_estimate_HF / (self.num_perturbations * (noise_std ** 2) * abs(avg_third_moment))
        return avg_estimate_HF
    
    def _estimate_JF_Y2(self, Y: torch.Tensor, FY: torch.Tensor) -> torch.Tensor:
        Y = Y.detach()
        epsilons = [(1.0 / np.sqrt(self.n_looks)) / (Y.norm(p=2) + 1e-12)] * self.num_perturbations

        avg_estimate_JF = torch.tensor(0.0, device=self.device)
        noise_std = 2.0
        
        if self.fix_perts:
            if self._cached_B_J is None or self._cached_B_J.shape != self.img_size:
                self._cached_B_J = self._get_normal_noise() * noise_std
            B = self._cached_B_J
        
        for p in range(self.num_perturbations):
            eps = epsilons[p]
            if not self.fix_perts:
                B = self._get_normal_noise() * noise_std
            
            B = B - B.mean()
            self.eps = eps
            
            pert_pos, pert_neg, YB = self.perturb_input_image(Y, B, eps)
            # with torch.no_grad():
            f_pert_pos = self.model(pert_pos)
            f_pert_neg = self.model(pert_neg)
            
            Jf = (f_pert_pos - f_pert_neg) / (2 * eps)
            avg_estimate_JF = avg_estimate_JF + (YB * Jf).sum()
            
            if self.debugmode:
                self._save_debug_tensors({
                    'Y': Y, 'YB': YB, 'Jf': Jf, 'positive': pert_pos, 'negative': pert_neg,
                    'f_positive': f_pert_pos, 'f_negative': f_pert_neg
                }, prefix='J_')
        
        avg_estimate_JF = avg_estimate_JF / (self.num_perturbations * (noise_std ** 2))
        return avg_estimate_JF
    
    def _save_debug_tensors(self, tensors: Dict[str, torch.Tensor], prefix: str = '') -> None:
        os.makedirs('./perturbation_outputs', exist_ok=True)
        for name, tensor in tensors.items():
            filepath = f"./perturbation_outputs/{self.iteration}_{prefix}{name}.raw"
            tensor.detach().cpu().numpy().squeeze().astype(np.float32).tofile(filepath)
    
    def estimate_check(
        self,
        Ys: torch.Tensor,
        FYs: torch.Tensor,
        Xs: torch.Tensor,
        factors: Any = True,
        verbose: bool = True
    ) -> torch.Tensor:
        if Ys.dim() == 4:
            img_size = Ys.shape[1:]
            n_images = Ys.shape[0]
            num_pixels = img_size[1] * img_size[2]
        else:
            img_size = Ys.shape
            n_images = 1
            num_pixels = img_size[0] * img_size[1]
        
        k = self.n_looks
        avg_loss = torch.tensor(0.0, device=self.device)
        
        for n in range(n_images):
            if Ys.dim() == 4:
                X, Y, FY = Xs[n, ...], Ys[n, ...], FYs[n, ...]
            else:
                X, Y, FY = Xs, Ys, FYs
            
            self.img_size = Y.shape
            
            Y_sq = (k / (k + 1)) * (Y.norm(2) ** 2) / num_pixels
            X_sq = (X.norm(2) ** 2) / num_pixels
            FY_sq = (FY.norm(2) ** 2) / num_pixels
            
            FY_Y = -2.0 * (Y * FY).sum() / num_pixels
            FY_X = -2.0 * (X * FY).sum() / num_pixels
            
            if self.debugmode:
                self._save_debug_tensors({'X': X}, prefix='')
            
            JF_Y2 = self._estimate_JF_Y2(Y, FY)
            JF_Y2 = (2 / (k + 1)) * JF_Y2 / num_pixels
            
            if self.use_hessian:
                HF_Y3 = self._estimate_HF_Y3(Y, FY)
                HF_Y3 = (2.0 / ((k + 2) * (k + 1))) * HF_Y3 / num_pixels
            else:
                HF_Y3 = torch.tensor(0.0, device=self.device)
            
            surr_loss = Y_sq + FY_sq + FY_Y + JF_Y2 + HF_Y3
            avg_loss = avg_loss + surr_loss
            true_loss = X_sq + FY_sq + FY_X
            
            gap_f = FY_X - FY_Y
            gap_x = X_sq - Y_sq
            corr = JF_Y2 + HF_Y3
            
            to_f = lambda x: x.item() if isinstance(x, torch.Tensor) else float(x)
            
            record = [
                to_f(surr_loss), to_f(Y_sq), to_f(FY_sq), to_f(FY_Y), to_f(JF_Y2), to_f(HF_Y3),
                to_f(gap_f), to_f(corr), to_f(gap_x), to_f(JF_Y2), to_f(HF_Y3),
                to_f(Y.max()), to_f(Y.min()), to_f(FY.max()), to_f(FY.min()), to_f(true_loss - surr_loss)
            ]
            
            self.histories.append(record)
            self.true_histories.append([to_f(true_loss), to_f(X_sq), to_f(FY_sq), to_f(FY_X)])
        
        avg_loss = avg_loss / n_images
        return avg_loss ** 2 if self.square_mode else torch.sqrt(avg_loss ** 2 + 1e-7)
    
    def forward(
        self,
        Ys: torch.Tensor,
        FYs: torch.Tensor,
        Xs: Optional[torch.Tensor] = None,
        factors: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if Xs is None:
            Xs = torch.zeros_like(Ys)
        return self.estimate_check(Ys, FYs, Xs, True, verbose=False)
    
    def reset_histories(self) -> None:
        self.histories = []
        self.true_histories = []