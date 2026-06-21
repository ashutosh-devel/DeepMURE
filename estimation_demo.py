#!/usr/bin/env python3
"""
Production training pipeline for DeepMURE and Oracle-guided SAR Despeckling.
Supports Multi-GPU deployment, automated plotting, and flexible command line overrides.
"""

import os
import argparse
import glob
from datetime import datetime
from typing import List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import ExponentialLR

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.io import savemat

# Core framework modules
from datasets import SynthMultiLookDataset, SpeckleRealCachedDataset
from losses import DeepMURELoss
from models.models import unet, dncnn
from models.SpeckleFormer import SpeckleFormer_Medium
from utils import mse_to_psnr

# ============================================================================
# PLOTTING STYLING CONFIGURATION
# ============================================================================
mpl.use('Agg')
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams.update({
    'figure.dpi': 150, 'figure.facecolor': 'white', 'figure.edgecolor': 'white',
    'font.size': 11, 'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 13, 'axes.titlesize': 14, 'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.fontsize': 10,
    'legend.frameon': True, 'legend.framealpha': 0.9, 'legend.edgecolor': '0.8',
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'lines.linewidth': 1.8,
    'grid.alpha': 0.3, 'grid.linestyle': '--', 'axes.grid': True, 'savefig.bbox': 'tight'
})

COLORS = {
    'true_train': '#2E86AB', 'surrogate_train': '#E94F37', 'true_val': '#1B998B',
    'surrogate_val': '#A23B72', 'noisy': '#424242', 'gradient': '#FF6B35', 'clipped_grad': '#004E89'
}

# ============================================================================
# HELPER PLOTTING ROUTINES
# ============================================================================
def create_professional_loss_plot(train_xaxis, true_loss_hist, surrogate_loss_hist,
                                   xaxis, true_val_loss_hist, surrogate_val_loss_hist,
                                   noisy_loss_hist, num_looks, epch, total_itrs,
                                   save_path, ylabel='MSE Loss', use_psnr=False):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    tl, sl = np.array(true_loss_hist), np.array(surrogate_loss_hist)
    tv, sv, ny = np.array(true_val_loss_hist), np.array(surrogate_val_loss_hist), np.array(noisy_loss_hist)
    
    if use_psnr:
        tl, sl, tv, sv, ny = map(mse_to_psnr, [tl, sl, tv, sv, ny])
        
    ax.plot(train_xaxis, tl, color=COLORS['true_train'], label='True MSE (Train)', alpha=0.85)
    ax.plot(train_xaxis, sl, color=COLORS['surrogate_train'], label='DeepMURE Est. (Train)', alpha=0.85)
    
    mark_every = max(1, len(xaxis) // 15)
    ax.plot(xaxis, tv, color=COLORS['true_val'], label='True (Val)', marker='o', markersize=3.5, markevery=mark_every)
    ax.plot(xaxis, sv, color=COLORS['surrogate_val'], label='DeepMURE Est. (Val)', marker='s', markersize=3.5, markevery=mark_every)
    ax.plot(xaxis, ny, color=COLORS['noisy'], label='Noisy Base', linestyle='--', alpha=0.65)
    
    ax.set_xlabel('Global Training Iterations')
    ax.set_ylabel(ylabel)
    ax.set_title(f'Pipeline Performance Tracking | Looks: {num_looks} | Epoch: {epch+1}')
    ax.legend(loc='best', ncol=2, frameon=True, shadow=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def create_gradient_norm_plot(gradient_norm_hist, clipped_grad_norm_hist, save_path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(gradient_norm_hist, color=COLORS['gradient'], label='Raw Grad Norm', alpha=0.8)
    ax.plot(clipped_grad_norm_hist, color=COLORS['clipped_grad'], label='Applied (Clipped) Norm', alpha=0.8, linestyle='--')
    ax.set_xlabel('Global Training Iterations')
    ax.set_ylabel('L2 Gradient Norm Magnitude')
    ax.set_title('Backpropagation Gradient Dynamics')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def create_stats_plot(term_histories, true_term_histories, save_path, plot_type='loss'):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.cm.Set2(np.linspace(0, 1, 8))
    
    if plot_type == 'loss':
        ax.plot(true_term_histories[:, 0], color=cmap[0], label='Oracle Reference MSE')
        ax.plot(term_histories[:, 0], color=cmap[1], label='Surrogate DeepMURE Output')
        ax.plot(term_histories[:, 1] + term_histories[:, 2] + term_histories[:, 3], color=cmap[2], linestyle=':', label=r'$Y^2 + F(Y)^2 - 2Y \cdot F(Y)$')
        ax.plot(term_histories[:, 4], color=cmap[3], label=r'Jacobian Component ($J_F \cdot Y^2$)')
        ax.plot(term_histories[:, 5], color=cmap[4], label=r'Hessian Component ($H_F \cdot Y^3$)')
    elif plot_type == 'inout':
        ax.plot(term_histories[:, -5], color=cmap[0], label=r'Noisy Max ($Y_{max}$)')
        ax.plot(term_histories[:, -4], color=cmap[1], label=r'Noisy Min ($Y_{min}$)')
        ax.plot(term_histories[:, -3], color=cmap[2], label=r'Denoised Max ($F(Y)_{max}$)')
        ax.plot(term_histories[:, -2], color=cmap[3], label=r'Denoised Min ($F(Y)_{min}$)')
    elif plot_type == 'debug':
        labels = [r'Cross Gap ($gap_f$)', 'Total Corr', r'Norm Gap ($gap_x$)', r'$J_{fy2}$', r'$H_{fy3}$', 'Residual Bias']
        indices = [-10, -9, -8, -7, -6, -1]
        for i, (idx, lbl) in enumerate(zip(indices, labels)):
            ax.plot(term_histories[:, idx], color=cmap[i % 8], label=lbl, alpha=0.8)
            
    ax.set_xlabel('Evaluated Samples')
    ax.set_title(f'Mathematical Diagnostic Breakdown: {plot_type.upper()}')
    ax.legend(loc='best', ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def create_image_grid(img_list, num_rows, num_cols, save_path):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3.5 * num_cols, 3.5 * num_rows))
    axes = np.array(axes).reshape(num_rows, num_cols)
    row_labels = ['Noisy Input (Y)', 'Denoised Target F(Y)', 'Ground Truth (X)']
    
    for idx, img in enumerate(img_list):
        row, col = idx // num_cols, idx % num_cols
        if row >= num_rows or col >= num_cols: break
        img_np = np.squeeze(img.cpu().numpy() if torch.is_tensor(img) else img)
        im = axes[row, col].imshow(img_np, cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off')
        if col == 0 and row < len(row_labels):
            axes[row, col].set_title(row_labels[row], fontsize=11, fontweight='bold', loc='left')
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# ============================================================================
# MASTER TRAINING ROUTINE
# ============================================================================
def run_training_pipeline(args):
    # Setup distributed framework if flagged
    is_distributed = args.world_size > 1 or args.multigpu
    local_rank = 0
    
    if is_distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        if torch.cuda.is_available() and len(args.gpu_ids) > 0:
            device = torch.device(f"cuda:{args.gpu_ids[0]}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

    is_master = not is_distributed or (local_rank == 0)

    if is_master:
        print(f"\nInitializing Execution Context. System Device: {device}")
        os.makedirs(f'./{args.arch}/weights', exist_ok=True)
        os.makedirs(f'./{args.arch}/figures', exist_ok=True)

    # Instantiate Target Model
    if args.model_type == "speckleformer":
        model = SpeckleFormer_Medium()
    elif args.model_type == "dncnn":
        model = dncnn([args.img_h, args.img_w, 1])
    else:
        model = unet([args.img_h, args.img_w, 1])
        
    model = model.to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    elif len(args.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=args.gpu_ids)

    # Initialize Dataset structures
    train_dataset = SynthMultiLookDataset(
        data_folder=args.train_path, crop_size=(args.img_h, args.img_w), gamma_shape=args.looks
    )
    val_dataset = SynthMultiLookDataset(
        data_folder=args.val_path, crop_size=(500, 500), gamma_shape=args.looks, isval=True
    )

    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False)

    # Set up objective frameworks
    true_mse = nn.MSELoss()
    surrogate_mse = DeepMURELoss(model.module if (is_distributed or isinstance(model, nn.DataParallel)) else model, 
                                 n_looks=args.looks, num_perturbations=1, img_size=(args.img_h, args.img_w), device=device)
    surrogate_mse.fix_perts = True
    surrogate_mse.use_hessian = not args.no_hessian
    surrogate_mse.square_mode = False
    
    val_surrogate_mse = DeepMURELoss(model.module if (is_distributed or isinstance(model, nn.DataParallel)) else model, 
                                     n_looks=args.looks, num_perturbations=1, img_size=(500, 500), device=device)
    val_surrogate_mse.fix_perts = True
    val_surrogate_mse.use_hessian = not args.no_hessian

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=args.lr_decay)

    # Storage Arrays for Progress Curves
    surrogate_loss_hist, true_loss_hist = [], []
    true_val_loss_hist, surrogate_val_loss_hist, noisy_loss_hist = [], [], []
    gradient_norm_hist, clipped_grad_norm_hist = [], []
    
    best_loss = float('inf')
    total_itrs = 0
    save_prefix = f"{args.arch}"
    
    if is_master:
        log_file = open(f"./{args.arch}/weights/{save_prefix}_{args.mode}_{args.model_type}_{args.looks}.txt", 'w')

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
            
        for Y, X, _ in train_loader:
            model.train()
            Y, X = Y.to(device), X.to(device)
            optimizer.zero_grad()
            
            surrogate_mse.iteration = total_itrs + 1
            val_surrogate_mse.iteration = total_itrs + 1
            
            pred_X = model(Y)
            
            # Loss and gradient optimization block
            if args.mode == "deepmure":
                if is_distributed or isinstance(model, nn.DataParallel):
                    surrogate_mse.model = model.module
                model.eval()
                loss = surrogate_mse.estimate_check(Y, pred_X, X, verbose=False)
                model.train()
                loss.backward()
                true_loss_val = true_mse(X, pred_X).detach().item()
            else:
                loss = true_mse(X, pred_X)
                loss.backward()
                true_loss_val = loss.item()
                with torch.no_grad():
                    model.eval()
                    loss = surrogate_mse.estimate_check(Y, pred_X.detach(), X, verbose=False)
            
            # Gradient clipping calculation
            if args.clip_mode == 'norm':
                act_norm = nn.utils.clip_grad_norm_(model.parameters(), args.clip_threshold).item()
                clip_norm = min(act_norm, args.clip_threshold)
            elif args.clip_mode == 'value':
                nn.utils.clip_grad_value_(model.parameters(), args.clip_threshold)
                p_grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
                act_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in p_grads]), 2).item() if p_grads else 0.0
                clip_norm = act_norm
            else:
                p_grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
                act_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in p_grads]), 2).item() if p_grads else 0.0
                clip_norm = act_norm

            gradient_norm_hist.append(act_norm)
            clipped_grad_norm_hist.append(clip_norm)
            
            optimizer.step()
            scheduler.step()
            
            surrogate_loss_hist.append(loss.item())
            true_loss_hist.append(true_loss_val)
            
            # Periodic evaluation execution block       
            if total_itrs % len(train_loader) == 0 and is_master:
                model.eval()
                with torch.no_grad():
                    val_Y, val_X, _ = next(iter(val_loader))
                    val_Y, val_X = val_Y.to(device), val_X.to(device)
                    
                    if is_distributed or isinstance(model, nn.DataParallel):
                        val_surrogate_mse.model = model.module
                        pred_val_X = model(val_Y)
                    else:
                        pred_val_X = model(val_Y)
                        
                    v_true = true_mse(val_X, pred_val_X).item()
                    v_surr = val_surrogate_mse.estimate_check(val_Y, pred_val_X, val_X, verbose=False).item()
                    v_noisy = true_mse(val_X, val_Y).item()
                
                # Enhanced print statement providing side-by-side visibility into both tracking frameworks
                print(
                    f"[{epoch+1}/{args.epochs}, Itr {total_itrs}]: "
                    f"Noisy MSE/PSNR: {v_noisy:.5f}/{mse_to_psnr(v_noisy):.2f} | "
                    f"Train True MSE: {true_loss_val:.5f}/{mse_to_psnr(true_loss_val):.2f} | "
                    f"Train Surr Est: {loss.item():.5f}/{mse_to_psnr(loss.item()):.2f} | "
                    f"Val True MSE: {v_true:.5f}/{mse_to_psnr(v_true):.2f} | "
                    f"Val Surr Est: {v_surr:.5f}/{mse_to_psnr(v_surr):.2f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                )
                
                # Update history metrics (keeps plotting arrays populated perfectly for both modes)
                true_val_loss_hist.append(v_true)
                surrogate_val_loss_hist.append(v_surr)
                noisy_loss_hist.append(v_noisy)
                
                # Check and preserve optimal model weights
                if v_true < best_loss or total_itrs % 50 == 0:
                    best_loss = v_true
                    psnr_str = f"{mse_to_psnr(v_true):.2f}"
                    ckpt = f"./{args.arch}/weights/{save_prefix}_{args.mode}_{args.looks}_bestmodel_{total_itrs}_{psnr_str}.pth"
                    torch.save(model.module.state_dict() if (is_distributed or isinstance(model, nn.DataParallel)) else model.state_dict(), ckpt)
                    
                    log_file.write(f"{datetime.now()} {epoch} {total_itrs} {mse_to_psnr(true_loss_val):.2f} {psnr_str}\n")
                    log_file.flush()
                    
                    # Generate evaluation sample mosaics
                    imgs = [val_Y[b] for b in range(val_Y.shape[0])] + [pred_val_X[b] for b in range(pred_val_X.shape[0])] + [val_X[b] for b in range(val_X.shape[0])]
                    create_image_grid(imgs, 3, val_Y.shape[0], f"./{args.arch}/figures/{save_prefix}_{args.mode}_{args.looks}_looks_best_valimages_.png")
                
                # Render tracking analytics dashboards
                tx = np.arange(len(true_loss_hist))
                vx = np.linspace(0, len(true_loss_hist), len(true_val_loss_hist))
                
                create_professional_loss_plot(tx, true_loss_hist, surrogate_loss_hist, vx, true_val_loss_hist, surrogate_val_loss_hist, noisy_loss_hist, args.looks, epoch, total_itrs, f"./{args.arch}/figures/{save_prefix}_{args.mode}_{args.looks}_looks_mse.png", use_psnr=False,ylabel='MSE Loss')
                create_professional_loss_plot(tx, true_loss_hist, surrogate_loss_hist, vx, true_val_loss_hist, surrogate_val_loss_hist, noisy_loss_hist, args.looks, epoch, total_itrs, f"./{args.arch}/figures/{save_prefix}_{args.mode}_{args.looks}_looks_psnr.png", use_psnr=True,ylabel='PSNR (dB)')
                create_gradient_norm_plot(gradient_norm_hist, clipped_grad_norm_hist, f"./{args.arch}/figures/{save_prefix}_{args.mode}_{args.looks}_gradnorm.png")
                
                if len(surrogate_mse.histories) > 0:
                    th, tth = np.array(surrogate_mse.histories), np.array(surrogate_mse.true_histories)
                    create_stats_plot(th, tth, f"./{args.arch}/figures/{save_prefix}_{args.mode}_{args.looks}_looks_stats_loss.png", 'loss')
                    create_stats_plot(th, tth, f"./{args.arch}/figures/{save_prefix}_{args.mode}_{args.looks}_looks_stats_inout.png", 'inout')
                    create_stats_plot(th, tth, f"./{args.arch}/figures/{save_prefix}_{args.mode}_{args.looks}_looks_stats_debug.png", 'debug')
                    
                # Store structural historical metrics files
                np.save(f"./{args.arch}/figures/{save_prefix}_{args.mode}_{args.looks}_train.npy", np.column_stack([true_loss_hist, surrogate_loss_hist]))
                np.save(f"./{args.arch}/figures/{save_prefix}_{args.mode}_{args.looks}_val.npy", np.column_stack([true_val_loss_hist, surrogate_val_loss_hist, noisy_loss_hist]))
                
            total_itrs += 1

    if is_master:
        log_file.close()
        if is_distributed: dist.destroy_process_group()
        print("Training execution pipeline finished successfully.")

# ============================================================================
# ENTRYPOINT CONTEXT
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production Training Wrapper for DeepMURE Optimization Framework")
    
    # Path & Environment Configurations
    parser.add_argument('--train_path', type=str, default='./data/s1hybrid/train/')
    parser.add_argument('--val_path', type=str, default='./data/s1hybrid/val/')
    parser.add_argument('--arch', type=str, default='SpeckleFormer_s1hybrid', help='Identifier token for output artifacts directory')
    parser.add_argument('--model_type', type=str, choices=['speckleformer', 'dncnn', 'unet'], default='speckleformer')
    
    # Hyperparameters
    parser.add_argument('--looks', type=int, default=4, help='Gamma noise distribution parameters (Equivalent Look Count)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lr_decay', type=float, default=1.0)
    
    # Structural Run Parameters
    parser.add_argument('--mode', type=str, choices=['deepmure', 'oracle'], default='oracle', help="Training criteria framework selection")
    parser.add_argument('--clip_mode', type=str, choices=['norm', 'value', 'none'], default='norm')
    parser.add_argument('--clip_threshold', type=float, default=1.0)
    parser.add_argument('--no_hessian', action='store_true', help="Bypasses Hessian computations inside the DeepMURE loop if specified")
    
    # Multi-GPU / Compute Cluster Settings
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='List of targeted CUDA physical GPU devices')
    parser.add_argument('--multigpu', action='store_true', help='Enables native DistributedDataParallel multi-GPU wrapper execution')
    parser.add_argument('--world_size', type=int, default=1, help='Total computing system nodes for DDP distributed configuration')
    
    parser.add_argument('--img_h', type=int, default=512)
    parser.add_argument('--img_w', type=int, default=512)

    cmd_args = parser.parse_args()
    run_training_pipeline(cmd_args)