"""
model_inference.py

This file defines the UNet3D + diffusion model architecture for cloud motion prediction,
loads a trained checkpoint, and provides an interface for running inference.

Expected Input:
- A torch.Tensor of shape (1, 6, 6, 128, 128)
  - 1: batch size
  - 6: timesteps (past frames)
  - 6: spectral bands per frame
  - 128x128: spatial resolution

Usage (Backend Integration):
1. Call `load_model(checkpoint_path)` once at server startup.
2. Preprocess input using `normalize_sequence(input_tensor)`:
   - Input shape must be (6, 6, 128, 128)
   - Then add batch dim: `input_tensor.unsqueeze(0)` → (1, 6, 6, 128, 128)
3. Call `predict_future_frames(diffusion, input_tensor, device)`
   - Returns: (1, 8, 6, 128, 128) → original 6 + 2 predicted frames

Note:
- The model requires input to be normalized to [-1, 1] per band.
- The checkpoint must contain 'model_state_dict' and 'diffusion_state_dict'
"""



# importing libraries
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import mean_absolute_error as mae



# Utility Functions - psnr, get_timestep_embedding
def psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# Model Components

# ResidualBlock3D
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResidualBlock3D, self).__init__()
        self.same_channels = in_channels == out_channels

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding)

        self.residual = nn.Identity() if self.same_channels else nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return self.relu(out + residual)


# Encoder3D
class Encoder3D(nn.Module):
    def __init__(self, in_channels, base_channels=64, time_embed_dim=128):
        super(Encoder3D, self).__init__()

        self.enc1 = nn.Sequential(
            ResidualBlock3D(in_channels, base_channels),
            ResidualBlock3D(base_channels, base_channels)
        )

        self.down1 = nn.Conv3d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.enc2 = nn.Sequential(
            ResidualBlock3D(base_channels * 2, base_channels * 2),
            ResidualBlock3D(base_channels * 2, base_channels * 2)
        )

        self.down2 = nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.enc3 = nn.Sequential(
            ResidualBlock3D(base_channels * 4, base_channels * 4),
            ResidualBlock3D(base_channels * 4, base_channels * 4)
        )

        # 🔌 Add timestep projection layers
        self.time_proj1 = nn.Linear(time_embed_dim, in_channels)  # project to 6 channels, not 64
        self.time_proj2 = nn.Linear(time_embed_dim, base_channels * 2)
        self.time_proj3 = nn.Linear(time_embed_dim, base_channels * 4)

    def forward(self, x, time_emb):
        # time_emb: (B, time_embed_dim)

        B, _, D, H, W = x.shape  # Batch, Channels, Depth, Height, Width

        # Project time embedding and reshape for broadcasting
        t1 = self.time_proj1(time_emb).view(B, -1, 1, 1, 1)
        t2 = self.time_proj2(time_emb).view(B, -1, 1, 1, 1)
        t3 = self.time_proj3(time_emb).view(B, -1, 1, 1, 1)

        # Inject time embeddings into encoder blocks
        x1 = self.enc1(x + t1)
        x2 = self.enc2(self.down1(x1) + t2)
        x3 = self.enc3(self.down2(x2) + t3)

        return x1, x2, x3



# Bottleneck3D
class Bottleneck3D(nn.Module):
    def __init__(self, in_channels, time_embed_dim=128):
        super(Bottleneck3D, self).__init__()
        self.time_proj = nn.Linear(time_embed_dim, in_channels)

        self.block = nn.Sequential(
            ResidualBlock3D(in_channels, in_channels),
            ResidualBlock3D(in_channels, in_channels)
        )

    def forward(self, x, time_emb):
        # time_emb shape: (B, time_embed_dim)
        B = x.shape[0]
        t = self.time_proj(time_emb).view(B, -1, 1, 1, 1)
        x = x + t
        return self.block(x)



# Decoder3D
class Decoder3D(nn.Module):
    def __init__(self, base_channels=64, out_channels=6, time_embed_dim=128):
        super(Decoder3D, self).__init__()

        # Upsample from bottleneck: 4*BC → 2*BC
        self.up1 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2,
                                      kernel_size=4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            ResidualBlock3D(base_channels * 4, base_channels * 2),
            ResidualBlock3D(base_channels * 2, base_channels * 2)
        )

        # Upsample from second block: 2*BC → BC
        self.up2 = nn.ConvTranspose3d(base_channels * 2, base_channels,
                                      kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            ResidualBlock3D(base_channels * 2, base_channels),
            ResidualBlock3D(base_channels, base_channels)
        )

        # Final 1×1 convolution to project to output channels
        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

        # 🔌 Timestep embedding projection layers:
        #   t1 injects into 4*BC feature map (after concat of up1 + x2)
        self.time_proj1 = nn.Linear(time_embed_dim, base_channels * 4)
        #   t2 injects into 2*BC feature map (after concat of up2 + x1)
        self.time_proj2 = nn.Linear(time_embed_dim, base_channels * 2)

    def forward(self, x3, x2, x1, time_emb):
        """
        x3: bottleneck output (B, 4*BC, D, H, W)
        x2: skip from encoder level 2 (B, 2*BC, D*2, H*2, W*2)
        x1: skip from encoder level 1 (B,   BC, D*4, H*4, W*4)
        time_emb: (B, time_embed_dim)
        """
        B = x3.shape[0]

        # Project timestep embeddings
        t1 = self.time_proj1(time_emb).view(B, -1, 1, 1, 1)  # (B, 256,1,1,1)
        t2 = self.time_proj2(time_emb).view(B, -1, 1, 1, 1)  # (B, 128,1,1,1)

        # First decoding stage
        x = self.up1(x3)                   # (B, 128, D*2, H*2, W*2)
        x = torch.cat([x, x2], dim=1)      # (B, 256, D*2, H*2, W*2)
        x = self.dec1(x + t1)              # inject t1 here

        # Second decoding stage
        x = self.up2(x)                    # (B,  64, D*4, H*4, W*4)
        x = torch.cat([x, x1], dim=1)      # (B, 128, D*4, H*4, W*4)
        x = self.dec2(x + t2)              # inject t2 here

        # Final output projection
        out = self.out_conv(x)             # (B, out_channels, D*4, H*4, W*4)
        return out



# SinusoidalPositionEmbeddings
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        """
        Create sinusoidal timestep embeddings (same as get_timestep_embedding).
        Input: timesteps (B,) — integer tensor
        Output: (B, embedding_dim)
        """
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb



# UNet3D_Full
class UNet3D_Full(nn.Module):
    def __init__(self, time_steps=6, image_channels=6, base_channels=64, time_embed_dim=128):
        super(UNet3D_Full, self).__init__()

        self.encoder = Encoder3D(image_channels, base_channels)
        self.bottleneck = Bottleneck3D(base_channels * 4, time_embed_dim)
        self.decoder = Decoder3D(base_channels, out_channels=image_channels, time_embed_dim=time_embed_dim)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.ReLU()
        )
        self.time_embed_dim = time_embed_dim

    def forward(self, x, t):
        # x: (B, C=6, T=8, H, W), t: scalar or tensor of shape (B,)
        time_emb = self.time_mlp(t)  # shape: (B, time_embed_dim)

        x1, x2, x3 = self.encoder(x, time_emb)
        bottleneck = self.bottleneck(x3, time_emb)
        out = self.decoder(bottleneck, x2, x1, time_emb)
        return out



# GaussianDiffusion
def psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))


class GaussianDiffusion(nn.Module):
    def __init__(self, model, image_size, timesteps=1000, sampling_timesteps=250):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.loss_fn = nn.MSELoss()

    def forward(self, x, t=None):
        # x shape: (B, T, C, H, W) → model expects (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

        if t is None:
            t = torch.randint(0, self.timesteps, (x.shape[0],), device=x.device)

        pred = self.model(x, t)  # (B, C, T, H, W)

        # --- MSE ---
        mse_loss = self.loss_fn(pred, x)

        # --- SSIM ---
        B, C, T, H, W = x.shape
        ssim_loss = 0.0

        for t_idx in range(T):
            pred_t = pred[:, :, t_idx, :, :]   # (B, C, H, W)
            target_t = x[:, :, t_idx, :, :]    # (B, C, H, W)
            ssim_val = ssim(pred_t, target_t, data_range=1.0)
            ssim_loss += 1 - ssim_val  # convert SSIM to loss

        ssim_loss = ssim_loss / T

        # --- MAE & PSNR ---
        mae_val = mae(pred.contiguous(), x.contiguous())  # Fix .view() error
        psnr_val = psnr(pred, x)

        # --- Combine ---
        total_loss = mse_loss + 0.5 * ssim_loss
        return total_loss, mse_loss.detach(), ssim_loss.detach(), mae_val.detach(), psnr_val

    @torch.no_grad()
    def sample(self, cond, batch_size=1):
        B, T, C, H, W = cond.shape  # cond: (B, 6, C, H, W)

        x = torch.randn(batch_size, 2, C, H, W, device=cond.device)  # predict 2 future frames

        for step in range(self.sampling_timesteps):
            combined = torch.cat([cond, x], dim=1)        # (B, 8, C, H, W)
            combined = combined.permute(0, 2, 1, 3, 4)     # (B, C, T=8, H, W)

            t = torch.full((B,), step, device=cond.device, dtype=torch.long)
            x_pred = self.model(combined, t)              # (B, C, T=8, H, W)

            x = x_pred.permute(0, 2, 1, 3, 4)[:, 6:]       # keep only T7/T8

        return torch.cat([cond, x], dim=1)  # Final shape: (B, 8, C, H, W)


# Normalization
def normalize_sequence(x):
    x_reshaped = x.permute(1, 0, 2, 3).reshape(6, -1)
    mins = x_reshaped.min(dim=1, keepdim=True)[0]
    maxs = x_reshaped.max(dim=1, keepdim=True)[0]
    norm = (x_reshaped - mins) / (maxs - mins + 1e-5)
    norm = norm * 2 - 1
    x_norm = norm.view(6, x.shape[0], 128, 128).permute(1, 0, 2, 3)
    return x_norm



# Load model - using CHECKPOINT file
def load_model(checkpoint_path, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = UNet3D_Full(
        time_steps=6,
        image_channels=6,
        base_channels=64
    ).to(device)

    diffusion = GaussianDiffusion(
        model=model,
        image_size=128,
        timesteps=1000
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    diffusion.load_state_dict(checkpoint['diffusion_state_dict'])

    model.eval()
    diffusion.eval()

    return diffusion, device



# Run Inference - prediction
@torch.no_grad()
def predict_future_frames(diffusion, cond, device):
    cond = cond.to(device)  # (B, 6, 6, 128, 128)
    return diffusion.sample(cond, batch_size=cond.shape[0]).cpu()  # (B, 8, 6, 128, 128)











