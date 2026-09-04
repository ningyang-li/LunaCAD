import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class UAGR(nn.Module):
    """Anisotropic Distribution Ball (UAGR) module，processes tensors (B, N, D)"""
    def __init__(self, dim, num_balls=32, use_diag_cov=True, tau=1.0):
        super().__init__()
        self.dim = dim
        self.num_balls = num_balls
        self.use_diag_cov = use_diag_cov
        self.tau_base = tau
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        self.centers = nn.Parameter(torch.randn(num_balls, dim) * 0.01)
        
        if use_diag_cov:
            self.log_sigma = nn.Parameter(torch.zeros(num_balls, dim))
        else:
            self.log_radius = nn.Parameter(torch.zeros(num_balls, 1))

        self.norm = nn.LayerNorm(dim)
        self.refine = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        B, N, D = x.shape
        z = self.norm(x)

        # (B, N, K, D)
        # Scaled Distance based on Anisotropic Balls
        dif = z.unsqueeze(2) - self.centers.view(1, 1, self.num_balls, D)
        if self.use_diag_cov:
            sigma = F.softplus(self.log_sigma).view(1, 1, self.num_balls, D)
        else:
            sigma = F.softplus(self.log_radius).view(1, 1, self.num_balls, 1)
        dist2 = ((dif / (sigma + 1e-6)) ** 2).sum(-1) # (B, N, K)

        # Uncertainty-Aware Adaptive Re-Distribution
        att_raw = F.softmax(-dist2, dim=-1)
        entropy = -(att_raw * torch.log(att_raw + 1e-8)).sum(-1)  # (B, N)
        entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-6)
        tau_adaptive = self.tau_base * (1 + F.softplus(self.alpha) * entropy.unsqueeze(-1))
        att = F.softmax(-dist2 / tau_adaptive.clamp(min=1e-6), dim=-1)
        recon = torch.matmul(att, self.centers)  # (B, N, K) (K, D) => (B, N, D)    # Centers will be updated during back propagation

        # Residual
        out = self.refine(recon) + x
        return out, att, dif, sigma


def wasserstein_diversity_loss(centers):
    """Wasserstein-based diversity loss for GBC centers (avoid collapse centers)"""
    K, D = centers.shape
    if K <= 1:
        return torch.tensor(0.0, device=centers.device)

    mu_hat = torch.mean(centers, dim=0)
    centers_centered = centers - mu_hat
    Sigma_hat = (centers_centered.t() @ centers_centered) / K

    # cast to float32
    Sigma_hat = Sigma_hat.float()

    term_mean = torch.sum(mu_hat ** 2)
    term_trace = torch.trace(Sigma_hat)

    try:
        eigenvalues = torch.linalg.eigvalsh(Sigma_hat)
        term_trace_sqrt = torch.sum(torch.sqrt(torch.clamp(eigenvalues, min=0)))
    except torch.linalg.LinAlgError:
        return torch.tensor(0.0, device=centers.device)

    w2 = term_mean + term_trace - 2.0 / math.sqrt(D) * term_trace_sqrt
    
    norms = torch.norm(centers, dim=1)  # (K,)
    norm_variance = torch.var(norms)    # variance
    w2 = w2 - 0.1 * norm_variance       # reduce w2 for small norm_variance
    
    return torch.clamp(w2, min=0).to(centers.dtype)  # cast to float32


def calculate_scale_loss(att, dif, log_sigma):
    """Anisotropic scale consistency loss"""
    if not log_sigma.requires_grad:
        return torch.tensor(0.0, device=att.device)

    with torch.no_grad():
        weight = att.detach().unsqueeze(-1)  # (B, N, K, 1)
        Mk = weight.sum(dim=1, keepdim=True) + 1e-6  # (B, 1, K, 1)

    diff_elem_sq = dif.pow(2)  # (B, N, K, D)
    s_k_sq = (weight * diff_elem_sq).sum(dim=1, keepdim=True) / Mk  # (B, 1, K, D)
    s_k_sq = s_k_sq.mean(dim=0).squeeze(0)  # (K, D)

    # cast to float32
    s_k_sq_f32 = s_k_sq.float()
    sigma_k_sq = F.softplus(log_sigma).pow(2).float()
    loss = F.mse_loss(s_k_sq_f32.detach(), sigma_k_sq)
    return loss.to(log_sigma.dtype)



