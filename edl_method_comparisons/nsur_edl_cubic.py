import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

# Initialise W&B
run = wandb.init(
    mode="disabled",
    project="edl_regression",
    name="NSUR_edl_cubic",
    group="NSUR",
    config={
        "total_points": 1000,
        "noise_std": 3.0,               
        "x_range": (-4.0, 4.0),         
        "test_x_range": (-7.0, 7.0),    
        "val_fraction": 0.2,
        "seed": 50,
        "hidden_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 20000,                
        "reg_weight": 1e-2,             
        "reg_warmup_frac": 0.3,
        "patience": 500,
        "min_delta": 0.0,
    },
)
config = wandb.config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Generation (Cubic)
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_cubic_split(total_points, noise_std, val_fraction, seed, x_range):
    rng = np.random.default_rng(seed)
    x = np.linspace(x_range[0], x_range[1], total_points, dtype=np.float32)
    
    y_clean = x**3
    y = y_clean + noise_std * rng.standard_normal(size=total_points).astype(np.float32)

    idx = np.arange(len(x))
    rng.shuffle(idx)
    split = int(len(idx) * (1.0 - val_fraction))
    tr_idx, va_idx = idx[:split], idx[split:]

    x_tr, y_tr = x[tr_idx], y[tr_idx]
    x_va, y_va = x[va_idx], y[va_idx]

    return (torch.from_numpy(x_tr).unsqueeze(1), torch.from_numpy(y_tr).unsqueeze(1),
            torch.from_numpy(x_va).unsqueeze(1), torch.from_numpy(y_va).unsqueeze(1))

set_seed(config.seed)
x_tr, y_tr, x_val, y_val = make_cubic_split(
    total_points=config.total_points,
    noise_std=config.noise_std,
    val_fraction=config.val_fraction,
    seed=config.seed,
    x_range=config.x_range,
)
x_tr, y_tr, x_val, y_val = x_tr.to(device), y_tr.to(device), x_val.to(device), y_val.to(device)

# Model & NSUR Loss
class EDLRegressor(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, x):
        out = self.net(x)
        mu, lambda_raw, alpha_raw, beta_raw = out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]
        
        eps = 1e-4
        lambda_ = F.softplus(lambda_raw) + eps
        alpha = 1.0 + F.softplus(alpha_raw) + eps
        beta = F.softplus(beta_raw) + eps
        return mu, lambda_, alpha, beta

def nsur_loss(y, mu, lambda_, alpha, beta, *, reg_weight=1e-4):
    twoBlambda = 2 * beta * (1 + lambda_)
    nll = (
        0.5 * torch.log(np.pi / lambda_)
        - alpha * torch.log(twoBlambda)
        + (alpha + 0.5) * torch.log(lambda_ * (y - mu) ** 2 + twoBlambda)
        + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    ).mean()

    error = (y - mu)
    inverse_uncertainty = (lambda_ * (alpha - 1)) / (beta * (lambda_ + 1) + 1e-12)
    nsur_reg = (error ** 2 * inverse_uncertainty).mean()

    return (nll + reg_weight * nsur_reg), nll.detach(), nsur_reg.detach()

# Training Loop
model = EDLRegressor(hidden=config.hidden_size).to(device)
opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

reg_warmup_epochs = int(config.reg_warmup_frac * config.epochs)
best_val_loss = float("inf")
epochs_no_improve = 0

for epoch in range(config.epochs):
    model.train()
    mu, lam, alp, bet = model(x_tr)
    reg_w = float(config.reg_weight) * min(1.0, epoch / max(1, reg_warmup_epochs))
    loss_tr, nll_tr, reg_tr = nsur_loss(y_tr, mu, lam, alp, bet, reg_weight=reg_w)

    opt.zero_grad()
    loss_tr.backward()
    opt.step()

    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            mu_v, lam_v, alp_v, bet_v = model(x_val)
            loss_v, nll_v, reg_v = nsur_loss(y_val, mu_v, lam_v, alp_v, bet_v, reg_weight=reg_w)
            
            ale_var = bet_v / (alp_v - 1.0)
            epi_var = bet_v / (lam_v * (alp_v - 1.0))

            ale_std = np.sqrt(ale_var.cpu().numpy().flatten())
            epi_std = np.sqrt(epi_var.cpu().numpy().flatten())

            wandb.log({
                "epoch": epoch, 
                "val/loss": loss_v.item(), 
                "val/mse": F.mse_loss(mu_v, y_val).item(),
                "uncertainty/aleatoric": ale_std.mean(), 
                "uncertainty/epistemic": epi_std.mean()
            }, step=epoch)

            if loss_v < best_val_loss:
                best_val_loss = loss_v
                epochs_no_improve = 0
                torch.save(model.state_dict(), "best_model.pt")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config.patience: 
                    break

# Test Evaluation
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

x_test = torch.linspace(config.test_x_range[0], config.test_x_range[1], 500, device=device).unsqueeze(1)
with torch.no_grad():
    mu_t, lam_t, alp_t, bet_t = model(x_test)
    
    ale_var = bet_t / (alp_t - 1.0)
    epi_var = bet_t / (lam_t * (alp_t - 1.0))
    total_var = ale_var + epi_var

    ale_std = np.sqrt(ale_var.cpu().numpy().flatten())
    epi_std = np.sqrt(epi_var.cpu().numpy().flatten())
    total_std = np.sqrt(total_var.cpu().numpy().flatten())

x_np, y_pred = x_test.cpu().numpy().flatten(), mu_t.cpu().numpy().flatten()
y_true = x_np**3

# Final Metrics
TRAIN_L, TRAIN_R = config.x_range

ood_mask = (x_np < TRAIN_L) | (x_np > TRAIN_R)
id_mask = (x_np >= TRAIN_L) & (x_np <= TRAIN_R)

id_rmse = np.sqrt(np.mean((y_true[id_mask] - y_pred[id_mask])**2))
ood_rmse = np.sqrt(np.mean((y_true[ood_mask] - y_pred[ood_mask])**2))
ood_max_error = np.max(np.abs(y_true[ood_mask] - y_pred[ood_mask]))

peak_ood_epi = np.max(epi_std[ood_mask])
mean_id_epi = np.mean(epi_std[id_mask])
mean_ood_ale = np.mean(ale_std[ood_mask])
mean_id_ale = np.mean(ale_std[id_mask])

id_ear = mean_id_epi / (mean_id_ale + 1e-8)
ood_ear = np.mean(epi_std[ood_mask] / (ale_std[ood_mask] + 1e-8))

id_coverage = np.mean(np.abs(y_true[id_mask] - y_pred[id_mask]) <= 2 * total_std[id_mask]) * 100
ood_coverage = np.mean(np.abs(y_true[ood_mask] - y_pred[ood_mask]) <= 2 * total_std[ood_mask]) * 100

def calc_nll(y_t, y_p, sig):
    var = sig**2 + 1e-8
    return np.mean(0.5 * np.log(2 * np.pi * var) + ((y_t - y_p)**2) / (2 * var))

id_nll = calc_nll(y_true[id_mask], y_pred[id_mask], total_std[id_mask])
ood_nll = calc_nll(y_true[ood_mask], y_pred[ood_mask], total_std[ood_mask])

print("\nFinal Evaluation Metrics (Cubic):")
print("-" * 40)
print("1. Predictive Accuracy")
print(f"ID RMSE:               {id_rmse:.4f}")
print(f"OOD RMSE:              {ood_rmse:.4f}")
print(f"Max OOD Error:         {ood_max_error:.4f}")

print(f"Peak OOD Epistemic σ:  {peak_ood_epi:.4f}")
print(f"Mean ID Epistemic σ:   {mean_id_epi:.4f}")
print(f"Mean OOD Aleatoric σ:  {mean_ood_ale:.4f}")
print(f"ID EAR:                {id_ear:.4f}")
print(f"OOD EAR:               {ood_ear:.4f}")

print(f"ID Coverage (2σ):      {id_coverage:.1f}%")
print(f"OOD Coverage (2σ):     {ood_coverage:.1f}%")
print(f"ID NLL:                {id_nll:.4f}")
print(f"OOD NLL:               {ood_nll:.4f}")
print("-" * 40)

wandb.run.summary.update({
    "metrics/id_rmse": id_rmse,
    "metrics/ood_rmse": ood_rmse,
    "metrics/ood_max_error": ood_max_error,
    "metrics/peak_ood_epi": peak_ood_epi,
    "metrics/mean_id_epi": mean_id_epi,
    "metrics/id_ear": id_ear,
    "metrics/ood_ear": ood_ear,
    "metrics/id_coverage_pct": id_coverage,
    "metrics/ood_coverage_pct": ood_coverage,
    "metrics/id_nll": id_nll,
    "metrics/ood_nll": ood_nll,
})

wandb.finish()