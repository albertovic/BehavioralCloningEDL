import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

# Initialise W&B
run = wandb.init(
    project="edl_regression",
    name="UE_edl_sine", 
    group="UE",
    config={
        "total_points": 1000,
        "noise_std": 0.05,
        "gap_region": (np.pi, 2 * np.pi),
        "val_fraction": 0.2,
        "seed": 50,
        "x_range": (0.0, 3 * np.pi),
        "hidden_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 20000,
        "reg_weight": 1e-4,
        "reg_warmup_frac": 0.3,
        "patience": 300,        
        "min_delta": 0.0,       
        "gap_probe_every": 50,
    },
)
config = wandb.config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Generation
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_sine_gap_split(total_points, noise_std, gap_region, val_fraction, seed, x_range):
    rng = np.random.default_rng(seed)
    x_all = np.linspace(x_range[0], x_range[1], total_points, dtype=np.float32)
    y_all = np.sin(x_all) + noise_std * rng.standard_normal(size=total_points).astype(np.float32)
    
    mask = (x_all < gap_region[0]) | (x_all > gap_region[1])
    x_id, y_id = x_all[mask], y_all[mask]
    
    idx = np.arange(len(x_id))
    rng.shuffle(idx)
    split = int(len(idx) * (1.0 - val_fraction))
    
    return (torch.from_numpy(x_id[idx[:split]]).unsqueeze(1), torch.from_numpy(y_id[idx[:split]]).unsqueeze(1),
            torch.from_numpy(x_id[idx[split:]]).unsqueeze(1), torch.from_numpy(y_id[idx[split:]]).unsqueeze(1))

set_seed(config.seed)
x_tr, y_tr, x_val, y_val = make_sine_gap_split(config.total_points, config.noise_std, config.gap_region, config.val_fraction, config.seed, config.x_range)
x_tr, y_tr, x_val, y_val = x_tr.to(device), y_tr.to(device), x_val.to(device), y_val.to(device)

# Model & UE Loss
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
        gamma = out[:, 0:1]
        eps = 1e-4
        lambda_ = F.softplus(out[:, 1:2]) + eps
        alpha  = 1.0 + F.softplus(out[:, 2:3]) + eps
        beta   = F.softplus(out[:, 3:4]) + eps
        return gamma, lambda_, alpha, beta

def ue_loss(y, gamma, lambda_, alpha, beta, *, reg_weight=1e-4):
    twoBlambda = 2 * beta * (1 + lambda_)
    nll = (
        0.5 * torch.log(np.pi / lambda_)
        - alpha * torch.log(twoBlambda)
        + (alpha + 0.5) * torch.log(lambda_ * (y - gamma) ** 2 + twoBlambda)
        + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    ).mean()

    w_st = torch.sqrt(beta * (1 + lambda_) / (alpha * lambda_ + 1e-12))
    reg = (torch.abs((y - gamma) / (w_st + 1e-12)) * (lambda_ + 2 * alpha)).mean()

    return (nll + reg_weight * reg), nll.detach(), reg.detach()

# Training Loop
model = EDLRegressor(hidden=config.hidden_size).to(device)
opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

reg_warmup_epochs = int(config.reg_warmup_frac * config.epochs)
best_val_loss = float("inf")
epochs_no_improve = 0
best_state = None

for epoch in range(config.epochs):
    model.train()
    gamma, lam, alp, bet = model(x_tr)
    reg_w = float(config.reg_weight) * min(1.0, epoch / max(1, reg_warmup_epochs))
    loss_tr, nll_tr, reg_tr = ue_loss(y_tr, gamma, lam, alp, bet, reg_weight=reg_w)
    
    opt.zero_grad()
    loss_tr.backward()
    opt.step()
    
    wandb.log({"train/loss": loss_tr.item(), "train/nll": nll_tr.item(), "train/reg": reg_tr.item(), "epoch": epoch})

    model.eval()
    with torch.no_grad():
        mu_v, lam_v, alp_v, bet_v = model(x_val)
        loss_v, nll_v, reg_v = ue_loss(y_val, mu_v, lam_v, alp_v, bet_v, reg_weight=reg_w)
            
        wandb.log({"val/loss": loss_v.item(), "val/nll": nll_v.item()}, step=epoch)

        if loss_v < (best_val_loss - config.min_delta):
            best_val_loss = loss_v
            epochs_no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
                
        if epochs_no_improve >= config.patience:
            break

if best_state is not None:
    model.load_state_dict(best_state)
torch.save(model.state_dict(), "best_ue_sine_model.pt")

# Test Evaluation
model.eval()
TRAIN_L, TRAIN_R = config.x_range
XMIN, XMAX = TRAIN_L, 4 * np.pi
x_test = torch.linspace(XMIN, XMAX, 400, device=device).unsqueeze(1)

with torch.no_grad():
    mu_t, lam_t, alp_t, bet_t = model(x_test)
    
    ale_var_std = torch.clamp(bet_t / (alp_t - 1), 1e-12)
    epi_var_std = torch.clamp(bet_t / (lam_t * (alp_t - 1)), 1e-12)
    
    ue_ale_raw = torch.sqrt(bet_t * (1 + lam_t) / (alp_t * lam_t + 1e-12))
    ue_epi_raw = torch.sqrt(1.0 / (lam_t + 1e-12))

x_np = x_test.squeeze(1).cpu().numpy()
y_pred = mu_t.squeeze(1).cpu().numpy()
y_true = np.sin(x_np)

ale_std = ale_var_std.sqrt().squeeze(1).cpu().numpy()
epi_std = epi_var_std.sqrt().squeeze(1).cpu().numpy()
total_std = np.sqrt(ale_std**2 + epi_std**2)

ue_ale = ue_ale_raw.squeeze(1).cpu().numpy()
ue_epi = ue_epi_raw.squeeze(1).cpu().numpy()
ue_total_std = np.sqrt(ue_ale**2 + ue_epi**2)

# Final Metrics
GAP_L, GAP_R = config.gap_region
ood_mask = ((x_np >= GAP_L) & (x_np <= GAP_R)) | (x_np > TRAIN_R)
id_mask = ((x_np >= TRAIN_L) & (x_np < GAP_L)) | ((x_np > GAP_R) & (x_np <= TRAIN_R))

id_rmse = np.sqrt(np.mean((y_true[id_mask] - y_pred[id_mask])**2))
ood_rmse = np.sqrt(np.mean((y_true[ood_mask] - y_pred[ood_mask])**2))
ood_max_error = np.max(np.abs(y_true[ood_mask] - y_pred[ood_mask]))

peak_ood_epi_std = np.max(epi_std[ood_mask])
mean_id_epi_std = np.mean(epi_std[id_mask])
mean_ood_ale_std = np.mean(ale_std[ood_mask])
mean_id_ale_std = np.mean(ale_std[id_mask])

id_ear_std = mean_id_epi_std / (mean_id_ale_std + 1e-8)
ood_ear_std = np.mean(epi_std[ood_mask] / (ale_std[ood_mask] + 1e-8))

peak_ood_epi_ue = np.max(ue_epi[ood_mask])
mean_id_epi_ue = np.mean(ue_epi[id_mask])
mean_ood_ale_ue = np.mean(ue_ale[ood_mask])
mean_id_ale_ue = np.mean(ue_ale[id_mask])

id_ear_ue = mean_id_epi_ue / (mean_id_ale_ue + 1e-8)
ood_ear_ue = np.mean(ue_epi[ood_mask] / (ue_ale[ood_mask] + 1e-8))

def calc_nll(y_t, y_p, sig):
    var = sig**2 + 1e-8
    return np.mean(0.5 * np.log(2 * np.pi * var) + ((y_t - y_p)**2) / (2 * var))

id_coverage_std = np.mean(np.abs(y_true[id_mask] - y_pred[id_mask]) <= 2 * total_std[id_mask]) * 100
ood_coverage_std = np.mean(np.abs(y_true[ood_mask] - y_pred[ood_mask]) <= 2 * total_std[ood_mask]) * 100
id_nll_std = calc_nll(y_true[id_mask], y_pred[id_mask], total_std[id_mask])
ood_nll_std = calc_nll(y_true[ood_mask], y_pred[ood_mask], total_std[ood_mask])

id_coverage_ue = np.mean(np.abs(y_true[id_mask] - y_pred[id_mask]) <= 2 * ue_total_std[id_mask]) * 100
ood_coverage_ue = np.mean(np.abs(y_true[ood_mask] - y_pred[ood_mask]) <= 2 * ue_total_std[ood_mask]) * 100
id_nll_ue = calc_nll(y_true[id_mask], y_pred[id_mask], ue_total_std[id_mask])
ood_nll_ue = calc_nll(y_true[ood_mask], y_pred[ood_mask], ue_total_std[ood_mask])

print("\nFinal Evaluation Metrics:")
print("-" * 40)
print(f"ID RMSE:               {id_rmse:.4f}")
print(f"OOD RMSE:              {ood_rmse:.4f}")
print(f"Max OOD Error:         {ood_max_error:.4f}")

print("\nStandard Proxies")
print(f"Peak OOD Epistemic σ:  {peak_ood_epi_std:.4f}")
print(f"Mean ID Epistemic σ:   {mean_id_epi_std:.4f}")
print(f"Mean OOD Aleatoric σ:  {mean_ood_ale_std:.4f}")
print(f"ID EAR:                {id_ear_std:.4f}")
print(f"OOD EAR:               {ood_ear_std:.4f}")
print(f"ID Coverage (2σ):      {id_coverage_std:.1f}%")
print(f"OOD Coverage (2σ):     {ood_coverage_std:.1f}%")
print(f"ID NLL:                {id_nll_std:.4f}")
print(f"OOD NLL:               {ood_nll_std:.4f}")

print("\nUE Proxies")
print(f"Peak OOD Epistemic σ:  {peak_ood_epi_ue:.4f}")
print(f"Mean ID Epistemic σ:   {mean_id_epi_ue:.4f}")
print(f"Mean OOD Aleatoric σ:  {mean_ood_ale_ue:.4f}")
print(f"ID EAR:                {id_ear_ue:.4f}")
print(f"OOD EAR:               {ood_ear_ue:.4f}")
print(f"ID Coverage (2σ):      {id_coverage_ue:.1f}%")
print(f"OOD Coverage (2σ):     {ood_coverage_ue:.1f}%")
print(f"ID NLL:                {id_nll_ue:.4f}")
print(f"OOD NLL:               {ood_nll_ue:.4f}")
print("-" * 40)