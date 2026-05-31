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
    name="hybrid_ur_nsur_cubic",
    group="hybrid",
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
        "reg_weight_ur": 1e-4,
        "reg_weight_nsur": 1e-4,
        "reg_warmup_frac": 0.3,
        "patience": 300,
        "min_delta": 0.0,
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

def make_cubic_split(total_points, noise_std, val_fraction, seed, x_range):
    rng = np.random.default_rng(seed)
    x_all = np.linspace(x_range[0], x_range[1], total_points, dtype=np.float32)
    y_all = x_all**3 + noise_std * rng.standard_normal(size=total_points).astype(np.float32)
    
    idx = np.arange(len(x_all))
    rng.shuffle(idx)
    split = int(len(idx) * (1.0 - val_fraction))
    return (torch.from_numpy(x_all[idx[:split]]).unsqueeze(1), torch.from_numpy(y_all[idx[:split]]).unsqueeze(1),
            torch.from_numpy(x_all[idx[split:]]).unsqueeze(1), torch.from_numpy(y_all[idx[split:]]).unsqueeze(1))

set_seed(config.seed)
x_tr, y_tr, x_val, y_val = make_cubic_split(
    config.total_points, config.noise_std, config.val_fraction, config.seed, config.x_range
)
x_tr, y_tr = x_tr.to(device), y_tr.to(device)
x_val, y_val = x_val.to(device), y_val.to(device)

# Model & Hybrid Loss Function
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
        mu = out[:, 0:1]
        eps = 1e-3
        lambda_ = F.softplus(out[:, 1:2]) + eps
        alpha  = 1.0 + F.softplus(out[:, 2:3]) + eps
        beta   = F.softplus(out[:, 3:4]) + eps
        return mu, lambda_, alpha, beta

def hybrid_urer_nsur_loss(y, gamma, lambda_, alpha, beta, *, reg_w_ur=1e-4, reg_w_nsur=1e-4):
    eps = 1e-8
    
    twoBlambda = 2 * beta * (1 + lambda_)
    nll = (
        0.5 * torch.log(np.pi / lambda_)
        - alpha * torch.log(twoBlambda)
        + (alpha + 0.5) * torch.log(lambda_ * (y - gamma) ** 2 + twoBlambda)
        + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    ).mean()

    error_abs = torch.abs(y - gamma)
    ur_penalty = torch.log(torch.exp(alpha - 1.0) - 1.0 + eps)
    reg_ur = (-error_abs * ur_penalty).mean()

    error_sq = (y - gamma) ** 2
    inverse_total_uncertainty = (lambda_ * (alpha - 1.0)) / (beta * (lambda_ + 1.0) + eps)
    reg_nsur = (0.5 * error_sq * inverse_total_uncertainty).mean()

    total_loss = nll + (reg_w_ur * reg_ur) + (reg_w_nsur * reg_nsur)

    return total_loss, nll.detach(), reg_ur.detach(), reg_nsur.detach()

# Training Loop
model = EDLRegressor(hidden=config.hidden_size).to(device)
opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

reg_warmup_epochs = int(config.reg_warmup_frac * config.epochs)
best_state = None
best_val_loss = float("inf")
epochs_no_improve = 0

for epoch in range(config.epochs):
    model.train()
    mu, lam, alp, bet = model(x_tr)
    
    warmup_factor = min(1.0, epoch / max(1, reg_warmup_epochs))
    w_ur = float(config.reg_weight_ur) * warmup_factor
    w_nsur = float(config.reg_weight_nsur) * warmup_factor
    
    loss_tr, nll_tr, reg_ur_tr, reg_nsur_tr = hybrid_urer_nsur_loss(
        y_tr, mu, lam, alp, bet, reg_w_ur=w_ur, reg_w_nsur=w_nsur
    )

    opt.zero_grad()
    loss_tr.backward()
    opt.step()

    model.eval()
    with torch.no_grad():
        mu_v, lam_v, a_v, b_v = model(x_val)
        loss_v, nll_v, reg_ur_v, reg_nsur_v = hybrid_urer_nsur_loss(
            y_val, mu_v, lam_v, a_v, b_v, reg_w_ur=w_ur, reg_w_nsur=w_nsur
        )

    if loss_v.item() + float(config.min_delta) < best_val_loss:
        best_val_loss = loss_v.item()
        best_state = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= int(config.patience):
            break

if best_state is not None:
    model.load_state_dict(best_state)

# Test Evaluation
model.eval()
TRAIN_L, TRAIN_R = config.x_range
XMIN, XMAX = config.test_x_range
x_test = torch.linspace(XMIN, XMAX, 400, device=device).unsqueeze(1)

with torch.no_grad():
    mu_t, lam_t, alp_t, bet_t = model(x_test)
    ale_var = torch.clamp(bet_t / (alp_t - 1), 1e-12)
    epi_var = torch.clamp(bet_t / (lam_t * (alp_t - 1)), 1e-12)
    total_var = ale_var + epi_var

x_np      = x_test.squeeze(1).cpu().numpy()
y_pred    = mu_t.squeeze(1).cpu().numpy()
ale_std   = ale_var.sqrt().squeeze(1).cpu().numpy()
epi_std   = epi_var.sqrt().squeeze(1).cpu().numpy()
total_std = total_var.sqrt().squeeze(1).cpu().numpy()
y_true    = x_np**3 

# Final Metrics Extraction
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

print("\nFinal Evaluation Metrics (Hybrid Cubic):")
print("-" * 40)

print(f"ID RMSE:               {id_rmse:.4f}")
print(f"OOD RMSE:              {ood_rmse:.4f}")
print(f"Max OOD Error:         {ood_max_error:.4f}")

print(f"Peak OOD Epistemic σ:  {peak_ood_epi:.4f}")
print(f"Mean ID Epistemic σ:   {mean_id_epi:.4f}")
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