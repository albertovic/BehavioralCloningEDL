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
    name="base_edl_cubic",
    group="base",
    config={
        "function_name": "der_cubic",
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
        "reg_weight": 1e-4,
        "reg_warmup_frac": 0.3,
        "patience": 300,
        "min_delta": 0.0,  
    },
)
config = wandb.config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data 
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def target_fn_np(x: np.ndarray, name: str) -> np.ndarray:
    if name in ["der_cubic", "der_cubic_hetero"]:
        return x**3
    else:
        raise ValueError(f"Unknown function_name: {name}")

def noise_sigma_np(x: np.ndarray, base: float, name: str) -> np.ndarray:
    if name == "der_cubic":
        return base * np.ones_like(x)
    elif name == "der_cubic_hetero":
        return base * (1.0 + 3.0 * np.exp(- (x**2) / (2.0 * 2.0**2)))
    else:
        raise ValueError(f"Unknown function_name for noise: {name}")

def make_dense_split(
    total_points,
    noise_std,
    val_fraction,
    seed,
    x_range,
    function_name: str,
):
    rng = np.random.default_rng(seed)
    x_all = np.linspace(x_range[0], x_range[1], total_points, dtype=np.float32)

    y_clean = target_fn_np(x_all, function_name).astype(np.float32)
    sigmas = noise_sigma_np(x_all, base=noise_std, name=function_name)
    y_all = y_clean + (sigmas * rng.standard_normal(size=total_points).astype(np.float32))

    x_id, y_id = x_all, y_all

    idx = np.arange(len(x_id))
    rng.shuffle(idx)
    split = int(len(idx) * (1.0 - val_fraction))
    tr_idx, va_idx = idx[:split], idx[split:]

    x_tr, y_tr = x_id[tr_idx], y_id[tr_idx]
    x_va, y_va = x_id[va_idx], y_id[va_idx]

    x_tr = torch.from_numpy(x_tr).unsqueeze(1)
    y_tr = torch.from_numpy(y_tr).unsqueeze(1)
    x_va = torch.from_numpy(x_va).unsqueeze(1)
    y_va = torch.from_numpy(y_va).unsqueeze(1)
    return x_tr, y_tr, x_va, y_va

set_seed(config.seed)
x_tr, y_tr, x_val, y_val = make_dense_split(
    total_points=config.total_points,
    noise_std=float(config.noise_std),
    val_fraction=config.val_fraction,
    seed=config.seed,
    x_range=config.x_range,
    function_name=config.function_name,
)
x_tr, y_tr = x_tr.to(device), y_tr.to(device)
x_val, y_val = x_val.to(device), y_val.to(device)

wandb.log({"n_train": len(x_tr), "n_val": len(x_val)}, step=0)


# Model & Loss 
class EDLRegressor(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
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

def evidential_loss(y, mu, lambda_, alpha, beta, *, reg_weight=1e-3):
    twoBlambda = 2 * beta * (1 + lambda_)
    nll = (
        0.5 * torch.log(np.pi / lambda_)
        - alpha * torch.log(twoBlambda)
        + (alpha + 0.5) * torch.log(lambda_ * (y - mu) ** 2 + twoBlambda)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    ).mean()

    error = torch.abs(y - mu)
    total_evidence = 2 * lambda_ + alpha
    reg = (error * total_evidence).mean()

    return (nll + reg_weight * reg), nll.detach(), reg.detach()

# Train loop with validation + early stopping

model = EDLRegressor(hidden=config.hidden_size).to(device)
opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

reg_warmup_epochs = int(config.reg_warmup_frac * config.epochs)

best_state = None
best_val_loss = float("inf")
epochs_no_improve = 0
stopped_early = False
final_epoch = 0

for epoch in range(config.epochs):
    model.train()
    
    mu, lambda_, alpha, beta = model(x_tr)
    reg_w = float(config.reg_weight) * min(1.0, epoch / max(1, reg_warmup_epochs))
    loss_tr, nll_tr, reg_tr = evidential_loss(y_tr, mu, lambda_, alpha, beta, reg_weight=reg_w)

    opt.zero_grad()
    loss_tr.backward()
    opt.step()

    with torch.no_grad():
        ale_mean_tr = (beta / (alpha - 1)).mean().item()
        epi_mean_tr = (beta / (lambda_ * (alpha - 1))).mean().item()

    model.eval()
    with torch.no_grad():
        mu_v, lam_v, a_v, b_v = model(x_val)
        loss_v, nll_v, reg_v = evidential_loss(y_val, mu_v, lam_v, a_v, b_v, reg_weight=reg_w)

    wandb.log(
        {
            "epoch": epoch,
            "train/loss": loss_tr.item(),
            "train/nll":  nll_tr.item(),
            "train/reg":  reg_tr.item(),
            "train/ale_mean": ale_mean_tr,
            "train/epi_mean": epi_mean_tr,
            "val/loss": loss_v.item(),
            "val/nll":  nll_v.item(),
            "val/reg":  reg_v.item(),
            "reg_weight_effective": reg_w,
        },
        step=epoch,
    )

    if loss_v.item() + float(config.min_delta) < best_val_loss:
        best_val_loss = loss_v.item()
        best_state = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
        wandb.run.summary["best_epoch"] = epoch
        wandb.run.summary["best_val_loss"] = best_val_loss
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= int(config.patience):
            stopped_early = True
            final_epoch = epoch
            break

    final_epoch = epoch

if best_state is not None:
    model.load_state_dict(best_state)

wandb.run.summary["stopped_early"] = bool(stopped_early)
wandb.run.summary["final_epoch"] = int(final_epoch)

model.eval()
with torch.no_grad():
    mu_tr, lam_tr, a_tr, b_tr = model(x_tr)
    loss_tr_f, nll_tr_f, reg_tr_f = evidential_loss(y_tr, mu_tr, lam_tr, a_tr, b_tr, reg_weight=reg_w)
    mu_vf, lam_vf, a_vf, b_vf = model(x_val)
    loss_v_f, nll_v_f, reg_v_f = evidential_loss(y_val, mu_vf, lam_vf, a_vf, b_vf, reg_weight=reg_w)

wandb.log(
    {
        "final/train_loss": loss_tr_f.item(),
        "final/train_nll":  nll_tr_f.item(),
        "final/train_reg":  reg_tr_f.item(),
        "final/val_loss":   loss_v_f.item(),
        "final/val_nll":    nll_v_f.item(),
        "final/val_reg":    reg_v_f.item(),
    },
    step=final_epoch,
)

ckpt_path = "best_model.pt"
torch.save(model.state_dict(), ckpt_path)
artifact = wandb.Artifact(f"{wandb.run.name}_model", type="model")
artifact.add_file(ckpt_path)
wandb.log_artifact(artifact)

# Test evaluation 

XMIN, XMAX = config.test_x_range
x_test = torch.linspace(XMIN, XMAX, 400, device=device).unsqueeze(1)

with torch.no_grad():
    mu_t, lam_t, a_t, b_t = model(x_test)
    ale_var = torch.clamp(b_t / (a_t - 1), 1e-12)
    epi_var = torch.clamp(b_t / (lam_t * (a_t - 1)), 1e-12)
    total_var = ale_var + epi_var

x_np      = x_test.squeeze(1).cpu().numpy()
y_pred    = mu_t.squeeze(1).cpu().numpy()
ale_std   = ale_var.sqrt().squeeze(1).cpu().numpy()
epi_std   = epi_var.sqrt().squeeze(1).cpu().numpy()
total_std = total_var.sqrt().squeeze(1).cpu().numpy()
y_true    = target_fn_np(x_np, config.function_name)

full_rows = [
    [float(x_), float(y_t), float(y_p), float(ale), float(epi), float(tot)]
    for x_, y_t, y_p, ale, epi, tot in zip(x_np, y_true, y_pred, ale_std, epi_std, total_std)
]
wandb.log(
    {
        "full_predictions": wandb.Table(
            columns=["x", "y_true", "y_pred", "ale_std", "epi_std", "total_std"],
            data=full_rows,
        )
    },
    step=final_epoch,
)

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

# Final Metrics
print("\nFinal Evaluation Metrics:")
print("-" * 30)
print(f"ID RMSE:         {id_rmse:.4f}")
print(f"OOD RMSE:        {ood_rmse:.4f}")
print(f"Max OOD Error:   {ood_max_error:.4f}")
print(f"Peak OOD Epi:    {peak_ood_epi:.4f}")
print(f"Mean ID Epi:     {mean_id_epi:.4f}")
print(f"ID EAR:          {id_ear:.4f}")
print(f"OOD EAR:         {ood_ear:.4f}")
print(f"ID Cov (2σ):     {id_coverage:.1f}%")
print(f"OOD Cov (2σ):    {ood_coverage:.1f}%")
print(f"ID NLL:          {id_nll:.4f}")
print(f"OOD NLL:         {ood_nll:.4f}")
print("-" * 30)

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