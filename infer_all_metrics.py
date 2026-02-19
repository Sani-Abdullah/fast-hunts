import os
import json
import time
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import config

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 11,
    "figure.titleweight": "bold",
})

# ============================================================
# Utils
# ============================================================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_float01(path):
    img = np.array(Image.open(path).convert("F"), dtype=np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return np.clip(img, 0.0, 1.0)

def psnr_np(gt, pred, data_range=1.0, eps=1e-12):
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)
    mse = np.mean((gt - pred) ** 2, dtype=np.float64)
    mse = max(float(mse), eps)
    return float(10.0 * np.log10((data_range * data_range) / mse))

def _gaussian_kernel_1d(ksize=11, sigma=1.5):
    ax = np.arange(ksize, dtype=np.float64) - (ksize - 1) / 2.0
    k = np.exp(-(ax * ax) / (2.0 * sigma * sigma))
    k /= np.sum(k)
    return k

def _conv2d_separable(img, k1d):
    H, W = img.shape
    pad = len(k1d) // 2
    tmp = np.pad(img, ((0, 0), (pad, pad)), mode="reflect")
    out_h = np.zeros((H, W), dtype=np.float64)
    for x in range(W):
        out_h[:, x] = np.sum(tmp[:, x:x + len(k1d)] * k1d[None, :], axis=1)
    tmp2 = np.pad(out_h, ((pad, pad), (0, 0)), mode="reflect")
    out = np.zeros((H, W), dtype=np.float64)
    for y in range(H):
        out[y, :] = np.sum(tmp2[y:y + len(k1d), :] * k1d[:, None], axis=0)
    return out

def ssim_np(gt, pred, data_range=1.0, ksize=11, sigma=1.5, eps=1e-12):
    gt = gt.astype(np.float64)
    pred = pred.astype(np.float64)
    k = _gaussian_kernel_1d(ksize=ksize, sigma=sigma)
    mu_x = _conv2d_separable(gt, k)
    mu_y = _conv2d_separable(pred, k)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = _conv2d_separable(gt * gt, k) - mu_x2
    sigma_y2 = _conv2d_separable(pred * pred, k) - mu_y2
    sigma_xy = _conv2d_separable(gt * pred, k) - mu_xy
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    num = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return float(np.mean(num / (den + eps)))

def read_test_ids(checkpoints_dir, data_root):
    p1 = os.path.join(checkpoints_dir, "split_test.json")
    if os.path.isfile(p1):
        with open(p1, "r") as f:
            return list(json.load(f)["test"])
    p2 = os.path.join(checkpoints_dir, "split.json")
    if os.path.isfile(p2):
        with open(p2, "r") as f:
            return list(json.load(f)["test"])
    test_ids_cfg = list(getattr(config, "TEST_SET", []))
    valid = [sid for sid in test_ids_cfg if os.path.isdir(os.path.join(data_root, sid))]
    if len(valid) == 0:
        raise RuntimeError("Cannot resolve test ids.")
    return valid

def mean_ci95(values):
    """
    95% CI using normal approximation: mean ± 1.96 * SEM.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return None, None
    m = float(np.mean(v))
    if v.size == 1:
        return m, 0.0
    sem = float(np.std(v, ddof=1) / np.sqrt(v.size))
    return m, float(1.96 * sem)

def std_ddof1(values):
    v = np.asarray(values, dtype=np.float64)
    if v.size <= 1:
        return 0.0
    return float(np.std(v, ddof=1))

# ============================================================
# Operators for LR-metrics
# ============================================================
def downsample_avg(x, K):
    return F.avg_pool2d(x, kernel_size=K, stride=K)

def circular_warp_grid_sample(x, dx, dy):
    _, _, H, W = x.shape
    device = x.device
    dtype = x.dtype
    xt = x.repeat(1, 1, 3, 3)
    yy = torch.linspace(0, H - 1, H, device=device, dtype=dtype)
    xx = torch.linspace(0, W - 1, W, device=device, dtype=dtype)
    Y, X = torch.meshgrid(yy, xx, indexing="ij")
    Xc = X + W
    Yc = Y + H
    Xs = Xc + dx
    Ys = Yc + dy
    Wt = 3 * W
    Ht = 3 * H
    gx = (2.0 * Xs / (Wt - 1.0)) - 1.0
    gy = (2.0 * Ys / (Ht - 1.0)) - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    return F.grid_sample(xt, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

def dt_x_to_lr(pred_hr_t, dx_dy_hr, K):
    """
    pred_hr_t: torch (1,1,H,W)
    dx_dy_hr: list of 3 tuples (dx_tensor, dy_tensor) in HR pixels
    """
    p0 = downsample_avg(pred_hr_t, K)[0, 0].detach().cpu().numpy().astype(np.float32)
    outs = [np.clip(p0, 0.0, 1.0)]
    for i in range(3):
        dx, dy = dx_dy_hr[i]
        w = circular_warp_grid_sample(pred_hr_t, dx, dy)
        pi = downsample_avg(w, K)[0, 0].detach().cpu().numpy().astype(np.float32)
        outs.append(np.clip(pi, 0.0, 1.0))
    return outs

def avg_lr_metrics(lr_gts, lr_preds):
    psnrs, ssims = [], []
    for i in range(4):
        psnrs.append(psnr_np(lr_gts[i], lr_preds[i]))
        ssims.append(ssim_np(lr_gts[i], lr_preds[i]))
    return float(np.mean(psnrs)), float(np.mean(ssims))

# ============================================================
# Models (UNet + HUN / UNN)
# ============================================================
def center_crop(enc, target):
    _, _, H, W = target.shape
    _, _, h, w = enc.shape
    y0 = (h - H) // 2
    x0 = (w - W) // 2
    return enc[:, :, y0:y0 + H, x0:x0 + W]

class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=1, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x, hr_hw):
        H_hr, W_hr = hr_hw
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, center_crop(e3, d3)], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, center_crop(e2, d2)], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, center_crop(e1, d1)], dim=1))
        out_lr = self.out(d1)
        return F.interpolate(out_lr, size=(H_hr, W_hr), mode="bilinear", align_corners=False)

def data_fidelity_and_grad(x, lrs, t_pred, K):
    B = x.size(0)
    grad = torch.zeros_like(x)
    loss_acc = 0.0

    pred0 = downsample_avg(x, K)
    diff0 = pred0 - lrs[:, 0:1]
    loss_acc += (diff0 ** 2).mean()
    grad += (F.interpolate(diff0, scale_factor=K, mode="nearest") / float(K * K))

    for i in range(3):
        dx = t_pred[:, i, 0]
        dy = t_pred[:, i, 1]

        preds = []
        for b in range(B):
            xb = x[b:b + 1]
            wb = circular_warp_grid_sample(xb, dx[b], dy[b])
            preds.append(downsample_avg(wb, K))
        pred_i = torch.cat(preds, dim=0)

        diff = pred_i - lrs[:, (i + 1):(i + 2)]
        loss_acc += (diff ** 2).mean()

        diff_up = F.interpolate(diff, scale_factor=K, mode="nearest") / float(K * K)
        backs = []
        for b in range(B):
            db = diff_up[b:b + 1]
            bb = circular_warp_grid_sample(db, -dx[b], -dy[b])
            backs.append(bb)
        grad += torch.cat(backs, dim=0)

    return (loss_acc / 4.0), (grad / 4.0)

class TranslationNet(nn.Module):
    def __init__(self, max_shift_hr=40.0):
        super().__init__()
        self.max_shift_hr = float(max_shift_hr)
        self.backbone = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, 6)

    def forward(self, lrs):
        f = self.backbone(lrs).view(lrs.size(0), -1)
        z = torch.tanh(self.fc(f)) * self.max_shift_hr
        return z.view(lrs.size(0), 3, 2)

class UnfoldedSR(nn.Module):
    def __init__(self, K_steps=8, max_shift_hr=40.0):
        super().__init__()
        self.K_steps = int(K_steps)
        self.tnet = TranslationNet(max_shift_hr=max_shift_hr)

        init = 0.1
        s_init = np.log(np.exp(init) - 1.0 + 1e-8)
        self.s = nn.Parameter(torch.full((self.K_steps,), float(s_init), dtype=torch.float32))

    def alphas(self):
        return F.softplus(self.s)

    def forward(self, lrs, hr_hw):
        B, _, h, w = lrs.shape
        H_hr, W_hr = hr_hw
        K = H_hr // h

        t_pred = self.tnet(lrs)
        x = F.interpolate(lrs[:, 0:1], size=(H_hr, W_hr), mode="bicubic", align_corners=False)
        x = torch.clamp(x, 0.0, 1.0)

        alphas = self.alphas()
        for k in range(self.K_steps):
            _, grad = data_fidelity_and_grad(x, lrs, t_pred, K=K)
            x = torch.clamp(x - alphas[k] * grad, 0.0, 1.0)

        return x, t_pred

def load_unet(device, ckpt_path):
    m = UNet().to(device)
    m.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
    m.eval()
    return m

def load_unn(device, ckpt_path, fallback_K_steps=8, fallback_max_shift_hr=40.0):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    K_steps = int(cfg.get("K_steps", fallback_K_steps))
    max_shift_hr = float(cfg.get("max_shift_hr", fallback_max_shift_hr))
    m = UnfoldedSR(K_steps=K_steps, max_shift_hr=max_shift_hr).to(device)
    m.load_state_dict(ckpt["model_state"], strict=True)
    m.eval()
    return m

@torch.no_grad()
def forward_unet(model, lrs_t, hr_hw):
    return model(lrs_t, hr_hw)

@torch.no_grad()
def forward_unn(model, lrs_t, hr_hw):
    pred, _t = model(lrs_t, hr_hw)
    return pred

@torch.no_grad()
def forward_unn_with_t(model, lrs_t, hr_hw):
    pred, t = model(lrs_t, hr_hw)
    return pred, t

# ============================================================
# Timing helpers
# ============================================================
def cuda_sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize()

def time_forward_ms(fn, device, repeats, warmup):
    """
    Runs warmup passes (not timed), then repeats timed runs.
    Returns:
      - times_ms: list length=repeats (each timing for one forward call)
      - last_out: output from last timed run
    """
    for _ in range(warmup):
        _ = fn()
    cuda_sync_if_needed(device)

    times = []
    out = None
    for _ in range(repeats):
        cuda_sync_if_needed(device)
        t0 = time.perf_counter()
        out = fn()
        cuda_sync_if_needed(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return times, out

def plot_inference_time_runs(out_png, run_means_ms, run_ci_ms, title):
    x = np.arange(1, len(run_means_ms) + 1)
    plt.figure(figsize=(7, 4))
    plt.errorbar(x, run_means_ms, yerr=run_ci_ms, fmt='-o')
    plt.xlabel("Run index")
    plt.ylabel("Inference time (ms)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def plot_best_model_time_20runs(out_png, times_ms, title):
    # plot raw times with mean ± 95%CI band (computed over the 20 runs)
    x = np.arange(1, len(times_ms) + 1)
    m, ci = mean_ci95(times_ms)
    plt.figure(figsize=(7, 4))
    plt.plot(x, times_ms, marker="o")
    plt.axhline(m, linestyle="--")
    plt.fill_between(x, m - ci, m + ci, alpha=0.2)
    plt.xlabel("Run index")
    plt.ylabel("Inference time (ms)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

# ============================================================
# Metric summary plots (mean ± CI, and STD shown)
# ============================================================
def plot_metric_bar(out_png, title, ylabel, labels, means, ci95, stds, colors=None):
    """
    2-bar plot (UNet, HUN) with CI error bars and value labels + STD annotated.
    - Different colors for bars.
    - Value labels placed above CI whiskers (no overlap).
    """
    x = np.arange(len(labels))
    plt.figure(figsize=(5.2, 4.2))
    if colors is None:
        colors = ["tab:blue", "tab:orange"]

    bars = plt.bar(x, means, yerr=ci95, capsize=6, color=colors[:len(labels)])
    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.title(title)

    # Place value labels above whiskers
    for i in range(len(labels)):
        val = means[i]
        whisk = ci95[i] if ci95 is not None else 0.0
        y = val + whisk + 0.02 * (abs(val) + 1e-6)
        plt.text(x[i], y, f"{val:.3f}" if ylabel.lower() == "ssim" else f"{val:.2f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
        # annotate std (smaller)
        plt.text(x[i], y + 0.04 * (abs(val) + 1e-6), f"std={stds[i]:.3f}",
                 ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

# ============================================================
# Qualitative grid (LR0..LR3, UNet, HUN)
# ============================================================
def save_grid_figure(
    out_png,
    data_root,
    ids,
    per_image_dict,
    show_n=5,
    cmap="inferno"
):
    """
    cols: LR0 LR1 LR2 LR3 UNet HUN
    rows: samples
    Titles for UNet/HUN show LR-avg PSNR/SSIM for that sample.
    """
    sel = ids[:max(0, int(show_n))]
    if len(sel) == 0:
        return

    nrows = len(sel)
    ncols = 6
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2.1*ncols, 2.1*nrows))
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, sid in enumerate(sel):
        sdir = os.path.join(data_root, sid)
        lrs = [load_float01(os.path.join(sdir, f"lr{i}_{sid}.png")) for i in range(4)]

        # UNet metrics
        pu = per_image_dict[sid].get("UNet_lr_psnr", None)
        su = per_image_dict[sid].get("UNet_lr_ssim", None)

        # HUN metrics (new keys) with backward-compatible fallback to old UNN keys
        pn = per_image_dict[sid].get("HUN_lr_psnr", per_image_dict[sid].get("UNN_lr_psnr", None))
        sn = per_image_dict[sid].get("HUN_lr_ssim", per_image_dict[sid].get("UNN_lr_ssim", None))

        # HR preds for display (new keys)
        unet_hr = per_image_dict[sid].get("UNet_pred_hr", None)
        hun_hr  = per_image_dict[sid].get("HUN_pred_hr", per_image_dict[sid].get("UNN_pred_hr", None))

        for c in range(4):
            ax = axes[r, c]
            ax.imshow(lrs[c], cmap=cmap, vmin=0.0, vmax=1.0)
            ax.set_title(f"LR{c}", fontsize=10)
            ax.axis("off")

        axu = axes[r, 4]
        if unet_hr is not None:
            axu.imshow(unet_hr, cmap=cmap, vmin=0.0, vmax=1.0)
        else:
            axu.text(0.5, 0.5, "UNet\n(no pred saved)", ha="center", va="center")
        if pu is not None and su is not None:
            axu.set_title(f"UNet\nPSNR {pu:.2f} SSIM {su:.3f}", fontsize=9)
        else:
            axu.set_title("UNet", fontsize=9)
        axu.axis("off")

        axh = axes[r, 5]
        if hun_hr is not None:
            axh.imshow(hun_hr, cmap=cmap, vmin=0.0, vmax=1.0)
        else:
            axh.text(0.5, 0.5, "HUN\n(no pred saved)", ha="center", va="center")
        if pn is not None and sn is not None:
            axh.set_title(f"HUN\nPSNR {pn:.2f} SSIM {sn:.3f}", fontsize=9)
        else:
            axh.set_title("HUN", fontsize=9)
        axh.axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close(fig)

def strip_non_json(res: dict) -> dict:
    """
    Remove any non-JSON-serializable fields (e.g., numpy arrays) from res.
    Keeps metrics; grid images are already saved to disk.
    """
    res2 = {
        "summary": res.get("summary", {}),
        "per_image": []
    }
    for d in res.get("per_image", []):
        dd = dict(d)
        dd.pop("UNet_pred_hr", None)
        dd.pop("HUN_pred_hr", None)
        dd.pop("UNN_pred_hr", None)  # backward compat
        res2["per_image"].append(dd)
    return res2

# ============================================================
# Fold evaluation (LR metrics + timing + plots + grid)
# ============================================================
def evaluate_fold(
    fold_tag: str,
    device,
    data_root: str,
    test_ids: list,
    unet_model: nn.Module,
    unn_model: nn.Module,
    repeats: int,
    warmup: int,
    out_root: str,
    max_n: int = -1,
    grid_n: int = 5,
    save_preds_for_grid: bool = True,
    grid_cmap: str = "inferno",
):
    if max_n > 0:
        test_ids = test_ids[:max_n]

    per_image = []
    per_image_map = {}

    # times_runs_model[r] = list of times across images for run r
    times_runs_unet = [[] for _ in range(repeats)]
    times_runs_unn  = [[] for _ in range(repeats)]

    for sid in test_ids:
        sdir = os.path.join(data_root, sid)

        lr_gts = [load_float01(os.path.join(sdir, f"lr{i}_{sid}.png")) for i in range(4)]
        lrs_t = torch.from_numpy(np.stack(lr_gts, axis=0)).unsqueeze(0).to(device=device, dtype=torch.float32)

        h, w = lrs_t.shape[-2:]
        H_hr, W_hr = 2 * h, 2 * w
        K = H_hr // h

        meta_path = os.path.join(sdir, f"meta_unn_{sid}.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Missing translations file: {meta_path}")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        t_list = meta.get("predicted_translations_hr", None)
        if t_list is None or len(t_list) != 3:
            raise RuntimeError(f"Bad predicted_translations_hr in {meta_path}")

        dx_dy_hr = []
        for (dx, dy) in t_list:
            dx_dy_hr.append((
                torch.tensor(float(dx), device=device, dtype=torch.float32),
                torch.tensor(float(dy), device=device, dtype=torch.float32),
            ))

        # UNet timing + pred (last timed output used for metrics/grid)
        def fn_u():
            return forward_unet(unet_model, lrs_t, (H_hr, W_hr))

        tms_u, pred_u_t = time_forward_ms(fn_u, device, repeats=repeats, warmup=warmup)
        for r in range(repeats):
            times_runs_unet[r].append(float(tms_u[r]))

        # HUN timing + pred
        def fn_n():
            return forward_unn(unn_model, lrs_t, (H_hr, W_hr))

        tms_n, pred_n_t = time_forward_ms(fn_n, device, repeats=repeats, warmup=warmup)
        for r in range(repeats):
            times_runs_unn[r].append(float(tms_n[r]))

        # LR metrics via D*T(x_pred)
        lr_preds_u = dt_x_to_lr(pred_u_t, dx_dy_hr, K=K)
        lr_preds_n = dt_x_to_lr(pred_n_t, dx_dy_hr, K=K)
        pu, su = avg_lr_metrics(lr_gts, lr_preds_u)
        pn, sn = avg_lr_metrics(lr_gts, lr_preds_n)

        # UNet-on-HUN reference (reconstruction-domain)
        unet_hr_np = np.clip(pred_u_t[0, 0].detach().cpu().numpy().astype(np.float32), 0.0, 1.0)
        hun_hr_np  = np.clip(pred_n_t[0, 0].detach().cpu().numpy().astype(np.float32), 0.0, 1.0)
        p_u_on_h = psnr_np(hun_hr_np, unet_hr_np)
        s_u_on_h = ssim_np(hun_hr_np, unet_hr_np)

        item = {
            "id": sid,
            "UNet_lr_psnr": float(pu),
            "UNet_lr_ssim": float(su),
            "HUN_lr_psnr": float(pn),
            "HUN_lr_ssim": float(sn),
            "UNet_on_HUN_psnr": float(p_u_on_h),
            "UNet_on_HUN_ssim": float(s_u_on_h),
            "UNet_infer_ms_mean_over_repeats": float(np.mean(tms_u)),
            "HUN_infer_ms_mean_over_repeats": float(np.mean(tms_n)),
        }

        if save_preds_for_grid:
            item["UNet_pred_hr"] = unet_hr_np
            item["HUN_pred_hr"]  = hun_hr_np

        per_image.append(item)
        per_image_map[sid] = item

    # Extract arrays for stats (across images)
    u_psnr = [d["UNet_lr_psnr"] for d in per_image]
    u_ssim = [d["UNet_lr_ssim"] for d in per_image]
    n_psnr = [d["HUN_lr_psnr"] for d in per_image]
    n_ssim = [d["HUN_lr_ssim"] for d in per_image]

    u_on_h_psnr = [d["UNet_on_HUN_psnr"] for d in per_image]
    u_on_h_ssim = [d["UNet_on_HUN_ssim"] for d in per_image]

    u_img_means = [d["UNet_infer_ms_mean_over_repeats"] for d in per_image]
    n_img_means = [d["HUN_infer_ms_mean_over_repeats"] for d in per_image]

    u_psnr_m, u_psnr_ci = mean_ci95(u_psnr)
    u_ssim_m, u_ssim_ci = mean_ci95(u_ssim)
    n_psnr_m, n_psnr_ci = mean_ci95(n_psnr)
    n_ssim_m, n_ssim_ci = mean_ci95(n_ssim)

    u_on_h_psnr_m, u_on_h_psnr_ci = mean_ci95(u_on_h_psnr)
    u_on_h_ssim_m, u_on_h_ssim_ci = mean_ci95(u_on_h_ssim)

    u_time_m, u_time_ci = mean_ci95(u_img_means)
    n_time_m, n_time_ci = mean_ci95(n_img_means)

    u_psnr_std = std_ddof1(u_psnr)
    u_ssim_std = std_ddof1(u_ssim)
    n_psnr_std = std_ddof1(n_psnr)
    n_ssim_std = std_ddof1(n_ssim)

    u_on_h_psnr_std = std_ddof1(u_on_h_psnr)
    u_on_h_ssim_std = std_ddof1(u_on_h_ssim)

    u_time_std = std_ddof1(u_img_means)
    n_time_std = std_ddof1(n_img_means)

    # Per-run time mean ± CI across images
    u_run_means, u_run_cis = [], []
    n_run_means, n_run_cis = [], []
    for r in range(repeats):
        m, ci = mean_ci95(times_runs_unet[r]); u_run_means.append(m); u_run_cis.append(ci)
        m, ci = mean_ci95(times_runs_unn[r]);  n_run_means.append(m); n_run_cis.append(ci)

    # Output folders
    fold_out_dir = os.path.join(out_root, fold_tag)
    plots_dir = os.path.join(fold_out_dir, "plots")
    grids_dir = os.path.join(fold_out_dir, "grids")
    ensure_dir(plots_dir)
    ensure_dir(grids_dir)

    # 1) Per-run inference time plots
    plot_inference_time_runs(
        out_png=os.path.join(plots_dir, f"infer_time_unet_{fold_tag}.png"),
        run_means_ms=u_run_means, run_ci_ms=u_run_cis,
        title=f"UNet inference time per run ({fold_tag})"
    )
    plot_inference_time_runs(
        out_png=os.path.join(plots_dir, f"infer_time_hun_{fold_tag}.png"),
        run_means_ms=n_run_means, run_ci_ms=n_run_cis,
        title=f"HUN inference time per run ({fold_tag})"
    )

    # 2) Metric bar plots (mean ± CI, std annotated, different colors)
    plot_metric_bar(
        out_png=os.path.join(plots_dir, f"psnr_ci_std_{fold_tag}.png"),
        title=f"LR-PSNR (mean ± 95% CI) ({fold_tag})",
        ylabel="PSNR (dB)",
        labels=["UNet", "HUN"],
        means=[u_psnr_m, n_psnr_m],
        ci95=[u_psnr_ci, n_psnr_ci],
        stds=[u_psnr_std, n_psnr_std],
        colors=["tab:blue", "tab:orange"],
    )
    plot_metric_bar(
        out_png=os.path.join(plots_dir, f"ssim_ci_std_{fold_tag}.png"),
        title=f"LR-SSIM (mean ± 95% CI) ({fold_tag})",
        ylabel="SSIM",
        labels=["UNet", "HUN"],
        means=[u_ssim_m, n_ssim_m],
        ci95=[u_ssim_ci, n_ssim_ci],
        stds=[u_ssim_std, n_ssim_std],
        colors=["tab:blue", "tab:orange"],
    )
    plot_metric_bar(
        out_png=os.path.join(plots_dir, f"infer_ms_ci_std_{fold_tag}.png"),
        title=f"Inference time (mean ± 95% CI) ({fold_tag})",
        ylabel="Time (ms)",
        labels=["UNet", "HUN"],
        means=[u_time_m, n_time_m],
        ci95=[u_time_ci, n_time_ci],
        stds=[u_time_std, n_time_std],
        colors=["tab:blue", "tab:orange"],
    )
    # UNet-on-HUN reference plot
    plot_metric_bar(
        out_png=os.path.join(plots_dir, f"unet_on_hun_psnr_ssim_{fold_tag}.png"),
        title=f"UNet vs HUN (reconstruction-domain) ({fold_tag})",
        ylabel="PSNR (dB)",
        labels=["UNet-on-HUN", " "],
        means=[u_on_h_psnr_m, u_on_h_psnr_m],  # dummy second to keep 2 bars style
        ci95=[u_on_h_psnr_ci, 0.0],
        stds=[u_on_h_psnr_std, 0.0],
        colors=["tab:green", "white"],
    )
    # (also save SSIM for UNet-on-HUN)
    plot_metric_bar(
        out_png=os.path.join(plots_dir, f"unet_on_hun_ssim_{fold_tag}.png"),
        title=f"UNet vs HUN (reconstruction-domain) ({fold_tag})",
        ylabel="SSIM",
        labels=["UNet-on-HUN", " "],
        means=[u_on_h_ssim_m, u_on_h_ssim_m],
        ci95=[u_on_h_ssim_ci, 0.0],
        stds=[u_on_h_ssim_std, 0.0],
        colors=["tab:green", "white"],
    )

    # 3) Qualitative grid (like your attached)
    grid_path = os.path.join(grids_dir, f"grid_{fold_tag}.png")
    save_grid_figure(
        out_png=grid_path,
        data_root=data_root,
        ids=test_ids,
        per_image_dict=per_image_map,
        show_n=grid_n,
        cmap=grid_cmap
    )

    summary = {
        "fold": fold_tag,
        "num_test_images": int(len(per_image)),
        "repeats": int(repeats),
        "warmup": int(warmup),

        "UNet_lr_psnr_mean": u_psnr_m,
        "UNet_lr_psnr_ci95": u_psnr_ci,
        "UNet_lr_psnr_std": u_psnr_std,

        "UNet_lr_ssim_mean": u_ssim_m,
        "UNet_lr_ssim_ci95": u_ssim_ci,
        "UNet_lr_ssim_std": u_ssim_std,

        "UNet_infer_ms_mean": u_time_m,
        "UNet_infer_ms_ci95": u_time_ci,
        "UNet_infer_ms_std": u_time_std,

        "HUN_lr_psnr_mean": n_psnr_m,
        "HUN_lr_psnr_ci95": n_psnr_ci,
        "HUN_lr_psnr_std": n_psnr_std,

        "HUN_lr_ssim_mean": n_ssim_m,
        "HUN_lr_ssim_ci95": n_ssim_ci,
        "HUN_lr_ssim_std": n_ssim_std,

        "HUN_infer_ms_mean": n_time_m,
        "HUN_infer_ms_ci95": n_time_ci,
        "HUN_infer_ms_std": n_time_std,

        "UNet_on_HUN_psnr_mean": u_on_h_psnr_m,
        "UNet_on_HUN_psnr_ci95": u_on_h_psnr_ci,
        "UNet_on_HUN_psnr_std": u_on_h_psnr_std,
        "UNet_on_HUN_ssim_mean": u_on_h_ssim_m,
        "UNet_on_HUN_ssim_ci95": u_on_h_ssim_ci,
        "UNet_on_HUN_ssim_std": u_on_h_ssim_std,

        "UNet_infer_run_means_ms": u_run_means,
        "UNet_infer_run_ci95_ms": u_run_cis,
        "HUN_infer_run_means_ms": n_run_means,
        "HUN_infer_run_ci95_ms": n_run_cis,

        "artifacts": {
            "plots_dir": plots_dir,
            "grids_dir": grids_dir,
            "grid_png": grid_path,
        }
    }

    return {"summary": summary, "per_image": per_image}

# ============================================================
# Across-fold average (ensemble per image) utilities
# ============================================================
@torch.no_grad()
def predict_unet_hr_tensor(model, lrs_t, hr_hw):
    return forward_unet(model, lrs_t, hr_hw)

@torch.no_grad()
def predict_hun_hr_tensor(model, lrs_t, hr_hw):
    return forward_unn(model, lrs_t, hr_hw)

def evaluate_across_folds_average(
    device,
    data_root: str,
    test_ids: list,
    unet_models: list,
    hun_models: list,
    out_dir: str,
    grid_n: int = 5,
    grid_cmap: str = "inferno",
):
    """
    Ensemble per-image:
      pred(image) = average over folds/models for that image.
    Metrics:
      - LR metrics using meta_unn_<sid>.json translations (same evaluation protocol as before).
      - UNet-on-HUN reconstruction-domain PSNR/SSIM using ensembled HR outputs.
    """
    per_image = []
    per_image_map = {}

    for sid in test_ids:
        sdir = os.path.join(data_root, sid)

        lr_gts = [load_float01(os.path.join(sdir, f"lr{i}_{sid}.png")) for i in range(4)]
        lrs_t = torch.from_numpy(np.stack(lr_gts, axis=0)).unsqueeze(0).to(device=device, dtype=torch.float32)

        h, w = lrs_t.shape[-2:]
        H_hr, W_hr = 2 * h, 2 * w
        K = H_hr // h

        meta_path = os.path.join(sdir, f"meta_unn_{sid}.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Missing translations file: {meta_path}")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        t_list = meta.get("predicted_translations_hr", None)
        if t_list is None or len(t_list) != 3:
            raise RuntimeError(f"Bad predicted_translations_hr in {meta_path}")

        dx_dy_hr = []
        for (dx, dy) in t_list:
            dx_dy_hr.append((
                torch.tensor(float(dx), device=device, dtype=torch.float32),
                torch.tensor(float(dy), device=device, dtype=torch.float32),
            ))

        # Ensemble predictions over folds
        pu_list = []
        pn_list = []
        for m in unet_models:
            pu_list.append(predict_unet_hr_tensor(m, lrs_t, (H_hr, W_hr)))
        for m in hun_models:
            pn_list.append(predict_hun_hr_tensor(m, lrs_t, (H_hr, W_hr)))

        pred_u_t = torch.mean(torch.stack(pu_list, dim=0), dim=0).clamp(0, 1)
        pred_n_t = torch.mean(torch.stack(pn_list, dim=0), dim=0).clamp(0, 1)

        # LR metrics
        lr_preds_u = dt_x_to_lr(pred_u_t, dx_dy_hr, K=K)
        lr_preds_n = dt_x_to_lr(pred_n_t, dx_dy_hr, K=K)
        pu, su = avg_lr_metrics(lr_gts, lr_preds_u)
        pn, sn = avg_lr_metrics(lr_gts, lr_preds_n)

        # UNet-on-HUN reference (reconstruction-domain)
        unet_hr_np = np.clip(pred_u_t[0, 0].detach().cpu().numpy().astype(np.float32), 0.0, 1.0)
        hun_hr_np  = np.clip(pred_n_t[0, 0].detach().cpu().numpy().astype(np.float32), 0.0, 1.0)
        p_u_on_h = psnr_np(hun_hr_np, unet_hr_np)
        s_u_on_h = ssim_np(hun_hr_np, unet_hr_np)

        item = {
            "id": sid,
            "UNet_lr_psnr": float(pu),
            "UNet_lr_ssim": float(su),
            "HUN_lr_psnr": float(pn),
            "HUN_lr_ssim": float(sn),
            "UNet_on_HUN_psnr": float(p_u_on_h),
            "UNet_on_HUN_ssim": float(s_u_on_h),
            "UNet_pred_hr": unet_hr_np,
            "HUN_pred_hr": hun_hr_np,
        }
        per_image.append(item)
        per_image_map[sid] = item

    # Summaries across images (ensemble-per-image)
    u_psnr = [d["UNet_lr_psnr"] for d in per_image]
    u_ssim = [d["UNet_lr_ssim"] for d in per_image]
    n_psnr = [d["HUN_lr_psnr"] for d in per_image]
    n_ssim = [d["HUN_lr_ssim"] for d in per_image]
    u_on_h_psnr = [d["UNet_on_HUN_psnr"] for d in per_image]
    u_on_h_ssim = [d["UNet_on_HUN_ssim"] for d in per_image]

    u_psnr_m, u_psnr_ci = mean_ci95(u_psnr)
    u_ssim_m, u_ssim_ci = mean_ci95(u_ssim)
    n_psnr_m, n_psnr_ci = mean_ci95(n_psnr)
    n_ssim_m, n_ssim_ci = mean_ci95(n_ssim)
    u_on_h_psnr_m, u_on_h_psnr_ci = mean_ci95(u_on_h_psnr)
    u_on_h_ssim_m, u_on_h_ssim_ci = mean_ci95(u_on_h_ssim)

    summary = {
        "mode": "ACROSS_FOLDS_AVERAGE",
        "num_test_images": int(len(per_image)),

        "UNet_lr_psnr_mean": u_psnr_m,
        "UNet_lr_psnr_ci95": u_psnr_ci,
        "UNet_lr_psnr_std": std_ddof1(u_psnr),

        "UNet_lr_ssim_mean": u_ssim_m,
        "UNet_lr_ssim_ci95": u_ssim_ci,
        "UNet_lr_ssim_std": std_ddof1(u_ssim),

        "HUN_lr_psnr_mean": n_psnr_m,
        "HUN_lr_psnr_ci95": n_psnr_ci,
        "HUN_lr_psnr_std": std_ddof1(n_psnr),

        "HUN_lr_ssim_mean": n_ssim_m,
        "HUN_lr_ssim_ci95": n_ssim_ci,
        "HUN_lr_ssim_std": std_ddof1(n_ssim),

        "UNet_on_HUN_psnr_mean": u_on_h_psnr_m,
        "UNet_on_HUN_psnr_ci95": u_on_h_psnr_ci,
        "UNet_on_HUN_psnr_std": std_ddof1(u_on_h_psnr),

        "UNet_on_HUN_ssim_mean": u_on_h_ssim_m,
        "UNet_on_HUN_ssim_ci95": u_on_h_ssim_ci,
        "UNet_on_HUN_ssim_std": std_ddof1(u_on_h_ssim),
    }

    # Artifacts: grid + plots for ensemble
    ensemble_dir = os.path.join(out_dir, "across_folds_average")
    plots_dir = os.path.join(ensemble_dir, "plots")
    grids_dir = os.path.join(ensemble_dir, "grids")
    ensure_dir(plots_dir)
    ensure_dir(grids_dir)

    grid_path = os.path.join(grids_dir, "grid_across_folds_average.png")
    save_grid_figure(
        out_png=grid_path,
        data_root=data_root,
        ids=test_ids,
        per_image_dict=per_image_map,
        show_n=grid_n,
        cmap=grid_cmap
    )

    plot_metric_bar(
        out_png=os.path.join(plots_dir, "psnr_ci_std_across_folds_average.png"),
        title="LR-PSNR (mean ± 95% CI) (ACROSS_FOLDS_AVERAGE)",
        ylabel="PSNR (dB)",
        labels=["UNet", "HUN"],
        means=[summary["UNet_lr_psnr_mean"], summary["HUN_lr_psnr_mean"]],
        ci95=[summary["UNet_lr_psnr_ci95"], summary["HUN_lr_psnr_ci95"]],
        stds=[summary["UNet_lr_psnr_std"], summary["HUN_lr_psnr_std"]],
        colors=["tab:blue", "tab:orange"],
    )
    plot_metric_bar(
        out_png=os.path.join(plots_dir, "ssim_ci_std_across_folds_average.png"),
        title="LR-SSIM (mean ± 95% CI) (ACROSS_FOLDS_AVERAGE)",
        ylabel="SSIM",
        labels=["UNet", "HUN"],
        means=[summary["UNet_lr_ssim_mean"], summary["HUN_lr_ssim_mean"]],
        ci95=[summary["UNet_lr_ssim_ci95"], summary["HUN_lr_ssim_ci95"]],
        stds=[summary["UNet_lr_ssim_std"], summary["HUN_lr_ssim_std"]],
        colors=["tab:blue", "tab:orange"],
    )

    summary["artifacts"] = {
        "ensemble_dir": ensemble_dir,
        "plots_dir": plots_dir,
        "grids_dir": grids_dir,
        "grid_png": grid_path,
    }

    return {"summary": summary, "per_image": per_image}

# ============================================================
# Main
# ============================================================
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ensure_dir(args.out_dir)

    test_ids = read_test_ids(args.unet_ckpt_dir, args.data_root)
    if args.max_n > 0:
        test_ids = test_ids[:args.max_n]

    # Decide folds
    folds = []
    if not args.kfold:
        folds.append({"tag": "single", "unet_ckpt": args.unet_ckpt, "unn_ckpt": args.unn_ckpt})
    else:
        fu = os.path.join(args.unet_ckpt_dir, "folds")
        fn = os.path.join(args.unn_ckpt_dir, "folds")
        for i in range(1, args.num_folds + 1):
            u = os.path.join(fu, f"{'best' if args.use_best else 'final'}_{i}.pt")
            n = os.path.join(fn, f"{'best' if args.use_best else 'final'}_{i}.pt")
            if not os.path.isfile(u):
                raise FileNotFoundError(u)
            if not os.path.isfile(n):
                raise FileNotFoundError(n)
            folds.append({"tag": f"fold_{i}", "unet_ckpt": u, "unn_ckpt": n})

    all_fold_results = []
    unet_models_all = []
    hun_models_all = []

    # Evaluate each fold (as before)
    for f in folds:
        unet_model = load_unet(device, f["unet_ckpt"])
        hun_model  = load_unn(device, f["unn_ckpt"], args.hun_K_steps, args.hun_max_shift_hr)

        unet_models_all.append(unet_model)
        hun_models_all.append(hun_model)

        res = evaluate_fold(
            fold_tag=f["tag"],
            device=device,
            data_root=args.data_root,
            test_ids=test_ids,
            unet_model=unet_model,
            unn_model=hun_model,
            repeats=args.repeats,
            warmup=args.warmup,
            out_root=args.out_dir,
            max_n=-1,  # max_n already applied above
            grid_n=args.grid_n,
            save_preds_for_grid=True,
            grid_cmap=args.grid_cmap
        )

        # Save per-fold json (keep filename; overall summary will be *_folds)
        fold_out_dir = os.path.join(args.out_dir, f["tag"])
        ensure_dir(fold_out_dir)
        out_json = os.path.join(fold_out_dir, "infer_all_lr_metrics.json")
        with open(out_json, "w") as fp:
            json.dump(strip_non_json(res), fp, indent=2)
        print(f"Saved: {out_json}")

        all_fold_results.append(res)

    # ------------------------------------------------------------
    # Best-model inference time: run 20 runs on ONE model (use_best fold_1 if kfold)
    # ------------------------------------------------------------
    best_unet = unet_models_all[0]
    best_hun  = hun_models_all[0]

    # Use first test sample as representative
    if len(test_ids) > 0:
        sid0 = test_ids[0]
        sdir0 = os.path.join(args.data_root, sid0)
        lr0s = [load_float01(os.path.join(sdir0, f"lr{i}_{sid0}.png")) for i in range(4)]
        lrs0_t = torch.from_numpy(np.stack(lr0s, axis=0)).unsqueeze(0).to(device=device, dtype=torch.float32)
        h, w = lrs0_t.shape[-2:]
        hr_hw0 = (2 * h, 2 * w)

        def fn_best_u():
            return forward_unet(best_unet, lrs0_t, hr_hw0)

        def fn_best_h():
            return forward_unn(best_hun, lrs0_t, hr_hw0)

        tms_u, _ = time_forward_ms(fn_best_u, device, repeats=args.repeats, warmup=args.warmup)
        tms_h, _ = time_forward_ms(fn_best_h, device, repeats=args.repeats, warmup=args.warmup)

        best_u_mean, best_u_ci = mean_ci95(tms_u)
        best_h_mean, best_h_ci = mean_ci95(tms_h)
        best_u_std = std_ddof1(tms_u)
        best_h_std = std_ddof1(tms_h)

        best_t_dir = os.path.join(args.out_dir, "best_model_timing")
        ensure_dir(best_t_dir)
        plot_best_model_time_20runs(
            out_png=os.path.join(best_t_dir, "best_unet_time_20runs.png"),
            times_ms=tms_u,
            title="UNet inference time (best model) – 20 runs"
        )
        plot_best_model_time_20runs(
            out_png=os.path.join(best_t_dir, "best_hun_time_20runs.png"),
            times_ms=tms_h,
            title="HUN inference time (best model) – 20 runs"
        )
    else:
        best_u_mean = best_u_ci = best_u_std = None
        best_h_mean = best_h_ci = best_h_std = None
        best_t_dir = None

    # ------------------------------------------------------------
    # Overall summary across folds (fold means as samples)
    # Also include mean/std over folds for UNet-on-HUN.
    # ------------------------------------------------------------
    overall = {"kfold": bool(args.kfold), "num_folds": int(args.num_folds) if args.kfold else 1}

    if args.kfold:
        u_psnr_f = [r["summary"]["UNet_lr_psnr_mean"] for r in all_fold_results]
        u_ssim_f = [r["summary"]["UNet_lr_ssim_mean"] for r in all_fold_results]
        u_time_f = [r["summary"]["UNet_infer_ms_mean"] for r in all_fold_results]

        n_psnr_f = [r["summary"]["HUN_lr_psnr_mean"] for r in all_fold_results]
        n_ssim_f = [r["summary"]["HUN_lr_ssim_mean"] for r in all_fold_results]
        n_time_f = [r["summary"]["HUN_infer_ms_mean"] for r in all_fold_results]

        u_on_h_psnr_f = [r["summary"]["UNet_on_HUN_psnr_mean"] for r in all_fold_results]
        u_on_h_ssim_f = [r["summary"]["UNet_on_HUN_ssim_mean"] for r in all_fold_results]

        u_psnr_m, u_psnr_ci = mean_ci95(u_psnr_f)
        u_ssim_m, u_ssim_ci = mean_ci95(u_ssim_f)
        u_time_m, u_time_ci = mean_ci95(u_time_f)

        n_psnr_m, n_psnr_ci = mean_ci95(n_psnr_f)
        n_ssim_m, n_ssim_ci = mean_ci95(n_ssim_f)
        n_time_m, n_time_ci = mean_ci95(n_time_f)

        u_on_h_psnr_m, u_on_h_psnr_ci = mean_ci95(u_on_h_psnr_f)
        u_on_h_ssim_m, u_on_h_ssim_ci = mean_ci95(u_on_h_ssim_f)

        overall.update({
            "repeats": int(args.repeats),
            "warmup": int(args.warmup),

            "UNet_lr_psnr_mean_over_folds": u_psnr_m,
            "UNet_lr_psnr_ci95_over_folds": u_psnr_ci,
            "UNet_lr_psnr_std_over_folds": std_ddof1(u_psnr_f),

            "UNet_lr_ssim_mean_over_folds": u_ssim_m,
            "UNet_lr_ssim_ci95_over_folds": u_ssim_ci,
            "UNet_lr_ssim_std_over_folds": std_ddof1(u_ssim_f),

            "UNet_infer_ms_mean_over_folds": u_time_m,
            "UNet_infer_ms_ci95_over_folds": u_time_ci,
            "UNet_infer_ms_std_over_folds": std_ddof1(u_time_f),

            "HUN_lr_psnr_mean_over_folds": n_psnr_m,
            "HUN_lr_psnr_ci95_over_folds": n_psnr_ci,
            "HUN_lr_psnr_std_over_folds": std_ddof1(n_psnr_f),

            "HUN_lr_ssim_mean_over_folds": n_ssim_m,
            "HUN_lr_ssim_ci95_over_folds": n_ssim_ci,
            "HUN_lr_ssim_std_over_folds": std_ddof1(n_ssim_f),

            "HUN_infer_ms_mean_over_folds": n_time_m,
            "HUN_infer_ms_ci95_over_folds": n_time_ci,
            "HUN_infer_ms_std_over_folds": std_ddof1(n_time_f),

            "UNet_on_HUN_psnr_mean_over_folds": u_on_h_psnr_m,
            "UNet_on_HUN_psnr_ci95_over_folds": u_on_h_psnr_ci,
            "UNet_on_HUN_psnr_std_over_folds": std_ddof1(u_on_h_psnr_f),

            "UNet_on_HUN_ssim_mean_over_folds": u_on_h_ssim_m,
            "UNet_on_HUN_ssim_ci95_over_folds": u_on_h_ssim_ci,
            "UNet_on_HUN_ssim_std_over_folds": std_ddof1(u_on_h_ssim_f),

            # Best-model timing over 20 runs (single sample)
            "best_model_timing": {
                "sid_used": test_ids[0] if len(test_ids) > 0 else None,
                "UNet_infer_20runs_mean_ms": best_u_mean,
                "UNet_infer_20runs_ci95_ms": best_u_ci,
                "UNet_infer_20runs_std_ms": best_u_std,
                "HUN_infer_20runs_mean_ms": best_h_mean,
                "HUN_infer_20runs_ci95_ms": best_h_ci,
                "HUN_infer_20runs_std_ms": best_h_std,
                "artifacts_dir": best_t_dir,
            }
        })

        # Overall plots across folds
        overall_plots = os.path.join(args.out_dir, "overall_plots")
        ensure_dir(overall_plots)

        plot_metric_bar(
            out_png=os.path.join(overall_plots, "psnr_ci_std_over_folds.png"),
            title="LR-PSNR over folds (mean ± 95% CI)",
            ylabel="PSNR (dB)",
            labels=["UNet", "HUN"],
            means=[overall["UNet_lr_psnr_mean_over_folds"], overall["HUN_lr_psnr_mean_over_folds"]],
            ci95=[overall["UNet_lr_psnr_ci95_over_folds"], overall["HUN_lr_psnr_ci95_over_folds"]],
            stds=[overall["UNet_lr_psnr_std_over_folds"], overall["HUN_lr_psnr_std_over_folds"]],
            colors=["tab:blue", "tab:orange"],
        )
        plot_metric_bar(
            out_png=os.path.join(overall_plots, "ssim_ci_std_over_folds.png"),
            title="LR-SSIM over folds (mean ± 95% CI)",
            ylabel="SSIM",
            labels=["UNet", "HUN"],
            means=[overall["UNet_lr_ssim_mean_over_folds"], overall["HUN_lr_ssim_mean_over_folds"]],
            ci95=[overall["UNet_lr_ssim_ci95_over_folds"], overall["HUN_lr_ssim_ci95_over_folds"]],
            stds=[overall["UNet_lr_ssim_std_over_folds"], overall["HUN_lr_ssim_std_over_folds"]],
            colors=["tab:blue", "tab:orange"],
        )
        plot_metric_bar(
            out_png=os.path.join(overall_plots, "infer_ms_ci_std_over_folds.png"),
            title="Inference time over folds (mean ± 95% CI)",
            ylabel="Time (ms)",
            labels=["UNet", "HUN"],
            means=[overall["UNet_infer_ms_mean_over_folds"], overall["HUN_infer_ms_mean_over_folds"]],
            ci95=[overall["UNet_infer_ms_ci95_over_folds"], overall["HUN_infer_ms_ci95_over_folds"]],
            stds=[overall["UNet_infer_ms_std_over_folds"], overall["HUN_infer_ms_std_over_folds"]],
            colors=["tab:blue", "tab:orange"],
        )

        overall["overall_plots_dir"] = overall_plots

    # ------------------------------------------------------------
    # ACROSS_FOLDS_AVERAGE option (ensemble-per-image)
    # ------------------------------------------------------------
    across_folds_avg_res = None
    if args.kfold and args.across_folds_average:
        across_folds_avg_res = evaluate_across_folds_average(
            device=device,
            data_root=args.data_root,
            test_ids=test_ids,
            unet_models=unet_models_all,
            hun_models=hun_models_all,
            out_dir=args.out_dir,
            grid_n=args.grid_n,
            grid_cmap=args.grid_cmap,
        )
        # Save ensemble-per-image json
        ensemble_json_path = os.path.join(args.out_dir, "across_folds_average", "infer_all_lr_metrics_across_folds_average.json")
        with open(ensemble_json_path, "w") as fp:
            json.dump(strip_non_json(across_folds_avg_res), fp, indent=2)
        print(f"Saved: {ensemble_json_path}")

    summary_out = {
        "overall": overall,
        "fold_summaries": [r["summary"] for r in all_fold_results],
        "across_folds_average": across_folds_avg_res["summary"] if across_folds_avg_res is not None else None,
    }

    # JSON filename must include _folds
    out_path = os.path.join(args.out_dir, "infer_all_lr_metrics_summary_folds.json")
    with open(out_path, "w") as f:
        json.dump(summary_out, f, indent=2)
    print(f"Saved: {out_path}")

# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", default="./dataset/4c")
    ap.add_argument("--out_dir", default="./inference")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_n", type=int, default=-1)

    # Checkpoints base dirs (for reading split_test.json and folds/)
    ap.add_argument("--unet_ckpt_dir", default="./checkpoints/unet")
    ap.add_argument("--unn_ckpt_dir", default="./checkpoints/unn")

    # Single (non-kfold) checkpoints
    ap.add_argument("--unet_ckpt", default="./checkpoints/unet/best.pt")
    ap.add_argument("--unn_ckpt", default="./checkpoints/unn/best.pt")

    # HUN fallback if ckpt config missing
    ap.add_argument("--hun_K_steps", type=int, default=8)
    ap.add_argument("--hun_max_shift_hr", type=float, default=40.0)

    # K-fold options
    ap.add_argument("--kfold", action="store_true")
    ap.add_argument("--num_folds", type=int, default=5)
    ap.add_argument("--use_best", action="store_true")

    # NEW: ACROSS_FOLDS_AVERAGE option
    ap.add_argument("--across_folds_average", action="store_true")

    # Timing options
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)

    # Grid options
    ap.add_argument("--grid_n", type=int, default=5)
    ap.add_argument("--grid_cmap", default="inferno")

    args = ap.parse_args()
    main(args)
