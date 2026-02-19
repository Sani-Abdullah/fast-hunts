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

# ============================================================
# Utils
# ============================================================
def load_float01(path: str) -> np.ndarray:
    img = np.array(Image.open(path).convert("F"), dtype=np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return np.clip(img, 0.0, 1.0)

def mse_loss_np(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.mean((a - b) ** 2))

def psnr_np(gt, pred, data_range=1.0, eps=1e-12):
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

    ssim_map = num / (den + eps)
    return float(np.mean(ssim_map))

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def read_test_ids_from_checkpoints(checkpoints_dir, data_root):
    p1 = os.path.join(checkpoints_dir, "split_test.json")
    if os.path.isfile(p1):
        with open(p1, "r") as f:
            obj = json.load(f)
        return list(obj["test"])

    p2 = os.path.join(checkpoints_dir, "split.json")
    if os.path.isfile(p2):
        with open(p2, "r") as f:
            obj = json.load(f)
        return list(obj["test"])

    test_ids_cfg = list(getattr(config, "TEST_SET", []))
    valid = []
    for sid in test_ids_cfg:
        if os.path.isdir(os.path.join(data_root, sid)):
            valid.append(sid)
    if len(valid) == 0:
        raise RuntimeError("Could not determine test set. Ensure split_test.json exists or config.TEST_SET is valid.")
    return valid

# ============================================================
# UNN model (same as training)
# ============================================================
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

def downsample_avg(x, K):
    return F.avg_pool2d(x, kernel_size=K, stride=K)

def upsample_adjoint_avg(diff_lr, K):
    up = F.interpolate(diff_lr, scale_factor=K, mode="nearest")
    return up / float(K * K)

def data_fidelity_and_grad(x, lrs, t_pred, K):
    B = x.size(0)
    grad = torch.zeros_like(x)
    loss_acc = 0.0

    pred0 = downsample_avg(x, K)
    diff0 = pred0 - lrs[:, 0:1]
    loss0 = (diff0 ** 2).mean()
    loss_acc = loss_acc + loss0
    grad = grad + upsample_adjoint_avg(diff0, K)

    for i in range(3):
        dx = t_pred[:, i, 0]
        dy = t_pred[:, i, 1]

        preds = []
        for b in range(B):
            xb = x[b:b + 1]
            wb = circular_warp_grid_sample(xb, dx[b], dy[b])
            pb = downsample_avg(wb, K)
            preds.append(pb)
        pred_i = torch.cat(preds, dim=0)

        diff = pred_i - lrs[:, (i + 1):(i + 2)]
        loss_i = (diff ** 2).mean()
        loss_acc = loss_acc + loss_i

        diff_up = upsample_adjoint_avg(diff, K)

        backs = []
        for b in range(B):
            db = diff_up[b:b + 1]
            bb = circular_warp_grid_sample(db, -dx[b], -dy[b])
            backs.append(bb)
        back = torch.cat(backs, dim=0)

        grad = grad + back

    data_loss = loss_acc / 4.0
    grad = grad / 4.0
    return data_loss, grad

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

# ============================================================
# Inference for one checkpoint on test ids
# ============================================================
def run_infer_on_ids(model, device, data_root, test_ids):
    is_cuda = (device.type == "cuda")
    per_image = []
    preds_by_id = {}

    with torch.no_grad():
        for sid in test_ids:
            sdir = os.path.join(data_root, sid)

            # LRs
            lrs_np = []
            for i in range(4):
                p = os.path.join(sdir, f"lr{i}_{sid}.png")
                lr = load_float01(p)
                lrs_np.append(lr)
            lrs_np = np.stack(lrs_np, axis=0)  # (4,h,w)
            lrs_t = torch.from_numpy(lrs_np).unsqueeze(0).to(device=device, dtype=torch.float32)

            h, w = lrs_t.shape[-2:]
            H_hr, W_hr = 2 * h, 2 * w

            # GT (HRL preferred)
            hrl_path = os.path.join(sdir, f"HRL_{sid}.png")
            hr_path = os.path.join(sdir, f"hr_{sid}.png")
            if os.path.isfile(hrl_path):
                gt = load_float01(hrl_path).astype(np.float32)
                gt_used = "HRL"
            else:
                gt = load_float01(hr_path).astype(np.float32)
                gt_used = "HR"

            # ensure shape
            if gt.shape != (H_hr, W_hr):
                gt_t = torch.from_numpy(gt)[None, None].to(device=device, dtype=torch.float32)
                gt_t = F.interpolate(gt_t, size=(H_hr, W_hr), mode="bilinear", align_corners=False)[0, 0]
                gt = gt_t.detach().cpu().numpy().astype(np.float32)

            # predict (timed)
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred_t, _t_pred = model(lrs_t, (H_hr, W_hr))
            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            infer_ms = (t1 - t0) * 1000.0
            pred = np.clip(pred_t[0, 0].detach().cpu().numpy().astype(np.float32), 0.0, 1.0)

            per_image.append({
                "id": sid,
                "gt_used": gt_used,
                "inference_time_ms": float(infer_ms),
                "mse_loss": float(mse_loss_np(gt, pred)),
                "psnr": float(psnr_np(gt, pred)),
                "ssim": float(ssim_np(gt, pred)),
            })
            preds_by_id[sid] = pred

    def mean_of(key):
        return float(np.mean([d[key] for d in per_image])) if per_image else None

    summary = {
        "num_test_images": int(len(per_image)),
        "device": str(device),
        "avg_inference_time_ms": mean_of("inference_time_ms"),
        "avg_mse_loss": mean_of("mse_loss"),
        "avg_psnr": mean_of("psnr"),
        "avg_ssim": mean_of("ssim"),
    }

    return {"summary": summary, "per_image": per_image}, preds_by_id

def save_grid_figure(out_path, data_root, test_ids, preds_by_id, title_suffix=""):
    n = len(test_ids)
    fig, axes = plt.subplots(n, 6, figsize=(18, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for row, sid in enumerate(test_ids):
        sdir = os.path.join(data_root, sid)

        # LRs
        for i in range(4):
            lr = load_float01(os.path.join(sdir, f"lr{i}_{sid}.png"))
            axes[row, i].imshow(lr, cmap="inferno")
            axes[row, i].set_title(f"{sid} – LR{i}")
            axes[row, i].axis("off")

        # GT
        hrl_path = os.path.join(sdir, f"HRL_{sid}.png")
        hr_path = os.path.join(sdir, f"hr_{sid}.png")
        if os.path.isfile(hrl_path):
            gt = load_float01(hrl_path)
            gt_title = "GT (HRL)"
        else:
            gt = load_float01(hr_path)
            gt_title = "GT (HR)"

        axes[row, 4].imshow(gt, cmap="inferno")
        axes[row, 4].set_title(f"{sid} – {gt_title}")
        axes[row, 4].axis("off")

        pred = preds_by_id[sid]
        p = psnr_np(gt, pred)
        s = ssim_np(gt, pred)
        axes[row, 5].imshow(pred, cmap="inferno")
        axes[row, 5].set_title(f"{sid} – Pred{title_suffix}\nPSNR {p:.2f} SSIM {s:.3f}")
        axes[row, 5].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# Main
# ============================================================
def infer(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    base_out = os.path.join(args.out_dir, "unn")
    ensure_dir(base_out)

    test_ids = read_test_ids_from_checkpoints(args.checkpoints_dir, args.data_root)
    print(f"[UNN] Test set: {len(test_ids)} samples")

    folds_dir = os.path.join(args.checkpoints_dir, "folds")

    if not args.kfold:
        ckpt = torch.load(args.checkpoint, map_location=device)
        cfg = ckpt.get("config", {})

        K_steps = int(cfg.get("K_steps", args.K_steps))
        max_shift_hr = float(cfg.get("max_shift_hr", args.max_shift_hr))

        model = UnfoldedSR(K_steps=K_steps, max_shift_hr=max_shift_hr).to(device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval()

        metrics, preds = run_infer_on_ids(model, device, args.data_root, test_ids)
        metrics["summary"]["K_steps"] = int(K_steps)
        metrics["summary"]["max_shift_hr"] = float(max_shift_hr)
        metrics["summary"]["checkpoint"] = str(args.checkpoint)

        fig_path = os.path.join(base_out, "unn_test_results.png")
        save_grid_figure(fig_path, args.data_root, test_ids, preds)

        json_path = os.path.join(base_out, "metrics_unn.json")
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"[UNN] Saved: {fig_path}")
        print(f"[UNN] Saved: {json_path}")
        return

    # kfold
    num_folds = int(args.num_folds)
    preds_per_fold = []
    metrics_per_fold = []

    for i in range(1, num_folds + 1):
        fold_out = os.path.join(base_out, f"fold_{i}")
        ensure_dir(fold_out)

        ckpt_path = os.path.join(folds_dir, f"{'best' if args.use_best else 'final'}_{i}.pt")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = ckpt.get("config", {})

        K_steps = int(cfg.get("K_steps", args.K_steps))
        max_shift_hr = float(cfg.get("max_shift_hr", args.max_shift_hr))

        model = UnfoldedSR(K_steps=K_steps, max_shift_hr=max_shift_hr).to(device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval()

        metrics, preds = run_infer_on_ids(model, device, args.data_root, test_ids)
        metrics["summary"]["fold"] = int(i)
        metrics["summary"]["K_steps"] = int(K_steps)
        metrics["summary"]["max_shift_hr"] = float(max_shift_hr)
        metrics["summary"]["checkpoint"] = str(ckpt_path)

        fig_path = os.path.join(fold_out, "unn_test_results.png")
        save_grid_figure(fig_path, args.data_root, test_ids, preds, title_suffix=f" (fold {i})")

        json_path = os.path.join(fold_out, "metrics_unn.json")
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"[UNN][fold {i}] Saved: {fig_path}")
        print(f"[UNN][fold {i}] Saved: {json_path}")

        preds_per_fold.append(preds)
        metrics_per_fold.append(metrics)

    # ensemble mean of predictions
    ens_out = os.path.join(base_out, "ensemble")
    ensure_dir(ens_out)

    ens_preds = {}
    for sid in test_ids:
        stack = np.stack([preds_per_fold[k][sid] for k in range(num_folds)], axis=0)
        ens_preds[sid] = np.clip(np.mean(stack, axis=0).astype(np.float32), 0.0, 1.0)

    per_image = []
    for sid in test_ids:
        sdir = os.path.join(args.data_root, sid)
        hrl_path = os.path.join(sdir, f"HRL_{sid}.png")
        hr_path = os.path.join(sdir, f"hr_{sid}.png")
        if os.path.isfile(hrl_path):
            gt = load_float01(hrl_path).astype(np.float32)
            gt_used = "HRL"
        else:
            gt = load_float01(hr_path).astype(np.float32)
            gt_used = "HR"

        pred = ens_preds[sid]
        per_image.append({
            "id": sid,
            "gt_used": gt_used,
            "mse_loss": float(mse_loss_np(gt, pred)),
            "psnr": float(psnr_np(gt, pred)),
            "ssim": float(ssim_np(gt, pred)),
        })

    def mean_of(key):
        return float(np.mean([d[key] for d in per_image])) if per_image else None

    ens_metrics = {
        "summary": {
            "mode": "kfold_ensemble_mean_pred",
            "num_folds": int(num_folds),
            "num_test_images": int(len(test_ids)),
            "device": str(device),
            "avg_mse_loss": mean_of("mse_loss"),
            "avg_psnr": mean_of("psnr"),
            "avg_ssim": mean_of("ssim"),
        },
        "per_image": per_image,
        "per_fold_summaries": [m["summary"] for m in metrics_per_fold],
    }

    fig_path = os.path.join(ens_out, "unn_test_results.png")
    save_grid_figure(fig_path, args.data_root, test_ids, ens_preds, title_suffix=" (ensemble)")

    json_path = os.path.join(ens_out, "metrics_unn.json")
    with open(json_path, "w") as f:
        json.dump(ens_metrics, f, indent=2)

    print(f"[UNN][ensemble] Saved: {fig_path}")
    print(f"[UNN][ensemble] Saved: {json_path}")

# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./dataset/4c")
    ap.add_argument("--checkpoints_dir", default="./checkpoints/unn")
    ap.add_argument("--checkpoint", default="./checkpoints/unn/best.pt")
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--out_dir", default="./inference")

    # fallback if checkpoint config missing (single mode)
    ap.add_argument("--K_steps", type=int, default=8)
    ap.add_argument("--max_shift_hr", type=float, default=40.0)

    # K-fold
    ap.add_argument("--kfold", default=True)
    ap.add_argument("--num_folds", type=int, default=5)
    ap.add_argument("--use_best", action="store_true")

    args = ap.parse_args()
    infer(args)