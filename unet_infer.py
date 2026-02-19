import os
import json
import argparse
import time
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
def load_float01(path):
    img = np.array(Image.open(path).convert("F"), dtype=np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return np.clip(img, 0.0, 1.0)

def center_crop(enc, target):
    _, _, H, W = target.shape
    _, _, h, w = enc.shape
    y0 = (h - H) // 2
    x0 = (w - W) // 2
    return enc[:, :, y0:y0+H, x0:x0+W]

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
        out_h[:, x] = np.sum(tmp[:, x:x+len(k1d)] * k1d[None, :], axis=1)

    tmp2 = np.pad(out_h, ((pad, pad), (0, 0)), mode="reflect")
    out = np.zeros((H, W), dtype=np.float64)
    for y in range(H):
        out[y, :] = np.sum(tmp2[y:y+len(k1d), :] * k1d[:, None], axis=0)

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
    """
    Preferred: checkpoints_dir/split_test.json written by training.
    Fallback: checkpoints_dir/split.json (legacy).
    If both missing, uses config.TEST_SET and verifies existence.
    """
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

    # last resort: config.TEST_SET
    test_ids_cfg = list(getattr(config, "TEST_SET", []))
    valid = []
    for sid in test_ids_cfg:
        if os.path.isdir(os.path.join(data_root, sid)):
            valid.append(sid)
    if len(valid) == 0:
        raise RuntimeError("Could not determine test set. Ensure split_test.json exists or config.TEST_SET is valid.")
    return valid

# ============================================================
# U-Net (same as training)
# ============================================================
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
        out_hr = F.interpolate(out_lr, size=(H_hr, W_hr), mode="bilinear", align_corners=False)
        return out_hr

# ============================================================
# Core inference for one model checkpoint
# ============================================================
def run_infer_on_ids(model, device, data_root, test_ids, acc_tol=0.05):
    is_cuda = (device.type == "cuda")
    per_image = []
    preds_by_id = {}

    with torch.no_grad():
        for sid in test_ids:
            sdir = os.path.join(data_root, sid)

            # load LRs
            lrs = []
            for i in range(4):
                lr = load_float01(os.path.join(sdir, f"lr{i}_{sid}.png"))
                lrs.append(lr)
            lrs_t = torch.from_numpy(np.stack(lrs, axis=0)).unsqueeze(0).to(device=device, dtype=torch.float32)

            h, w = lrs_t.shape[-2:]
            H_hr, W_hr = 2 * h, 2 * w

            # load GT (HRL preferred)
            hrl_path = os.path.join(sdir, f"HRL_{sid}.png")
            hr_path = os.path.join(sdir, f"hr_{sid}.png")
            if os.path.isfile(hrl_path):
                gt = load_float01(hrl_path).astype(np.float32)
                gt_used = "HRL"
            else:
                gt = load_float01(hr_path).astype(np.float32)
                gt_used = "HR"

            # predict (timed)
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred_t = model(lrs_t, (H_hr, W_hr))
            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            infer_ms = (t1 - t0) * 1000.0
            pred = np.clip(pred_t[0, 0].cpu().numpy().astype(np.float32), 0.0, 1.0)

            mse = mse_loss_np(gt, pred)
            psnr = psnr_np(gt, pred, data_range=1.0)
            ssim = ssim_np(gt, pred, data_range=1.0)

            per_image.append({
                "id": sid,
                "gt_used": gt_used,
                "inference_time_ms": float(infer_ms),
                "mse_loss": float(mse),
                "psnr": float(psnr),
                "ssim": float(ssim),
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
# Main inference entry
# ============================================================
def infer(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # base output dir
    base_out = os.path.join(args.out_dir, "unet")
    ensure_dir(base_out)

    test_ids = read_test_ids_from_checkpoints(args.checkpoints_dir, args.data_root)
    print(f"[UNet] Test set: {len(test_ids)} samples")

    folds_dir = os.path.join(args.checkpoints_dir, "folds")

    # ---------- non-kfold ----------
    if not args.kfold:
        ckpt_path = args.checkpoint
        model = UNet().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        metrics, preds = run_infer_on_ids(model, device, args.data_root, test_ids, acc_tol=args.acc_tol)

        fig_path = os.path.join(base_out, "unet_test_results.png")
        save_grid_figure(fig_path, args.data_root, test_ids, preds)

        json_path = os.path.join(base_out, "metrics_unet.json")
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"[UNet] Saved: {fig_path}")
        print(f"[UNet] Saved: {json_path}")
        return

    # ---------- kfold: infer each fold + ensemble average ----------
    num_folds = int(args.num_folds)

    preds_per_fold = []
    metrics_per_fold = []

    for i in range(1, num_folds + 1):
        fold_out = os.path.join(base_out, f"fold_{i}")
        ensure_dir(fold_out)

        ckpt = os.path.join(folds_dir, f"{'best' if args.use_best else 'final'}_{i}.pt")
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

        model = UNet().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        metrics, preds = run_infer_on_ids(model, device, args.data_root, test_ids, acc_tol=args.acc_tol)
        metrics["summary"]["fold"] = int(i)
        metrics["summary"]["checkpoint"] = ckpt

        fig_path = os.path.join(fold_out, "unet_test_results.png")
        save_grid_figure(fig_path, args.data_root, test_ids, preds, title_suffix=f" (fold {i})")

        json_path = os.path.join(fold_out, "metrics_unet.json")
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"[UNet][fold {i}] Saved: {fig_path}")
        print(f"[UNet][fold {i}] Saved: {json_path}")

        preds_per_fold.append(preds)
        metrics_per_fold.append(metrics)

    # Ensemble: average predictions across folds (per test image)
    ens_out = os.path.join(base_out, "ensemble")
    ensure_dir(ens_out)

    ens_preds = {}
    for sid in test_ids:
        stack = np.stack([preds_per_fold[k][sid] for k in range(num_folds)], axis=0)  # (K,H,W)
        ens_preds[sid] = np.clip(np.mean(stack, axis=0).astype(np.float32), 0.0, 1.0)

    # compute ensemble metrics
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

    fig_path = os.path.join(ens_out, "unet_test_results.png")
    save_grid_figure(fig_path, args.data_root, test_ids, ens_preds, title_suffix=" (ensemble)")

    json_path = os.path.join(ens_out, "metrics_unet.json")
    with open(json_path, "w") as f:
        json.dump(ens_metrics, f, indent=2)

    print(f"[UNet][ensemble] Saved: {fig_path}")
    print(f"[UNet][ensemble] Saved: {json_path}")

# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./dataset/4c")
    ap.add_argument("--checkpoints_dir", default="./checkpoints/unet")
    ap.add_argument("--checkpoint", default="./checkpoints/unet/best.pt")
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--out_dir", default="./inference")
    ap.add_argument("--acc_tol", type=float, default=0.05)

    # K-fold inference options
    ap.add_argument("--kfold", default=True)
    ap.add_argument("--num_folds", type=int, default=5)
    ap.add_argument("--use_best", action="store_true", help="Use best_i.pt (default). If not set, uses final_i.pt.")

    args = ap.parse_args()
    infer(args)