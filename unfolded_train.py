import os
import json
import argparse
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import config

# ============================================================
# GLOBAL SWITCH
# ============================================================
FULLY_UNSUPERVISED = True  # keep your original behavior

# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============================================================
# Utils
# ============================================================
def load_float01(path: str) -> np.ndarray:
    img = np.array(Image.open(path).convert("F"), dtype=np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return np.clip(img, 0.0, 1.0)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# ============================================================
# K-Fold splitter (deterministic)
# ============================================================
def make_kfold_splits(ids, k=5, seed=0):
    ids = list(ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    folds = [[] for _ in range(k)]
    for i, sid in enumerate(ids):
        folds[i % k].append(sid)
    return folds

# ============================================================
# Dataset
# ============================================================
class SR4CamDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.sample_ids = sorted(
            [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        )

        good = []
        for sid in self.sample_ids:
            sdir = os.path.join(data_root, sid)

            ok = True
            for i in range(4):
                if not os.path.isfile(os.path.join(sdir, f"lr{i}_{sid}.png")):
                    ok = False
                    break
            if not ok:
                continue

            if FULLY_UNSUPERVISED:
                good.append(sid)
            else:
                if os.path.isfile(os.path.join(sdir, f"hr_{sid}.png")):
                    good.append(sid)

        self.sample_ids = good

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        sdir = os.path.join(self.data_root, sid)

        lrs = []
        for i in range(4):
            lr = load_float01(os.path.join(sdir, f"lr{i}_{sid}.png"))
            lrs.append(lr)
        lrs = np.stack(lrs, axis=0)  # (4,h,w)

        if FULLY_UNSUPERVISED:
            hr = torch.zeros((1,), dtype=torch.float32)  # unused
        else:
            hr = torch.from_numpy(load_float01(os.path.join(sdir, f"hr_{sid}.png")))

        return torch.from_numpy(lrs), hr, sid

# ============================================================
# Physics + UNN (unchanged)
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
    loss_acc += (diff0 ** 2).mean()
    grad += upsample_adjoint_avg(diff0, K)

    for i in range(3):
        dx = t_pred[:, i, 0]
        dy = t_pred[:, i, 1]

        preds = []
        for b in range(B):
            wb = circular_warp_grid_sample(x[b:b+1], dx[b], dy[b])
            preds.append(downsample_avg(wb, K))
        pred_i = torch.cat(preds, dim=0)

        diff = pred_i - lrs[:, (i+1):(i+2)]
        loss_acc += (diff ** 2).mean()

        diff_up = upsample_adjoint_avg(diff, K)

        backs = []
        for b in range(B):
            backs.append(circular_warp_grid_sample(diff_up[b:b+1], -dx[b], -dy[b]))
        grad += torch.cat(backs, dim=0)

    return loss_acc / 4.0, grad / 4.0

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
        x = F.interpolate(lrs[:, 0:1], size=(H_hr, W_hr), mode="bicubic", align_corners=False).clamp(0, 1)

        alphas = self.alphas()
        for k in range(self.K_steps):
            _, grad = data_fidelity_and_grad(x, lrs, t_pred, K)
            x = (x - alphas[k] * grad).clamp(0, 1)

        return x, t_pred

# ============================================================
# Train one fold
# ============================================================
def train_one_fold(
    fold_idx_1based: int,
    ds: SR4CamDataset,
    train_ids: list,
    val_ids: list,
    id_to_idx: dict,
    args,
    device,
    ckpt_folds_dir: str
):
    train_idxs = [id_to_idx[sid] for sid in train_ids]
    val_idxs   = [id_to_idx[sid] for sid in val_ids]

    dl_train = DataLoader(Subset(ds, train_idxs), batch_size=args.batch_size, shuffle=True, num_workers=0)
    dl_val   = DataLoader(Subset(ds, val_idxs), batch_size=1, shuffle=False, num_workers=0)

    model = UnfoldedSR(args.K_steps, args.max_shift_hr).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = None

    for ep in range(args.epochs):
        model.train()
        tr_losses = []

        for lrs, hr, _ in dl_train:
            lrs = lrs.to(device, torch.float32)
            h, w = lrs.shape[-2:]
            pred, t_pred = model(lrs, (2*h, 2*w))

            if FULLY_UNSUPERVISED:
                loss, _ = data_fidelity_and_grad(pred, lrs, t_pred, K=2)
            else:
                hr = hr.to(device)[:, None]
                loss = F.mse_loss(pred, hr)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        va_losses = []
        with torch.no_grad():
            for lrs, hr, _ in dl_val:
                lrs = lrs.to(device, torch.float32)
                h, w = lrs.shape[-2:]
                pred, t_pred = model(lrs, (2*h, 2*w))

                if FULLY_UNSUPERVISED:
                    loss, _ = data_fidelity_and_grad(pred, lrs, t_pred, K=2)
                else:
                    hr = hr.to(device)[:, None]
                    loss = F.mse_loss(pred, hr)

                va_losses.append(float(loss.detach().cpu().item()))

        tr = float(np.mean(tr_losses)) if tr_losses else float("nan")
        va = float(np.mean(va_losses)) if va_losses else float("nan")

        print(f"[UNN][fold {fold_idx_1based}] Epoch {ep+1:03d}/{args.epochs} | train={tr:.6e} | val={va:.6e}")

        if best_val is None or va < best_val:
            best_val = va
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": vars(args),
                    "fully_unsupervised": bool(FULLY_UNSUPERVISED),
                },
                os.path.join(ckpt_folds_dir, f"best_{fold_idx_1based}.pt")
            )

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": vars(args),
            "fully_unsupervised": bool(FULLY_UNSUPERVISED),
        },
        os.path.join(ckpt_folds_dir, f"final_{fold_idx_1based}.pt")
    )

    return {
        "fold": int(fold_idx_1based),
        "best_val_loss": float(best_val) if best_val is not None else None,
        "num_train": int(len(train_ids)),
        "num_val": int(len(val_ids)),
    }

# ============================================================
# Main train
# ============================================================
def train(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ensure_dir(args.checkpoints_dir)
    ckpt_folds_dir = os.path.join(args.checkpoints_dir, "folds")
    ensure_dir(ckpt_folds_dir)

    ds = SR4CamDataset(args.data_root)
    ids_all = ds.sample_ids
    if len(ids_all) == 0:
        raise RuntimeError("No samples found.")

    id_set = set(ids_all)
    test_ids_cfg = list(getattr(config, "TEST_SET", []))
    test_ids = [sid for sid in test_ids_cfg if sid in id_set]
    if len(test_ids) == 0:
        raise RuntimeError("config.TEST_SET is empty or none of its IDs exist in the dataset.")

    trainval_ids = [sid for sid in ids_all if sid not in set(test_ids)]
    if len(trainval_ids) == 0:
        raise RuntimeError("After removing TEST_SET, no samples remain for train/val.")

    print(f"[UNN] Total valid: {len(ids_all)} | TEST_SET: {len(test_ids)} | TrainVal: {len(trainval_ids)} | Steps: {args.K_steps}")

    save_json(
        {
            "data_root": str(args.data_root),
            "seed": int(args.seed),
            "kfold": bool(args.kfold),
            "num_folds": int(args.num_folds),
            "test": test_ids,
            "trainval": trainval_ids,
            "fully_unsupervised": bool(FULLY_UNSUPERVISED),
        },
        os.path.join(args.checkpoints_dir, "split_test.json"),
    )

    id_to_idx = {sid: i for i, sid in enumerate(ids_all)}

    if not args.kfold:
        folds = make_kfold_splits(trainval_ids, k=5, seed=args.seed)
        val_ids = folds[0]
        train_ids = [sid for sid in trainval_ids if sid not in set(val_ids)]

        save_json(
            {
                "mode": "single_80_20_from_trainval",
                "seed": int(args.seed),
                "train": train_ids,
                "val": val_ids,
                "test": test_ids,
                "fully_unsupervised": bool(FULLY_UNSUPERVISED),
            },
            os.path.join(args.checkpoints_dir, "split.json"),
        )

        meta = train_one_fold(
            fold_idx_1based=1,
            ds=ds,
            train_ids=train_ids,
            val_ids=val_ids,
            id_to_idx=id_to_idx,
            args=args,
            device=device,
            ckpt_folds_dir=ckpt_folds_dir,
        )

        best1 = os.path.join(ckpt_folds_dir, "best_1.pt")
        final1 = os.path.join(ckpt_folds_dir, "final_1.pt")
        torch.save(torch.load(best1, map_location="cpu"), os.path.join(args.checkpoints_dir, "best.pt"))
        torch.save(torch.load(final1, map_location="cpu"), os.path.join(args.checkpoints_dir, "final.pt"))

        save_json({"single_run": meta}, os.path.join(args.checkpoints_dir, "kfold_summary.json"))
        print(f"[UNN] Saved: {args.checkpoints_dir}/best.pt and folds/best_1.pt")
        return

    folds = make_kfold_splits(trainval_ids, k=args.num_folds, seed=args.seed)

    fold_summaries = []
    for i in range(args.num_folds):
        fold_idx_1based = i + 1
        val_ids = folds[i]
        train_ids = [sid for sid in trainval_ids if sid not in set(val_ids)]

        save_json(
            {
                "mode": "kfold_trainval_only",
                "seed": int(args.seed),
                "fold": int(fold_idx_1based),
                "num_folds": int(args.num_folds),
                "train": train_ids,
                "val": val_ids,
                "test": test_ids,
                "fully_unsupervised": bool(FULLY_UNSUPERVISED),
            },
            os.path.join(ckpt_folds_dir, f"split_{fold_idx_1based}.json"),
        )

        meta = train_one_fold(
            fold_idx_1based=fold_idx_1based,
            ds=ds,
            train_ids=train_ids,
            val_ids=val_ids,
            id_to_idx=id_to_idx,
            args=args,
            device=device,
            ckpt_folds_dir=ckpt_folds_dir,
        )
        fold_summaries.append(meta)

    save_json(
        {
            "model": "unn",
            "kfold": True,
            "num_folds": int(args.num_folds),
            "seed": int(args.seed),
            "fully_unsupervised": bool(FULLY_UNSUPERVISED),
            "folds": fold_summaries,
            "paths": {
                "folds_dir": ckpt_folds_dir,
                "best_pattern": "best_{i}.pt",
                "final_pattern": "final_{i}.pt",
            },
        },
        os.path.join(args.checkpoints_dir, "kfold_summary.json"),
    )

    print(f"[UNN] Done. Checkpoints saved under: {ckpt_folds_dir}")

# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./dataset/4c")
    ap.add_argument("--checkpoints_dir", default="./checkpoints/unn")
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--K_steps", type=int, default=8)
    ap.add_argument("--max_shift_hr", type=float, default=40.0)

    # K-fold options
    ap.add_argument("--kfold", default=True)
    ap.add_argument("--num_folds", type=int, default=5)

    args = ap.parse_args()
    train(args)