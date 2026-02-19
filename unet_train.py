import os
import json
import argparse
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset

import config

DATASET = "hr_unn"  # one of: "hr" | "hr_unn" | "HRL"

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
    return enc[:, :, y0:y0 + H, x0:x0 + W]

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# ============================================================
# K-Fold splitter (deterministic)
# ============================================================
def make_kfold_splits(ids, k=5, seed=0):
    """
    Deterministic K-fold split of `ids`.
    Returns list of folds, each is a list of ids (validation fold).
    """
    ids = list(ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    folds = [[] for _ in range(k)]
    for i, sid in enumerate(ids):
        folds[i % k].append(sid)
    return folds

# ============================================================
# Dataset (4-channel input, 2x output)
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
            if os.path.isfile(os.path.join(sdir, f"{DATASET}_{sid}.png")):
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

        hr = load_float01(os.path.join(sdir, f"{DATASET}_{sid}.png"))  # (2h,2w)
        return torch.from_numpy(lrs), torch.from_numpy(hr), sid

# ============================================================
# U-Net
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

    model = UNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = None
    hist = {"train_loss": [], "val_loss": []}

    for ep in range(args.epochs):
        model.train()
        tr_losses = []
        for lrs, hr, _ in dl_train:
            lrs = lrs.to(device=device, dtype=torch.float32)  # (B,4,h,w)
            hr = hr.to(device=device, dtype=torch.float32)    # (B,H,W)
            hr = hr[:, None, :, :]

            h, w = lrs.shape[-2:]
            H_hr, W_hr = 2 * h, 2 * w

            pred = model(lrs, (H_hr, W_hr))
            loss = F.mse_loss(pred, hr)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tr_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        va_losses = []
        with torch.no_grad():
            for lrs, hr, _ in dl_val:
                lrs = lrs.to(device=device, dtype=torch.float32)
                hr = hr.to(device=device, dtype=torch.float32)
                hr = hr[:, None, :, :]

                h, w = lrs.shape[-2:]
                H_hr, W_hr = 2 * h, 2 * w

                pred = model(lrs, (H_hr, W_hr))
                loss = F.mse_loss(pred, hr)
                va_losses.append(float(loss.detach().cpu().item()))

        tr = float(np.mean(tr_losses)) if tr_losses else float("nan")
        va = float(np.mean(va_losses)) if va_losses else float("nan")
        hist["train_loss"].append(tr)
        hist["val_loss"].append(va)

        print(f"[UNet][fold {fold_idx_1based}] Epoch {ep+1:03d}/{args.epochs} | train={tr:.6e} | val={va:.6e}")

        if best_val is None or va < best_val:
            best_val = va
            torch.save(model.state_dict(), os.path.join(ckpt_folds_dir, f"best_{fold_idx_1based}.pt"))

    torch.save(model.state_dict(), os.path.join(ckpt_folds_dir, f"final_{fold_idx_1based}.pt"))

    # loss curve per fold
    try:
        plt.figure(figsize=(7, 4))
        plt.plot(hist["train_loss"], label="train")
        plt.plot(hist["val_loss"], label="val")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("mse")
        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_folds_dir, f"loss_curve_{fold_idx_1based}.png"), dpi=160)
        plt.close()
    except Exception:
        pass

    return {
        "fold": int(fold_idx_1based),
        "best_val_mse": float(best_val) if best_val is not None else None,
        "num_train": int(len(train_ids)),
        "num_val": int(len(val_ids)),
    }

# ============================================================
# Main train
# ============================================================
def train(args):
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # dirs
    ensure_dir(args.checkpoints_dir)
    ckpt_folds_dir = os.path.join(args.checkpoints_dir, "folds")
    ensure_dir(ckpt_folds_dir)

    # dataset
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

    print(f"[UNet] Total valid: {len(ids_all)} | TEST_SET: {len(test_ids)} | TrainVal: {len(trainval_ids)}")

    # save fixed test set file (used by inference)
    save_json(
        {
            "data_root": str(args.data_root),
            "dataset_target": str(DATASET),
            "seed": int(args.seed),
            "kfold": bool(args.kfold),
            "num_folds": int(args.num_folds),
            "test": test_ids,
            "trainval": trainval_ids,
        },
        os.path.join(args.checkpoints_dir, "split_test.json"),
    )

    # id mapping
    id_to_idx = {sid: i for i, sid in enumerate(ids_all)}

    # single split mode (non-kfold): make one 80/20 split over trainval
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
            },
            os.path.join(args.checkpoints_dir, "split.json"),
        )

        # train once, and also keep legacy best.pt/final.pt
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

        # copy fold-1 naming to legacy
        best1 = os.path.join(ckpt_folds_dir, "best_1.pt")
        final1 = os.path.join(ckpt_folds_dir, "final_1.pt")
        torch.save(torch.load(best1, map_location="cpu"), os.path.join(args.checkpoints_dir, "best.pt"))
        torch.save(torch.load(final1, map_location="cpu"), os.path.join(args.checkpoints_dir, "final.pt"))

        save_json({"single_run": meta}, os.path.join(args.checkpoints_dir, "kfold_summary.json"))
        print(f"[UNet] Saved: {args.checkpoints_dir}/best.pt and folds/best_1.pt")

        return

    # k-fold mode
    folds = make_kfold_splits(trainval_ids, k=args.num_folds, seed=args.seed)

    fold_summaries = []
    for i in range(args.num_folds):
        fold_idx_1based = i + 1
        val_ids = folds[i]
        val_set = set(val_ids)
        train_ids = [sid for sid in trainval_ids if sid not in val_set]

        save_json(
            {
                "mode": "kfold_trainval_only",
                "seed": int(args.seed),
                "fold": int(fold_idx_1based),
                "num_folds": int(args.num_folds),
                "train": train_ids,
                "val": val_ids,
                "test": test_ids,
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
            "model": "unet",
            "dataset_target": str(DATASET),
            "kfold": True,
            "num_folds": int(args.num_folds),
            "seed": int(args.seed),
            "folds": fold_summaries,
            "paths": {
                "folds_dir": ckpt_folds_dir,
                "best_pattern": "best_{i}.pt",
                "final_pattern": "final_{i}.pt",
            },
        },
        os.path.join(args.checkpoints_dir, "kfold_summary.json"),
    )

    print(f"[UNet] Done. Checkpoints saved under: {ckpt_folds_dir}")

# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./dataset/4c")
    ap.add_argument("--checkpoints_dir", default="./checkpoints/unet")
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)

    # K-fold options
    ap.add_argument("--kfold", default=True)
    ap.add_argument("--num_folds", type=int, default=5, help="Number of folds (default 5 => 80/20).")

    args = ap.parse_args()
    train(args)