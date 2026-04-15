"""
Training Pipeline — Multi-Platform Bot Detector (3-class)

Supports four data modes:
  --data synthetic      → train on generated synthetic data (default)
  --data real           → train on real Instagram datasets
  --data hybrid         → train on real data augmented with synthetic data
  --data multiplatform  → train on Instagram real + Twitter/Facebook/Snapchat synthetic

Usage:
  python model/train.py                              # synthetic (legacy)
  python model/train.py --data real                  # real Instagram only
  python model/train.py --data hybrid                # real + synthetic augmentation
  python model/train.py --data multiplatform         # unified multi-platform
  python model/train.py --data multiplatform --epochs 100 --lr 5e-4
"""

import os, sys, json, pickle, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.network import BotDetectorNet, FocalLoss
from data.generate_data import generate_dataset, save_splits, LABELS

# ── Constants ─────────────────────────────────────────────────────────────
ARTIFACTS = ROOT / "model" / "artifacts"

LOG_COLS = [
    "followers_count", "following_count", "posts_count",
    "avg_likes_per_post", "avg_comments_per_post",
    "account_age_days", "avg_caption_length",
    "likes_comments_ratio",  # can be 0-100, benefits from log compression
]
FEAT_COLS = BotDetectorNet.FEATURE_COLS
TARGET    = "label"


# ── Dataset ───────────────────────────────────────────────────────────────
class IGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ── Preprocessing ─────────────────────────────────────────────────────────
def make_scaler(df):
    scaler = RobustScaler()
    df2    = df.copy()
    for c in LOG_COLS:
        df2[c] = np.log1p(df2[c])
    scaler.fit(df2[FEAT_COLS])
    return scaler

def transform(df, scaler):
    df2 = df.copy()
    for c in LOG_COLS:
        df2[c] = np.log1p(df2[c])
    return scaler.transform(df2[FEAT_COLS]).astype(np.float32)


# ── Data loading ──────────────────────────────────────────────────────────
def load_data(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test splits based on --data mode."""
    DATA = ROOT / "data"

    if args.data == "real":
        # Real data from prepare_real_data.py
        real_train = DATA / "real_train.csv"
        if not real_train.exists():
            print("ERROR: Real data not found. Run first:")
            print("  python data/prepare_real_data.py")
            sys.exit(1)
        tr = pd.read_csv(DATA / "real_train.csv")
        va = pd.read_csv(DATA / "real_val.csv")
        te = pd.read_csv(DATA / "real_test.csv")
        print(f"Data mode: REAL ({len(tr)+len(va)+len(te):,} samples)")

    elif args.data == "hybrid":
        # Real data + synthetic augmentation
        real_train = DATA / "real_train.csv"
        if not real_train.exists():
            print("ERROR: Real data not found. Run first:")
            print("  python data/prepare_real_data.py")
            sys.exit(1)

        tr_real = pd.read_csv(DATA / "real_train.csv")
        va = pd.read_csv(DATA / "real_val.csv")
        te = pd.read_csv(DATA / "real_test.csv")

        # Generate synthetic data to augment underrepresented classes
        real_counts = tr_real["label"].value_counts()
        max_class = real_counts.max()

        # Generate enough synthetic data to balance classes + add diversity
        n_synthetic = max(args.n_samples, int(len(tr_real) * 0.3))
        print(f"Generating {n_synthetic:,} synthetic samples for augmentation...")
        df_synth = generate_dataset(n_synthetic)

        # Only use synthetic train split for augmentation
        synth_train, _, _ = (
            df_synth.iloc[:int(n_synthetic * 0.7)],
            df_synth.iloc[int(n_synthetic * 0.7):int(n_synthetic * 0.85)],
            df_synth.iloc[int(n_synthetic * 0.85):],
        )

        tr = pd.concat([tr_real, synth_train], ignore_index=True)
        tr = tr.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"Data mode: HYBRID")
        print(f"  Real train   : {len(tr_real):,}")
        print(f"  Synthetic aug: {len(synth_train):,}")
        print(f"  Total train  : {len(tr):,}")
        print(f"  Val (real)   : {len(va):,}")
        print(f"  Test (real)  : {len(te):,}")

    elif args.data == "multiplatform":
        # Unified multi-platform: Instagram real + Twitter/Facebook/Snapchat synthetic
        real_train = DATA / "real_train.csv"
        if not real_train.exists():
            print("ERROR: Instagram real data not found. Run first:")
            print("  python data/prepare_real_data.py")
            sys.exit(1)

        # Load Instagram real data
        tr_ig = pd.read_csv(DATA / "real_train.csv")
        va_ig = pd.read_csv(DATA / "real_val.csv")
        te_ig = pd.read_csv(DATA / "real_test.csv")
        print(f"  Instagram (real): train={len(tr_ig):,} val={len(va_ig):,} test={len(te_ig):,}")

        # Load platform data (real or synthetic)
        platform_dfs_tr, platform_dfs_va, platform_dfs_te = [], [], []
        for plat in ["twitter", "facebook", "snapchat"]:
            plat_train = DATA / f"{plat}_train.csv"
            if not plat_train.exists():
                print(f"  WARNING: {plat} data not found — run: python data/generate_platform_data.py")
                continue
            ptr = pd.read_csv(DATA / f"{plat}_train.csv")
            pva = pd.read_csv(DATA / f"{plat}_val.csv")
            pte = pd.read_csv(DATA / f"{plat}_test.csv")

            # Check if dataset is missing "Suspicious" class (label=2)
            # If so, augment with synthetic suspicious samples
            has_suspicious = (ptr["label"] == 2).sum() > 0
            if not has_suspicious:
                from data.generate_platform_data import GENERATORS
                if plat in GENERATORS:
                    n_susp = int(len(ptr) * 0.15)  # add ~15% suspicious
                    print(f"  {plat.capitalize():10}: adding {n_susp:,} synthetic Suspicious samples")
                    susp_rows = [GENERATORS[plat][2]() for _ in range(n_susp)]
                    susp_df = pd.DataFrame(susp_rows)
                    ptr = pd.concat([ptr, susp_df.iloc[:int(n_susp*0.7)]], ignore_index=True)
                    pva = pd.concat([pva, susp_df.iloc[int(n_susp*0.7):int(n_susp*0.85)]], ignore_index=True)
                    pte = pd.concat([pte, susp_df.iloc[int(n_susp*0.85):]], ignore_index=True)

            src = "real" if plat == "twitter" and (DATA / "real" / "twitter").exists() else "synth"
            platform_dfs_tr.append(ptr)
            platform_dfs_va.append(pva)
            platform_dfs_te.append(pte)
            print(f"  {plat.capitalize():10} ({src:5}): train={len(ptr):,} val={len(pva):,} test={len(pte):,}")

        # Combine all platforms
        tr = pd.concat([tr_ig] + platform_dfs_tr, ignore_index=True)
        va = pd.concat([va_ig] + platform_dfs_va, ignore_index=True)
        te = pd.concat([te_ig] + platform_dfs_te, ignore_index=True)

        # Shuffle training set
        tr = tr.sample(frac=1, random_state=42).reset_index(drop=True)

        total = len(tr) + len(va) + len(te)
        print(f"\n  Data mode: MULTIPLATFORM ({total:,} total samples)")
        print(f"  Combined: train={len(tr):,} val={len(va):,} test={len(te):,}")

    else:
        # Synthetic only (legacy)
        if not (DATA / "train.csv").exists():
            print("Generating synthetic dataset...")
            df = generate_dataset(args.n_samples)
            save_splits(df, out_dir=str(DATA))
        tr = pd.read_csv(DATA / "train.csv")
        va = pd.read_csv(DATA / "val.csv")
        te = pd.read_csv(DATA / "test.csv")
        print(f"Data mode: SYNTHETIC ({len(tr)+len(va)+len(te):,} samples)")

    # Print class distribution for training set
    print(f"\nTraining set distribution:")
    for label_id, label_name in LABELS.items():
        count = (tr["label"] == label_id).sum()
        print(f"  {label_name:>12}: {count:,}  ({count/len(tr):.1%})")

    return tr, va, te


# ── Training loops ────────────────────────────────────────────────────────
def train_epoch(model, loader, opt, crit, device):
    model.train()
    loss_sum, n, correct = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        logits = model(X)
        loss   = crit(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_sum += loss.item() * len(y)
        correct  += (logits.argmax(1) == y).sum().item()
        n        += len(y)
    return loss_sum / n, correct / n

@torch.no_grad()
def eval_epoch(model, loader, crit, device):
    model.eval()
    loss_sum, preds, truths = 0.0, [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss_sum += crit(logits, y).item() * len(y)
        preds  += logits.argmax(1).cpu().tolist()
        truths += y.cpu().tolist()
    n = len(truths)
    return {
        "loss":     loss_sum / n,
        "accuracy": accuracy_score(truths, preds),
        "macro_f1": f1_score(truths, preds, average="macro"),
    }, truths, preds


# ── Main ──────────────────────────────────────────────────────────────────
def main(args):
    tr, va, te = load_data(args)

    scaler = make_scaler(tr)
    Xtr, Xva, Xte = transform(tr, scaler), transform(va, scaler), transform(te, scaler)
    ytr, yva, yte  = tr[TARGET].values, va[TARGET].values, te[TARGET].values

    # Class weights (inverse freq)
    counts = np.bincount(ytr)
    weights = torch.tensor(len(ytr) / (len(counts) * counts), dtype=torch.float32)

    trDS = IGDataset(Xtr, ytr)
    vaDS = IGDataset(Xva, yva)
    teDS = IGDataset(Xte, yte)

    trL = DataLoader(trDS, batch_size=args.batch, shuffle=True,  num_workers=0)
    vaL = DataLoader(vaDS, batch_size=args.batch, shuffle=False, num_workers=0)
    teL = DataLoader(teDS, batch_size=args.batch, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")

    model   = BotDetectorNet(input_dim=27, hidden=128, dropout=0.3).to(device)
    crit    = FocalLoss(gamma=2.0, weight=weights.to(device))
    opt     = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched   = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    best_f1, patience = 0.0, 0

    header = f"{'Ep':>4} | {'TrLoss':>7} | {'TrAcc':>6} | {'VaF1':>6} | {'VaAcc':>6}"
    print(f"\n{'─'*50}\n{header}\n{'─'*50}")

    for ep in range(1, args.epochs + 1):
        tl, ta = train_epoch(model, trL, opt, crit, device)
        vm, *_ = eval_epoch(model, vaL, crit, device)
        sched.step()

        print(f"{ep:>4} | {tl:>7.4f} | {ta:>6.4f} | {vm['macro_f1']:>6.4f} | {vm['accuracy']:>6.4f}")

        if vm["macro_f1"] > best_f1:
            best_f1, patience = vm["macro_f1"], 0
            torch.save(model.state_dict(), ARTIFACTS / "best_model.pt")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  Early stop at epoch {ep}")
                break

    # ── Test evaluation ──────────────────────────────────────────────────
    model.load_state_dict(torch.load(ARTIFACTS / "best_model.pt", map_location=device))
    metrics, yt, yp = eval_epoch(model, teL, crit, device)

    print(f"\n{'─'*50}")
    print("  Test results")
    print(f"{'─'*50}")
    print(classification_report(yt, yp, target_names=list(LABELS.values())))
    print("Confusion matrix:")
    cm = confusion_matrix(yt, yp)
    print(cm)

    # Per-class precision/recall (focus on Bot detection as recommended)
    prec_per = precision_score(yt, yp, average=None, labels=[0,1,2])
    rec_per  = recall_score(yt, yp, average=None, labels=[0,1,2])
    print(f"\n  Bot Detection Quality (class=1):")
    print(f"    Precision : {prec_per[1]:.4f}  (of predicted bots, how many are real bots)")
    print(f"    Recall    : {rec_per[1]:.4f}  (of real bots, how many did we catch)")
    print(f"    F1        : {2*prec_per[1]*rec_per[1]/(prec_per[1]+rec_per[1]+1e-9):.4f}")
    print(f"  Suspicious Detection Quality (class=2):")
    print(f"    Precision : {prec_per[2]:.4f}")
    print(f"    Recall    : {rec_per[2]:.4f}")
    print(f"    F1        : {2*prec_per[2]*rec_per[2]/(prec_per[2]+rec_per[2]+1e-9):.4f}")

    # ── Save artifacts ───────────────────────────────────────────────────
    with open(ARTIFACTS / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "input_dim":      27,
        "hidden":         128,
        "dropout":        0.3,
        "feature_cols":   FEAT_COLS,
        "log_cols":       LOG_COLS,
        "label_names":    list(LABELS.values()),
        "test_metrics":   metrics,
        "class_weights":  weights.tolist(),
        "data_mode":      args.data,
        "train_samples":  len(tr),
        "val_samples":    len(va),
        "test_samples":   len(te),
    }
    with open(ARTIFACTS / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Saved to {ARTIFACTS}")
    for f in ["best_model.pt", "scaler.pkl", "model_meta.json"]:
        print(f"   - {f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",      type=str,   default="synthetic",
                   choices=["synthetic", "real", "hybrid", "multiplatform"],
                   help="Data source: synthetic, real, hybrid, or multiplatform")
    p.add_argument("--epochs",    type=int,   default=80)
    p.add_argument("--batch",     type=int,   default=256)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--patience",  type=int,   default=12)
    p.add_argument("--n_samples", type=int,   default=12_000)
    main(p.parse_args())
