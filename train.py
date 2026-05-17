"""
train.py — WELFake ANN training pipeline for TruthLens

Usage
─────
  python train.py                           # looks for WELFake_Dataset.csv in CWD
  python train.py --data path/to/file.csv   # custom path
  python train.py --epochs 30 --samples 20000

Output
──────
  ann.pth  — saved every time val_acc improves; used by predictor.py
"""

import os
import re
import sys
import time
import argparse
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer

from model import FakeNewsANN

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ── Text cleaning (same as used at inference time) ────────────
_RE_URL  = re.compile(r'https?://\S+|www\.\S+')
_RE_HTML = re.compile(r'<[^>]+>')
_RE_PUNC = re.compile(r'[^\w\s]')
_RE_WS   = re.compile(r'\s+')


def clean(text: str) -> str:
    text = str(text)
    text = _RE_URL.sub(' ', text)
    text = _RE_HTML.sub(' ', text)
    text = _RE_PUNC.sub(' ', text)
    return _RE_WS.sub(' ', text).lower().strip()


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Train TruthLens ANN on WELFake')
    parser.add_argument('--data',    default='WELFake_Dataset.csv',
                        help='Path to WELFake_Dataset.csv')
    parser.add_argument('--out',     default='ann.pth',
                        help='Output model path')
    parser.add_argument('--samples', type=int, default=15_000,
                        help='Balanced samples per class (≤ 15 000)')
    parser.add_argument('--epochs',  type=int, default=40,
                        help='Training epochs')
    parser.add_argument('--lr',      type=float, default=1e-3)
    parser.add_argument('--batch',   type=int,   default=64)
    parser.add_argument('--seed',    type=int,   default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info('Device: %s', device)

    # ── Load dataset ──────────────────────────────────────────
    if not os.path.exists(args.data):
        log.error('Dataset not found: %s', args.data)
        log.error('Download from https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification')
        sys.exit(1)

    log.info('Loading %s …', args.data)
    df = pd.read_csv(args.data)
    df.columns = [c.lower().strip() for c in df.columns]

    missing = {'title', 'text', 'label'} - set(df.columns)
    if missing:
        log.error('Missing columns: %s  (found: %s)', missing, list(df.columns))
        sys.exit(1)

    df = df.dropna(subset=['title', 'text', 'label'])
    df['label']   = df['label'].astype(int)
    df['content'] = (df['title'].fillna('') + ' ' + df['text'].fillna('')).apply(clean)
    df = df[df['content'].str.len() > 15]

    n_per_class = min(
        args.samples,
        (df.label == 0).sum(),
        (df.label == 1).sum(),
    )
    log.info('Using %d samples per class (%d total)', n_per_class, n_per_class * 2)

    balanced = pd.concat([
        df[df.label == 0].sample(n_per_class, random_state=args.seed),
        df[df.label == 1].sample(n_per_class, random_state=args.seed),
    ]).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # ── Embed ─────────────────────────────────────────────────
    log.info('Loading sentence encoder …')
    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    log.info('Encoding %d texts …', len(balanced))
    t0 = time.time()
    X  = encoder.encode(
        balanced['content'].tolist(),
        batch_size=256,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    log.info('Encoded in %.1f s', time.time() - t0)

    y = balanced['label'].values.astype(np.float32).reshape(-1, 1)

    # ── Split ─────────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.15, random_state=args.seed, stratify=y
    )
    log.info('Train: %d  |  Test: %d', len(y_tr), len(y_te))

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=args.batch, shuffle=True,
    )

    X_te_t = torch.from_numpy(X_te).to(device)
    y_te_t = torch.from_numpy(y_te).to(device)

    # ── Model setup ───────────────────────────────────────────
    model     = FakeNewsANN(input_dim=X.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # ReduceLROnPlateau steps once per epoch — zero step-count crash risk
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=4, factor=0.5, min_lr=1e-6,
    )

    best_acc = 0.0

    # ── Training loop ─────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()

        avg_loss = ep_loss / len(tr_loader)

        # Eval
        model.eval()
        with torch.no_grad():
            preds = model(X_te_t)
            acc   = ((preds >= 0.5).float() == y_te_t).float().mean().item()

        # Step scheduler once per epoch (NOT inside batch loop)
        scheduler.step(avg_loss)

        if epoch % 5 == 0 or epoch == 1:
            log.info(
                'Epoch %3d/%d  loss=%.4f  val_acc=%.4f  lr=%.2e',
                epoch, args.epochs, avg_loss, acc,
                optimizer.param_groups[0]['lr'],
            )

        if acc > best_acc:
            best_acc = acc
            torch.save({
                'state_dict': model.state_dict(),
                'val_acc':    acc,
                'epoch':      epoch,
                'input_dim':  X.shape[1],
            }, args.out)
            log.info('  ↳ New best (%.4f) — saved to %s', acc, args.out)

    # ── Final report ──────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        preds_np = (model(X_te_t) >= 0.5).cpu().int().numpy().flatten()
    gt_np = y_te_t.cpu().int().numpy().flatten()

    log.info('\nBest val accuracy: %.4f', best_acc)
    print(classification_report(gt_np, preds_np, target_names=['real', 'fake']))
    log.info('Model saved → %s', args.out)


if __name__ == '__main__':
    main()