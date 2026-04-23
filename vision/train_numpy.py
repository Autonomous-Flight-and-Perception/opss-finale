#!/usr/bin/env python3
"""
Train the DeltaAccelNN3D using pure numpy (no torch dependency).

The network is tiny (15→32→3, ~611 params), so manual forward/backward
through 2 linear layers + tanh is straightforward.

Architecture:
    y = tanh(tanh(x @ W1.T + b1) @ W2.T + b2) * A_MAX

Usage:
    python train_numpy.py
    python train_numpy.py --data data/episodes/train --val data/episodes/val
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
A_MAX = 15.0
FEAT_DIM = 15
HIDDEN_DIM = 32
OUTPUT_DIM = 3


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_episodes(data_dir):
    """Load all episode NPZs, return (features, targets) arrays."""
    data_dir = Path(data_dir)
    all_feats, all_targets = [], []

    for path in sorted(data_dir.glob('*.npz')):
        d = np.load(path)
        feat = d['feat']
        target = d['target']
        valid = d['valid']
        mask = valid & ~np.isnan(target[:, 0]) & ~np.isnan(feat[:, 0])
        if mask.sum() > 0:
            all_feats.append(feat[mask])
            all_targets.append(target[mask])

    if not all_feats:
        raise ValueError(f"No valid samples in {data_dir}")

    return np.vstack(all_feats), np.vstack(all_targets)


# ---------------------------------------------------------------------------
# Model (numpy)
# ---------------------------------------------------------------------------

class NumpyModel:
    """
    DeltaAccelNN3D in pure numpy: Linear(15,32)->Tanh->Linear(32,3)->Tanh->*A_MAX

    Parameters:
        W1: (32, 15), b1: (32,)
        W2: (3, 32),  b2: (3,)
    """

    def __init__(self, a_max=A_MAX):
        self.a_max = a_max
        # Xavier initialization
        self.W1 = np.random.randn(HIDDEN_DIM, FEAT_DIM) * np.sqrt(2.0 / (FEAT_DIM + HIDDEN_DIM))
        self.b1 = np.zeros(HIDDEN_DIM)
        self.W2 = np.random.randn(OUTPUT_DIM, HIDDEN_DIM) * np.sqrt(2.0 / (HIDDEN_DIM + OUTPUT_DIM))
        self.b2 = np.zeros(OUTPUT_DIM)

    def forward(self, X):
        """
        Forward pass.

        Args:
            X: (batch, 12)

        Returns:
            Y: (batch, 3), cache for backward
        """
        # Layer 1: z1 = X @ W1.T + b1, h1 = tanh(z1)
        z1 = X @ self.W1.T + self.b1  # (batch, 32)
        h1 = np.tanh(z1)

        # Layer 2: z2 = h1 @ W2.T + b2, h2 = tanh(z2)
        z2 = h1 @ self.W2.T + self.b2  # (batch, 3)
        h2 = np.tanh(z2)

        Y = h2 * self.a_max

        cache = (X, z1, h1, z2, h2)
        return Y, cache

    def backward(self, dY, cache):
        """
        Backward pass.

        Args:
            dY: (batch, 3) gradient of loss w.r.t. output
            cache: from forward

        Returns:
            grads dict with dW1, db1, dW2, db2
        """
        X, z1, h1, z2, h2 = cache
        batch = X.shape[0]

        # dY = dL/dY, Y = h2 * a_max
        dh2 = dY * self.a_max  # (batch, 3)

        # h2 = tanh(z2), d_tanh = 1 - tanh²
        dz2 = dh2 * (1 - h2 ** 2)  # (batch, 3)

        # z2 = h1 @ W2.T + b2
        dW2 = dz2.T @ h1 / batch  # (3, 32)
        db2 = dz2.mean(axis=0)     # (3,)
        dh1 = dz2 @ self.W2        # (batch, 32)

        # h1 = tanh(z1)
        dz1 = dh1 * (1 - h1 ** 2)  # (batch, 32)

        # z1 = X @ W1.T + b1
        dW1 = dz1.T @ X / batch  # (32, 12)
        db1 = dz1.mean(axis=0)   # (32,)

        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

    def n_params(self):
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load(self, path):
        d = np.load(path)
        self.W1 = d['W1']
        self.b1 = d['b1']
        self.W2 = d['W2']
        self.b2 = d['b2']


# ---------------------------------------------------------------------------
# Adam optimizer
# ---------------------------------------------------------------------------

class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-6):
        self.params = params  # dict of name -> array
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads):
        self.t += 1
        for k in self.params:
            g = grads['d' + k]
            # Weight decay (decoupled, AdamW style)
            if self.weight_decay > 0:
                self.params[k] *= (1 - self.lr * self.weight_decay)

            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * g ** 2

            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)

            self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    train_feats, train_targets,
    val_feats=None, val_targets=None,
    epochs=200, batch_size=1024, lr=1e-3,
    patience=15, min_delta=1e-5, seed=42,
):
    np.random.seed(seed)

    # Compute normalization stats from TRAINING data only
    feat_mean = train_feats.mean(axis=0)
    feat_std = train_feats.std(axis=0)
    feat_std = np.maximum(feat_std, 1e-8)

    stats = {
        'mean': feat_mean.tolist(),
        'std': feat_std.tolist(),
        'feat_dim': FEAT_DIM,
        'nn_output_dim': OUTPUT_DIM,
        'version': 'feat15_3d_v2',
    }

    # Normalize
    X_train = (train_feats - feat_mean) / feat_std
    Y_train = train_targets

    if val_feats is not None:
        X_val = (val_feats - feat_mean) / feat_std  # use TRAIN stats
        Y_val = val_targets
    else:
        # Auto-split 80/20
        n = len(X_train)
        idx = np.random.permutation(n)
        n_train = int(n * 0.8)
        X_val = X_train[idx[n_train:]]
        Y_val = Y_train[idx[n_train:]]
        X_train = X_train[idx[:n_train]]
        Y_train = Y_train[idx[:n_train]]

    print(f"  Training: {len(X_train)} samples, Validation: {len(X_val)} samples")

    # Initialize model
    model = NumpyModel(a_max=A_MAX)
    print(f"  Model parameters: {model.n_params()}")

    params = {'W1': model.W1, 'b1': model.b1, 'W2': model.W2, 'b2': model.b2}
    optimizer = Adam(params, lr=lr, weight_decay=1e-6)

    best_val_loss = float('inf')
    best_state = None
    wait = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # Shuffle training data
        perm = np.random.permutation(len(X_train))
        X_shuf = X_train[perm]
        Y_shuf = Y_train[perm]

        # Mini-batch training
        train_losses = []
        for i in range(0, len(X_shuf), batch_size):
            X_batch = X_shuf[i:i + batch_size]
            Y_batch = Y_shuf[i:i + batch_size]

            # Forward
            Y_pred, cache = model.forward(X_batch)

            # MSE loss
            diff = Y_pred - Y_batch
            loss = np.mean(diff ** 2)
            train_losses.append(loss)

            # Backward: dL/dY = 2 * (Y_pred - Y_batch) / (batch * output_dim)
            dY = 2 * diff / (diff.shape[0] * diff.shape[1])
            grads = model.backward(dY, cache)

            # Update
            optimizer.step(grads)

        train_loss = np.mean(train_losses)

        # Validation
        Y_val_pred, _ = model.forward(X_val)
        val_loss = np.mean((Y_val_pred - Y_val) ** 2)

        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = {
                'W1': model.W1.copy(), 'b1': model.b1.copy(),
                'W2': model.W2.copy(), 'b2': model.b2.copy(),
            }
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch+1} (best val={best_val_loss:.6f})")
                break

    # Restore best
    if best_state is not None:
        model.W1 = best_state['W1']
        model.b1 = best_state['b1']
        model.W2 = best_state['W2']
        model.b2 = best_state['b2']

    return model, stats, history


def main():
    parser = argparse.ArgumentParser(description="Train DeltaAccelNN3D (numpy, no torch)")
    parser.add_argument("--data", default="data/episodes/train", help="Training data directory")
    parser.add_argument("--data-extra", type=str, default=None,
                        help="Extra training data dir (e.g. DAgger episodes)")
    parser.add_argument("--extra-weight", type=float, default=1.0,
                        help="Weight for extra data (use <1 for subsampling)")
    parser.add_argument("--val", default="data/episodes/val", help="Validation data directory")
    parser.add_argument("--output", default="models", help="Output directory")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("OPSS NN Training (numpy, no torch)")
    print("=" * 50)

    # Load data
    print(f"\n  Loading training data from {args.data}...")
    train_feats, train_targets = load_episodes(args.data)
    print(f"    {len(train_feats)} samples, {train_feats.shape[1]} features")

    # Mix in extra data (e.g. DAgger episodes)
    if args.data_extra is not None:
        extra_dir = Path(args.data_extra)
        if extra_dir.exists():
            print(f"  Loading extra data from {args.data_extra}...")
            extra_feats, extra_targets = load_episodes(args.data_extra)
            if args.extra_weight < 1.0:
                n_keep = max(1, int(len(extra_feats) * args.extra_weight))
                idx = np.random.default_rng(args.seed).choice(
                    len(extra_feats), n_keep, replace=False)
                extra_feats = extra_feats[idx]
                extra_targets = extra_targets[idx]
            print(f"    {len(extra_feats)} extra samples (weight={args.extra_weight})")
            n_orig = len(train_feats)
            train_feats = np.vstack([train_feats, extra_feats])
            train_targets = np.vstack([train_targets, extra_targets])
            pct = len(extra_feats) / len(train_feats) * 100
            print(f"    Mixed: {n_orig} orig + {len(extra_feats)} extra"
                  f" = {len(train_feats)} total ({pct:.0f}% extra)")

    val_feats, val_targets = None, None
    val_dir = Path(args.val)
    if val_dir.exists():
        print(f"  Loading validation data from {args.val}...")
        val_feats, val_targets = load_episodes(args.val)
        print(f"    {len(val_feats)} samples")

    # Train
    print(f"\n  Training (epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr})...")
    model, stats, history = train(
        train_feats, train_targets,
        val_feats=val_feats, val_targets=val_targets,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, seed=args.seed,
    )

    # Save
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "nn_3d.weights.npz"
    stats_path = out_dir / "feat_stats_3d.json"

    model.save(model_path)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n  Saved model to {model_path}")
    print(f"  Saved stats to {stats_path}")
    print(f"  Final: train={history['train_loss'][-1]:.6f}  val={history['val_loss'][-1]:.6f}")

    # Quick sanity: predict on a zero-feature input
    zero_feat = np.zeros((1, FEAT_DIM))
    zero_norm = (zero_feat - np.array(stats['mean'])) / np.array(stats['std'])
    y_zero, _ = model.forward(zero_norm)
    print(f"  NN(zeros): [{y_zero[0,0]:.4f}, {y_zero[0,1]:.4f}, {y_zero[0,2]:.4f}] m/s²")

    print("\nDone.")


if __name__ == "__main__":
    main()
