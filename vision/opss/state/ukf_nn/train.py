"""
Training script for the 3D UKF-NN acceleration correction network.

Usage:
    python -m opss.state.ukf_nn.train --data trajectories/ --output models/

Accepts data in two formats (auto-detected from NPZ keys):

  1. Trajectory files (original):
     - NPZ: keys 't' (N,), 'x' (N,3), 'v' (N,3)
     - CSV: columns t, x, y, z, vx, vy, vz
     - Features are reconstructed offline (simplified innovation model)

  2. Logged-feature episodes (from generate_training_episodes.py):
     - NPZ: keys 'feat' (N,12), 'target' (N,3), 'valid' (N,)
     - Features logged from the actual tracker's FeatureExtractor3D
     - Training distribution matches inference exactly
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from . import config as cfg
from .nn_model import DeltaAccelNN3D, count_parameters
from .features import (
    extract_training_features_3d,
    compute_normalization_stats,
    normalize_features,
    save_stats,
)

logger = logging.getLogger(__name__)


def load_trajectory(path):
    """
    Load a single trajectory file.

    Returns:
        dict with keys 't' (N,), 'x' (N,3), 'v' (N,3)
    """
    path = Path(path)

    if path.suffix == '.npz':
        data = np.load(path)
        return {
            't': np.asarray(data['t'], dtype=np.float64),
            'x': np.asarray(data['x'], dtype=np.float64),
            'v': np.asarray(data['v'], dtype=np.float64),
        }

    elif path.suffix == '.csv':
        import csv
        rows = []
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        n = len(rows)
        t = np.zeros(n)
        x = np.zeros((n, 3))
        v = np.zeros((n, 3))
        for i, row in enumerate(rows):
            t[i] = float(row['t'])
            x[i] = [float(row['x']), float(row['y']), float(row['z'])]
            v[i] = [float(row['vx']), float(row['vy']), float(row['vz'])]

        return {'t': t, 'x': x, 'v': v}

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def load_all_trajectories(data_dir):
    """Load all trajectory files from a directory."""
    data_dir = Path(data_dir)
    trajectories = []

    for ext in ('*.npz', '*.csv'):
        for path in sorted(data_dir.glob(ext)):
            try:
                traj = load_trajectory(path)
                trajectories.append(traj)
                logger.info("Loaded %s (%d timesteps)", path.name, len(traj['t']))
            except Exception as e:
                logger.warning("Failed to load %s: %s", path, e)

    return trajectories


def _detect_data_format(data_dir):
    """
    Auto-detect whether data_dir contains trajectory files or logged episodes.

    Returns 'episodes' if first NPZ has 'feat' and 'target' keys,
    'trajectories' otherwise.
    """
    data_dir = Path(data_dir)
    for path in sorted(data_dir.glob('*.npz')):
        data = np.load(path, allow_pickle=True)
        keys = set(data.keys())
        if 'feat' in keys and 'target' in keys and 'valid' in keys:
            return 'episodes'
        if 'x' in keys and 'v' in keys:
            return 'trajectories'
        break
    # Fall back to trajectory format (also handles CSV-only dirs)
    return 'trajectories'


def load_all_episodes(data_dir):
    """
    Load all logged-feature episode NPZs from a directory.

    Returns:
        features: (N_total, 12) array — pre-built features from tracker runtime
        targets:  (N_total, 3) array — NN training targets
    """
    data_dir = Path(data_dir)
    all_features = []
    all_targets = []

    for path in sorted(data_dir.glob('*.npz')):
        try:
            data = np.load(path)
            feat = data['feat']      # (N, 15)
            target = data['target']  # (N, 3)
            valid = data['valid']    # (N,) bool

            # Keep only valid steps that also have valid targets
            mask = valid & ~np.isnan(target[:, 0]) & ~np.isnan(feat[:, 0])
            if mask.sum() == 0:
                continue

            all_features.append(feat[mask])
            all_targets.append(target[mask])
            logger.info("Loaded episode %s (%d valid samples)", path.name, mask.sum())
        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)

    if not all_features:
        raise ValueError("No valid samples in episode files")

    return np.vstack(all_features), np.vstack(all_targets)


def build_dataset(trajectories):
    """
    Build training features and targets from trajectories.

    Returns:
        features: (N_total, 15) array
        targets:  (N_total, 3) array
    """
    all_features = []
    all_targets = []

    for traj in trajectories:
        t = traj['t']
        # Compute dt from timestamps
        if len(t) > 1:
            dt = np.median(np.diff(t))
        else:
            dt = cfg.DT

        feats, tgts = extract_training_features_3d(traj['x'], traj['v'], dt=dt)
        if len(feats) > 0:
            all_features.append(feats)
            all_targets.append(tgts)

    if not all_features:
        raise ValueError("No valid training samples extracted")

    return np.vstack(all_features), np.vstack(all_targets)


def train(
    features,
    targets,
    val_split=None,
    epochs=None,
    batch_size=None,
    lr=None,
    patience=None,
):
    """
    Train the DeltaAccelNN3D model.

    Returns:
        model: trained DeltaAccelNN3D
        stats: normalization stats dict
        history: dict with 'train_loss' and 'val_loss' lists
    """
    val_split = val_split or cfg.TRAIN_VAL_SPLIT
    epochs = epochs or cfg.NN_EPOCHS
    batch_size = batch_size or cfg.NN_BATCH_SIZE
    lr = lr or cfg.NN_LR
    patience = patience or cfg.NN_EARLY_STOP_PATIENCE

    # Compute normalization stats
    stats = compute_normalization_stats([features])
    features_norm = normalize_features(features, stats)

    # Train/val split
    n = len(features_norm)
    n_train = int(n * val_split)
    indices = np.random.permutation(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    X_train = torch.FloatTensor(features_norm[train_idx])
    y_train = torch.FloatTensor(targets[train_idx])
    X_val = torch.FloatTensor(features_norm[val_idx])
    y_val = torch.FloatTensor(targets[val_idx])

    logger.info("Training samples: %d, validation samples: %d", len(X_train), len(X_val))

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Model
    model = DeltaAccelNN3D()
    logger.info("Model parameters: %d", count_parameters(model))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=cfg.NN_WEIGHT_DECAY
    )
    loss_fn = nn.MSELoss()

    # Training loop with early stopping
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, y_val).item()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                "Epoch %3d/%d  train_loss=%.6f  val_loss=%.6f",
                epoch + 1, epochs, train_loss, val_loss,
            )

        # Early stopping
        if val_loss < best_val_loss - cfg.NN_EARLY_STOP_MIN_DELTA:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info("Early stopping at epoch %d (best val_loss=%.6f)", epoch + 1, best_val_loss)
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model, stats, history


def main():
    parser = argparse.ArgumentParser(description="Train 3D UKF-NN acceleration correction model")
    parser.add_argument("--data", required=True, help="Directory containing trajectory files (NPZ/CSV)")
    parser.add_argument("--output", default="models", help="Output directory for model and stats")
    parser.add_argument("--epochs", type=int, default=None, help=f"Training epochs (default: {cfg.NN_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=None, help=f"Batch size (default: {cfg.NN_BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=None, help=f"Learning rate (default: {cfg.NN_LR})")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load data — auto-detect format
    data_format = _detect_data_format(args.data)
    logger.info("Detected data format: %s", data_format)

    if data_format == 'episodes':
        logger.info("Loading logged-feature episodes from %s", args.data)
        features, targets = load_all_episodes(args.data)
    else:
        logger.info("Loading trajectories from %s", args.data)
        trajectories = load_all_trajectories(args.data)
        if not trajectories:
            logger.error("No trajectory files found in %s", args.data)
            sys.exit(1)
        logger.info("Loaded %d trajectories", len(trajectories))
        features, targets = build_dataset(trajectories)

    logger.info("Dataset: %d samples, %d features, %d targets", len(features), features.shape[1], targets.shape[1])

    # Train
    model, stats, history = train(
        features, targets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "nn_3d.pt"
    stats_path = output_dir / "feat_stats_3d.json"

    torch.save(model.state_dict(), model_path)
    save_stats(stats, stats_path)

    logger.info("Saved model to %s", model_path)
    logger.info("Saved stats to %s", stats_path)
    logger.info("Final train_loss=%.6f  val_loss=%.6f", history['train_loss'][-1], history['val_loss'][-1])


if __name__ == "__main__":
    main()
