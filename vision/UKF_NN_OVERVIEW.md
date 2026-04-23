# UKF-NN State Estimation System

## What It Does

This is a **3D Unscented Kalman Filter augmented with a small neural network** (UKF-NN) for tracking drone positions in real time. The UKF handles the core physics (gravity, kinematics), while the NN learns residual forces the physics model can't capture — wind, aerodynamic drag, thrust lag, etc.

The system takes noisy camera+depth measurements (pixels + depth in meters) and produces smoothed, predicted 3D position and velocity estimates for one or more tracked objects.

## How It Works

**State**: 6D — `[x, y, z, vx, vy, vz]` (all in meters / m/s)

**Predict step**:
1. The NN evaluates on the current state to predict a residual acceleration correction (`delta_a`, up to +/-5 m/s²)
2. 13 sigma points are generated and propagated through the process model: `v_next = v + dt*(gravity + delta_a)`, `pos_next = pos + dt*v_next` (semi-implicit Euler)
3. Predicted mean and covariance are reconstructed from the sigma points

**Update step**:
1. Measurement arrives as `[x, y, z]` in meters (converted from pixels+depth at the tracker boundary)
2. Supports **partial observations** — any axis can be NaN and the filter gracefully skips it
3. **Outlier gating**: if the Mahalanobis distance (NIS) exceeds 25 (5-sigma), measurement noise is inflated 4x instead of rejecting the measurement entirely
4. Kalman gain computed, state and covariance updated

**NN architecture**: `Linear(12, 32) -> Tanh -> Linear(32, 3) -> Tanh -> * 5.0` (~515 parameters). Input is a 12D feature vector: per-axis velocity, previous innovation, |previous innovation|, and total speed.

**Training**: The NN is trained offline on simulated drone flight data (waypoint navigation with wind). Targets are the residual acceleration after subtracting gravity. MSE loss, Adam optimizer, early stopping.

## File Map

### Core Filter

| File | What's in it |
|------|-------------|
| `opss/state/ukf_nn/config.py` | All constants: dimensions, noise matrices Q/R, sigma point params, NN hyperparameters, thresholds, camera intrinsics |
| `opss/state/ukf_nn/ukf.py` | **`UKF3D`** — the main filter. `predict()` and `update()` methods, sigma point generation, Cholesky fallback, partial observation handling, outlier gating |
| `opss/state/ukf_nn/nn_model.py` | **`DeltaAccelNN3D`** — the neural network. Forward pass + `predict_numpy()` convenience for UKF integration |
| `opss/state/ukf_nn/features.py` | **`FeatureExtractor3D`** — builds the 12D feature vector. `build_features()` (pure), `commit_innovation()` (mutates). Also `extract_training_features_3d()` for offline data |
| `opss/state/ukf_nn/metrics.py` | Evaluation: 3D RMSE (position/velocity/per-axis), NIS inbound rate, acceptance criteria |
| `opss/state/ukf_nn/train.py` | Training CLI: loads trajectories, builds dataset, trains NN, saves model + normalization stats |

### Tracker Layer

| File | What's in it |
|------|-------------|
| `opss/state/ukf_nn_tracker.py` | **`UKFNNTracker`** (single object) and **`MultiObjectUKFNN`** (multi-object with association). Handles pixel-to-meter conversion via `CameraIntrinsics`, depth validation, model loading, and metadata guards against stale models |
| `opss/state/kalman.py` | **`MultiObjectKalmanFilter`** — the simpler constant-velocity Kalman baseline (no NN, pixel-space) |

### Data Generation and Training

| File | What's in it |
|------|-------------|
| `generate_training_data.py` | Simulates realistic drone flights (waypoints, wind via Ornstein-Uhlenbeck, hover, first-order velocity lag). Outputs NPZ for training + CSV for Blender visualization |
| `models/nn_3d.pt` | Trained NN weights |
| `models/feat_stats_3d.json` | Normalization stats (mean/std per feature) |

### Tests

| File | What's in it |
|------|-------------|
| `tests/test_ukf_nn.py` | 25 tests covering UKF3D predict/update, partial obs, outlier gating, feature extraction purity, training feature consistency, depth rejection, metadata guards, config consistency |
| `tests/conftest.py` | Stubs torch imports so tests run on the Jetson without CUDA |

## Key Design Decisions

- **Gravity is baked into the process model** — the NN only learns the residual (drag, wind, thrust lag), which keeps targets small and training stable
- **Single NN for all 3 axes** (not per-axis) — captures cross-axis coupling (e.g., velocity-dependent drag)
- **Features are predict-time only** — no current-innovation features, so train and inference are consistent
- **All math in meters** — pixel-to-meter conversion happens once at the tracker boundary, not inside the filter
- **Depth=0 detections are rejected** — prevents pixel-space values from leaking into meter-space state

## Retraining the NN

If the feature definition or NN architecture changes, existing models become incompatible. The loader has metadata guards that will raise an error if you try to load a stale model. To retrain:

```bash
python generate_training_data.py
python -m opss.state.ukf_nn.train --data data/generated/training/train
```
