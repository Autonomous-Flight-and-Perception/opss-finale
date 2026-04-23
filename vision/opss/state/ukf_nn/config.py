"""
Single source of truth for all hyperparameters and constants.

3D UKF-NN: 6D state [x, y, z, vx, vy, vz], 3D measurement, 3D NN correction.

MEASUREMENT CONVENTION:
    The UKF state and measurements are ALL in meters.
    Pixel-to-meter conversion happens at the tracker boundary
    (ukf_nn_tracker.py) using camera intrinsics and depth.
    The pinhole model is:
        x_m = (u_px - cx) * depth / fx
        y_m = (v_px - cy) * depth / fy
        z_m = depth
"""
import numpy as np

# --- Dimensions ---
STATE_DIM = 6        # [x, y, z, vx, vy, vz]
MEAS_DIM = 3         # [x, y, z]
ACCEL_DIM = 3        # [delta_ax, delta_ay, delta_az]
FEAT_DIM = 15        # 5 features per axis x 3 axes (v, r_prev, |r_prev|, ||v||, v_error)

# --- Timing ---
DT = 1.0 / 30.0      # 30 FPS camera rate

# --- Physics ---
GRAVITY = np.array([0.0, 0.0, -9.81])  # m/s^2, world frame (z-up)
A_HOVER = np.array([0.0, 0.0, +9.81])  # hover thrust (cancels gravity for multirotor)

# --- Process noise (6x6 diagonal) ---
# Position variance small, velocity variance larger.  All meters / (m/s).
Q = np.diag([1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2])

# --- Measurement noise (3x3 diagonal) ---
# Calibrated via calibrate_filter.py sweep (2026-02-06).
# Lateral axes (x,y) need 7x base R to match observed pixel→meter noise
# at typical camera angles.  Depth axis (z) unchanged.
# Pre-calibration: NIS ~37% (overconfident).  Post: ~68% (target 65%).
R = np.diag([0.07, 0.07, 0.01])

# --- Camera intrinsics (for pixel-to-meter conversion at tracker boundary) ---
CAMERA_FX = 600.0    # focal length x (pixels)
CAMERA_FY = 600.0    # focal length y (pixels)
CAMERA_CX = 320.0    # principal point x (pixels)
CAMERA_CY = 240.0    # principal point y (pixels)
INTRINSIC_WARN_TOLERANCE = 0.05  # warn if hardware intrinsics differ by >5%

# --- Sigma point parameters (Merwe scaled) ---
ALPHA = 1e-3
BETA = 2.0
KAPPA = 0.0

# --- Neural network parameters ---
NN_INPUT_DIM = 15
NN_OUTPUT_DIM = 3
NN_HIDDEN = 32
A_MAX = 15.0         # Maximum acceleration correction (m/s^2)
NN_LR = 1e-3
NN_WEIGHT_DECAY = 1e-6
NN_BATCH_SIZE = 1024
NN_EPOCHS = 200
NN_EARLY_STOP_PATIENCE = 15
NN_EARLY_STOP_MIN_DELTA = 1e-5

# --- Training data ---
TRAIN_VAL_SPLIT = 0.8
TRAIN_DATA_DIR = 'data/trajectories'
MODEL_DIR = 'models'

# --- Chi-squared 95th percentile by DOF ---
# Used for NIS gating. DOF = number of *observed* measurement dimensions.
CHI2_95_TABLE = {1: 3.841, 2: 5.991, 3: 7.815}
CHI2_95 = CHI2_95_TABLE[3]  # default for full 3D observation

# --- Acceptance criteria ---
RMSE_POS_THRESHOLD = 0.95   # meters (3D position RMSE)
RMSE_VEL_THRESHOLD = 1.15   # m/s (3D velocity RMSE)
NIS_INBOUND_THRESHOLD = 0.65  # 65% of steps within bounds

# --- Runtime robustness ---
ADAPTIVE_R_MULTIPLIER = 4.0
INNOVATION_THRESHOLD_SIGMA = 5.0
MAX_SCENE_RADIUS = 200.0   # meters — hard sanity bound on converted position
CHOLESKY_JITTER = 1e-9
CHOLESKY_JITTER_RETRY = 1e-7

# --- Covariance bounds ---
P_MAX_DIAG = 100.0  # Max allowed P diagonal element before reset

# --- Track limits ---
MAX_TRACKS = 50  # Maximum concurrent tracks (prevents unbounded growth)

# --- Paths ---
MODEL_PATH = 'models/nn_3d.weights.npz'
STATS_PATH = 'models/feat_stats_3d.json'

# --- Legacy 1D aliases (kept for backward compat with old scripts) ---
MASS = 1.0
T_SEC = 30.0
RMSE_P_THRESHOLD = RMSE_POS_THRESHOLD
RMSE_V_THRESHOLD = RMSE_VEL_THRESHOLD
