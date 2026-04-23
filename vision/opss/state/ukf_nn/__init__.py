"""
UKF-NN: Unscented Kalman Filter with Neural Network Correction

3D system: 6D state [x,y,z,vx,vy,vz], 3D measurement, 3D NN correction.

Originally from: https://github.com/Autonomous-Flight-and-Perception/UKF-NN_MVP
"""

from .ukf import UKF3D, UKF
from .features import FeatureExtractor3D, extract_training_features_3d

# nn_model requires torch — make import optional for Jetson deployment
try:
    from .nn_model import DeltaAccelNN3D, DeltaAccelNN, AccelCorrectionNet, count_parameters
except ImportError:
    DeltaAccelNN3D = None
    DeltaAccelNN = None
    AccelCorrectionNet = None
    count_parameters = None
from .config import (
    STATE_DIM, MEAS_DIM, ACCEL_DIM, FEAT_DIM,
    DT, Q, R, GRAVITY, A_HOVER,
    NN_INPUT_DIM, NN_OUTPUT_DIM, NN_HIDDEN, A_MAX,
    RMSE_POS_THRESHOLD, RMSE_VEL_THRESHOLD,
    CHI2_95, CHI2_95_TABLE,
    CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY,
    # Legacy aliases
    T_SEC, MASS,
    RMSE_P_THRESHOLD, RMSE_V_THRESHOLD,
)

__all__ = [
    # 3D (primary)
    "UKF3D",
    "DeltaAccelNN3D",
    "FeatureExtractor3D",
    "extract_training_features_3d",
    # Legacy 1D
    "UKF",
    "DeltaAccelNN",
    "AccelCorrectionNet",
    # Config
    "STATE_DIM", "MEAS_DIM", "ACCEL_DIM", "FEAT_DIM",
    "DT", "T_SEC", "MASS",
    "Q", "R",
    "NN_INPUT_DIM", "NN_OUTPUT_DIM", "NN_HIDDEN", "A_MAX",
    "CHI2_95",
    # Utilities
    "count_parameters",
]
