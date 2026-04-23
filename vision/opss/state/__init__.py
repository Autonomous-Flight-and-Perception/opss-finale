"""
OPSS State Estimation Module

Provides multiple state estimation approaches:

1. Basic Kalman Filter (kalman.py)
   - Multi-object tracking
   - Simple constant-velocity model
   - Fast, suitable for real-time

2. UKF-NN (ukf_nn/)
   - Unscented Kalman Filter with Neural Network correction
   - 3D: 6D state, 3D measurement, 3D NN correction
   - Handles unmodeled dynamics (wind, drag, thrust lag)
   - Higher accuracy, requires trained model

3. UKF-NN Tracker (ukf_nn_tracker.py)
   - Multi-object wrapper for UKF-NN
   - Drop-in replacement for basic Kalman tracker
"""

from .kalman import (
    MultiObjectKalmanFilter,
    KalmanTracker,
    ObjectState,
    create_tracker
)

__all__ = [
    # Basic Kalman
    "MultiObjectKalmanFilter",
    "KalmanTracker",
    "ObjectState",
    "create_tracker",
]

# Optional: UKF-NN (requires torch)
try:
    from .ukf_nn import UKF3D, DeltaAccelNN3D, FeatureExtractor3D
    from .ukf_nn import UKF, AccelCorrectionNet
    from .ukf_nn_tracker import (
        MultiObjectUKFNN, UKFNNTracker, create_ukf_nn_tracker, CameraIntrinsics,
    )
    __all__.extend([
        "UKF3D",
        "DeltaAccelNN3D",
        "FeatureExtractor3D",
        "UKF",
        "AccelCorrectionNet",
        "MultiObjectUKFNN",
        "UKFNNTracker",
        "create_ukf_nn_tracker",
        "CameraIntrinsics",
    ])
except ImportError:
    pass  # torch not available
