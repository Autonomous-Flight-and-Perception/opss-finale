"""
Neural network models for acceleration correction.

DeltaAccelNN3D: 3D model (15 inputs -> 3 outputs) for cross-axis coupling.
DeltaAccelNN: Legacy 1D model (5 inputs -> 1 output), kept for backward compat.
"""
import torch
import torch.nn as nn
from . import config as cfg


class DeltaAccelNN3D(nn.Module):
    """
    3D acceleration correction network.

    Architecture: Linear(15,32) -> Tanh -> Linear(32,3) -> Tanh -> * A_MAX
    Output: [delta_ax, delta_ay, delta_az] clamped to [-A_MAX, +A_MAX] m/s^2

    ~611 parameters — small enough for Jetson real-time inference.
    """

    def __init__(self, input_dim=None, hidden_dim=None, output_dim=None, a_max=None):
        super().__init__()
        input_dim = input_dim or cfg.NN_INPUT_DIM
        hidden_dim = hidden_dim or cfg.NN_HIDDEN
        output_dim = output_dim or cfg.NN_OUTPUT_DIM
        self.a_max = a_max if a_max is not None else cfg.A_MAX

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 15) or (15,) tensor of normalized features

        Returns:
            delta_a: (batch, 3) or (3,) predicted acceleration corrections
        """
        return self.net(x) * self.a_max

    def predict_numpy(self, features_normalized):
        """
        Convenience method for numpy input (for UKF integration).

        Args:
            features_normalized: (15,) numpy array

        Returns:
            delta_a: (3,) numpy array
        """
        with torch.no_grad():
            x = torch.FloatTensor(features_normalized).unsqueeze(0)
            out = self.forward(x)
            return out.squeeze(0).numpy()


class DeltaAccelNN(nn.Module):
    """
    Legacy 1D acceleration correction network (deprecated).

    Use DeltaAccelNN3D for new code. This class is kept for loading
    old 1D-trained models only.

    Architecture: Linear(5,hidden) -> Tanh -> Linear(hidden,1) -> Tanh -> * a_max
    """

    def __init__(self, input_dim=5, hidden_dim=None, a_max=None):
        super().__init__()
        hidden_dim = hidden_dim or cfg.NN_HIDDEN
        self.a_max = a_max if a_max is not None else cfg.A_MAX

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x) * self.a_max

    def predict_numpy(self, features_normalized):
        with torch.no_grad():
            x = torch.FloatTensor(features_normalized).unsqueeze(0)
            out = self.forward(x)
            return out.item()


# Backward-compat alias
AccelCorrectionNet = DeltaAccelNN3D


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
