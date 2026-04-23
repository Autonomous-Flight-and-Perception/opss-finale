"""
OPSS Physics Module

Provides physics validation for tracked object states.
Based on the b2 physics engine with analytical and numerical trajectory computation.
"""

from .validator import PhysicsValidator, ValidationResult, create_validator

__all__ = [
    "PhysicsValidator",
    "ValidationResult",
    "create_validator"
]
