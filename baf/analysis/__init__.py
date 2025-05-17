"""
BAF Analysis Module
================

This module provides tools for biomechanical analysis:
- Kinematics analysis (joint angles, segment positions, etc.)
- Dynamics analysis (forces, moments, power, etc.)
- Muscle analysis (activations, forces, lengths, etc.)
"""

# Import analysis components
from . import kinematics
from . import dynamics
from . import muscle_analysis

__all__ = [
    "kinematics",
    "dynamics",
    "muscle_analysis",
] 