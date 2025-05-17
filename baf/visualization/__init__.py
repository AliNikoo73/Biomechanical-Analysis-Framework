"""
BAF Visualization Module
======================

This module provides visualization tools for biomechanical data:
- Joint angle plots
- Ground reaction force visualization
- Muscle activation visualization
- 3D motion visualization
- Comparative analysis plots
"""

# Import visualization components as they are developed
# from .motion_viewer import MotionViewer
from .joint_plots import JointPlotter
from .grf_plots import GRFPlotter
from .emg_plots import EMGPlotter
from .comparative_plots import ComparativePlotter

# Import future visualization components as they are developed
# from .motion_viewer import MotionViewer

__all__ = [
    # "MotionViewer",
    "JointPlotter",
    "GRFPlotter",
    "EMGPlotter",
    "ComparativePlotter",
] 