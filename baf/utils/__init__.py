"""
BAF Utilities Module
==================

This module provides utility functions for:
- Data processing
- File I/O
- Signal processing
- Statistical analysis
- Common biomechanical calculations
"""

from .data_processing import normalize_gait_cycle, detect_gait_events, compute_gait_metrics

# Import utility functions as they are developed
# from .signal_processing import filter_signal, compute_derivatives
# from .file_io import load_c3d, save_results
# from .statistics import compute_mean_std, compare_conditions

__all__ = [
    "normalize_gait_cycle",
    "detect_gait_events",
    "compute_gait_metrics",
    # "filter_signal",
    # "compute_derivatives",
    # "load_c3d",
    # "save_results",
    # "compute_mean_std",
    # "compare_conditions",
] 