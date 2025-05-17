"""
Data Processing Utilities

This module provides functions for processing biomechanical data, particularly
related to gait analysis and movement data.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import pandas as pd


def normalize_gait_cycle(
    data: Union[np.ndarray, pd.DataFrame],
    events: Dict[str, Union[int, float]],
    n_points: int = 101,
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Normalize data to a gait cycle (0-100%).

    Parameters
    ----------
    data : array-like or DataFrame
        Data to normalize. If array-like, the first dimension should be time.
        If DataFrame, the index should be time.
    events : dict
        Dictionary containing event indices or times. Must contain at least
        'foot_strike_1' and 'foot_strike_2' keys.
    n_points : int, optional
        Number of points in the normalized gait cycle. Default is 101 (0-100%).

    Returns
    -------
    normalized_data : array-like or DataFrame
        Data normalized to the gait cycle.
    """
    # Extract event indices
    start_idx = events["foot_strike_1"]
    end_idx = events["foot_strike_2"]
    
    # Handle DataFrame input
    if isinstance(data, pd.DataFrame):
        # Extract the data between the events
        cycle_data = data.iloc[start_idx:end_idx + 1]
        
        # Create a new index representing the gait cycle percentage
        old_idx = np.linspace(0, 100, len(cycle_data))
        new_idx = np.linspace(0, 100, n_points)
        
        # Interpolate to get the normalized data
        normalized_data = pd.DataFrame(index=new_idx)
        for col in cycle_data.columns:
            normalized_data[col] = np.interp(new_idx, old_idx, cycle_data[col].values)
        
        return normalized_data
    
    # Handle array-like input
    else:
        # Extract the data between the events
        cycle_data = data[start_idx:end_idx + 1]
        
        # Create arrays representing the original and target time points
        old_idx = np.linspace(0, 100, cycle_data.shape[0])
        new_idx = np.linspace(0, 100, n_points)
        
        # Initialize the output array
        if cycle_data.ndim == 1:
            normalized_data = np.zeros(n_points)
            normalized_data = np.interp(new_idx, old_idx, cycle_data)
        else:
            normalized_data = np.zeros((n_points,) + cycle_data.shape[1:])
            # Interpolate each dimension
            for i in range(cycle_data.shape[1]):
                normalized_data[:, i] = np.interp(new_idx, old_idx, cycle_data[:, i])
        
        return normalized_data


def detect_gait_events(
    marker_data: Union[np.ndarray, pd.DataFrame],
    force_data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    method: str = "coordinate",
    threshold: float = 20.0,
    min_samples_between: int = 50,
) -> Dict[str, int]:
    """
    Detect gait events (foot strikes and toe-offs) from marker or force data.

    Parameters
    ----------
    marker_data : array-like or DataFrame
        Marker trajectory data. If using the 'coordinate' method, this should
        contain the heel and toe marker positions.
    force_data : array-like or DataFrame, optional
        Vertical ground reaction force data. Required if method is 'force'.
    method : str, optional
        Method to use for event detection. Options are:
        - 'coordinate': Based on marker positions
        - 'force': Based on force plate data
        Default is 'coordinate'.
    threshold : float, optional
        Threshold for event detection. For 'force' method, this is the force
        threshold in Newtons. For 'coordinate' method, this is the velocity
        threshold in mm/s. Default is 20.0.
    min_samples_between : int, optional
        Minimum number of samples between consecutive events. Default is 50.

    Returns
    -------
    events : dict
        Dictionary containing event indices:
        - 'foot_strike_1': First foot strike
        - 'foot_strike_2': Second foot strike
        - 'toe_off': Toe off
    """
    events = {}
    
    if method == "force" and force_data is not None:
        # Convert to numpy array if DataFrame
        if isinstance(force_data, pd.DataFrame):
            force_values = force_data.values
        else:
            force_values = force_data
        
        # Detect foot strikes and toe-offs based on force threshold
        # A foot strike occurs when the force exceeds the threshold
        # A toe-off occurs when the force drops below the threshold
        force_above_threshold = force_values > threshold
        
        # Find transitions (0->1 for foot strike, 1->0 for toe-off)
        foot_strikes = np.where(np.diff(force_above_threshold.astype(int)) > 0)[0] + 1
        toe_offs = np.where(np.diff(force_above_threshold.astype(int)) < 0)[0] + 1
        
        # Filter events that are too close together
        if len(foot_strikes) > 1:
            valid_strikes = [foot_strikes[0]]
            for i in range(1, len(foot_strikes)):
                if foot_strikes[i] - valid_strikes[-1] >= min_samples_between:
                    valid_strikes.append(foot_strikes[i])
            foot_strikes = np.array(valid_strikes)
        
        # Store the events
        if len(foot_strikes) >= 2:
            events["foot_strike_1"] = int(foot_strikes[0])
            events["foot_strike_2"] = int(foot_strikes[1])
        
        if len(toe_offs) >= 1:
            events["toe_off"] = int(toe_offs[0])
    
    elif method == "coordinate":
        # Assuming marker_data contains heel and toe marker positions
        # Convert to numpy array if DataFrame
        if isinstance(marker_data, pd.DataFrame):
            # Assuming columns are named 'heel_z' and 'toe_z' for vertical positions
            heel_z = marker_data["heel_z"].values
            toe_z = marker_data["toe_z"].values
        else:
            # Assuming marker_data is an array with shape (n_frames, n_markers, 3)
            # and heel is index 0, toe is index 1
            heel_z = marker_data[:, 0, 2]  # Z-coordinate of heel marker
            toe_z = marker_data[:, 1, 2]   # Z-coordinate of toe marker
        
        # Compute velocities
        heel_vel = np.diff(heel_z)
        toe_vel = np.diff(toe_z)
        
        # Detect foot strikes (heel velocity changes from negative to near-zero)
        heel_vel_below_threshold = np.abs(heel_vel) < threshold
        foot_strikes = np.where(np.diff(heel_vel_below_threshold.astype(int)) > 0)[0] + 1
        
        # Detect toe-offs (toe velocity becomes positive)
        toe_offs = np.where(np.diff(toe_vel > 0) > 0)[0] + 1
        
        # Filter events that are too close together
        if len(foot_strikes) > 1:
            valid_strikes = [foot_strikes[0]]
            for i in range(1, len(foot_strikes)):
                if foot_strikes[i] - valid_strikes[-1] >= min_samples_between:
                    valid_strikes.append(foot_strikes[i])
            foot_strikes = np.array(valid_strikes)
        
        # Store the events
        if len(foot_strikes) >= 2:
            events["foot_strike_1"] = int(foot_strikes[0])
            events["foot_strike_2"] = int(foot_strikes[1])
        
        if len(toe_offs) >= 1:
            events["toe_off"] = int(toe_offs[0])
    
    return events


def compute_gait_metrics(
    joint_angles: pd.DataFrame,
    grf_data: pd.DataFrame,
    events: Dict[str, int],
) -> Dict[str, float]:
    """
    Compute common gait metrics from joint angles and ground reaction forces.

    Parameters
    ----------
    joint_angles : DataFrame
        Joint angle data with columns for hip, knee, and ankle angles.
    grf_data : DataFrame
        Ground reaction force data with columns for vertical, anterior-posterior,
        and medial-lateral forces.
    events : dict
        Dictionary containing event indices.

    Returns
    -------
    metrics : dict
        Dictionary containing computed gait metrics:
        - 'stride_length': Length of one stride in meters
        - 'cadence': Steps per minute
        - 'walking_speed': Walking speed in m/s
        - 'stance_phase_percent': Percentage of gait cycle in stance phase
        - 'swing_phase_percent': Percentage of gait cycle in swing phase
        - 'peak_knee_flexion': Peak knee flexion angle in degrees
        - 'peak_hip_extension': Peak hip extension angle in degrees
        - 'peak_ankle_plantarflexion': Peak ankle plantarflexion in degrees
        - 'peak_vgrf': Peak vertical ground reaction force (% body weight)
    """
    metrics = {}
    
    # Calculate temporal-spatial parameters
    stride_time = events["foot_strike_2"] - events["foot_strike_1"]
    stance_time = events["toe_off"] - events["foot_strike_1"]
    
    metrics["stance_phase_percent"] = (stance_time / stride_time) * 100
    metrics["swing_phase_percent"] = 100 - metrics["stance_phase_percent"]
    
    # Calculate joint angle metrics
    if "knee_angle" in joint_angles.columns:
        metrics["peak_knee_flexion"] = joint_angles["knee_angle"].max()
    
    if "hip_angle" in joint_angles.columns:
        metrics["peak_hip_extension"] = -joint_angles["hip_angle"].min()  # Extension is negative
    
    if "ankle_angle" in joint_angles.columns:
        metrics["peak_ankle_plantarflexion"] = -joint_angles["ankle_angle"].min()  # Plantarflexion is negative
    
    # Calculate GRF metrics
    if "vertical_force" in grf_data.columns:
        metrics["peak_vgrf"] = grf_data["vertical_force"].max()
    
    # Additional metrics can be calculated here
    
    return metrics 