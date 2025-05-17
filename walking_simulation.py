#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Walking Simulation Example

This script demonstrates how to generate walking simulation data using OpenSim
and visualize it using the Biomechanical Analysis Framework (BAF).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import opensim as osim

# Import BAF modules for visualization
from baf.utils.data_processing import normalize_gait_cycle, detect_gait_events, compute_gait_metrics
from baf.visualization.joint_plots import JointPlotter
from baf.visualization.grf_plots import GRFPlotter
from baf.visualization.emg_plots import EMGPlotter
from baf.visualization.comparative_plots import ComparativePlotter


def generate_walking_data(num_frames=100, noise_level=0.05):
    """Generate synthetic walking data."""
    # Time vector (0 to 1 second)
    time = np.linspace(0, 1, num_frames)
    
    # Generate hip angle data (flexion/extension)
    # Hip flexes at initial contact, extends during stance, flexes during swing
    hip_angle = 30 * np.sin(2 * np.pi * time - np.pi/6) - 5
    
    # Generate knee angle data (flexion/extension)
    # Knee flexes slightly at initial contact, extends in mid-stance, flexes in swing
    knee_angle = 5 + 60 * (0.5 - 0.5 * np.cos(2 * np.pi * time + np.pi/4))
    knee_angle -= 20 * np.exp(-((time - 0.3) ** 2) / 0.02)  # Extension in mid-stance
    
    # Generate ankle angle data (dorsiflexion/plantarflexion)
    # Dorsiflexion is positive, plantarflexion is negative
    ankle_angle = 10 * np.sin(2 * np.pi * time + np.pi/2)
    ankle_angle -= 15 * np.exp(-((time - 0.6) ** 2) / 0.01)  # Plantarflexion at push-off
    
    # Add some noise
    hip_angle += noise_level * np.random.randn(num_frames) * np.max(np.abs(hip_angle))
    knee_angle += noise_level * np.random.randn(num_frames) * np.max(np.abs(knee_angle))
    ankle_angle += noise_level * np.random.randn(num_frames) * np.max(np.abs(ankle_angle))
    
    # Generate ground reaction force data
    # Vertical GRF (double-peak pattern)
    vgrf = np.zeros(num_frames)
    vgrf[:int(0.6*num_frames)] = 800 * (
        1.2 * np.exp(-((time[:int(0.6*num_frames)] - 0.15) ** 2) / 0.01) + 
        1.0 * np.exp(-((time[:int(0.6*num_frames)] - 0.45) ** 2) / 0.01)
    )
    vgrf = np.maximum(vgrf, 0)  # Force can't be negative
    
    # Anterior-posterior GRF (braking then propulsion)
    apgrf = np.zeros(num_frames)
    apgrf[:int(0.3*num_frames)] = -100 * np.sin(np.pi * time[:int(0.3*num_frames)] / 0.3)
    apgrf[int(0.3*num_frames):int(0.6*num_frames)] = 120 * np.sin(np.pi * (time[int(0.3*num_frames):int(0.6*num_frames)] - 0.3) / 0.3)
    
    # Medial-lateral GRF (small oscillations)
    mlgrf = 30 * np.sin(4 * np.pi * time)
    
    # Add noise to GRF data
    vgrf += noise_level * 50 * np.random.randn(num_frames)
    apgrf += noise_level * 20 * np.random.randn(num_frames)
    mlgrf += noise_level * 10 * np.random.randn(num_frames)
    
    # Generate EMG data for key muscles
    gastrocnemius = np.zeros(num_frames)
    gastrocnemius[int(0.2*num_frames):int(0.6*num_frames)] = np.sin(np.pi * (time[int(0.2*num_frames):int(0.6*num_frames)] - 0.2) / 0.4)
    
    tibialis = np.zeros(num_frames)
    tibialis[:int(0.2*num_frames)] = 0.8 * (1 - time[:int(0.2*num_frames)] / 0.2)
    tibialis[int(0.6*num_frames):] = 0.9 * ((time[int(0.6*num_frames):] - 0.6) / 0.4)
    
    quadriceps = np.zeros(num_frames)
    quadriceps[:int(0.3*num_frames)] = 0.9 * (1 - time[:int(0.3*num_frames)] / 0.3)
    quadriceps[int(0.8*num_frames):] = 0.7 * ((time[int(0.8*num_frames):] - 0.8) / 0.2)
    
    hamstrings = np.zeros(num_frames)
    hamstrings[:int(0.1*num_frames)] = 0.8 * (1 - time[:int(0.1*num_frames)] / 0.1)
    hamstrings[int(0.7*num_frames):] = 0.9 * ((time[int(0.7*num_frames):] - 0.7) / 0.3)
    
    # Add noise to EMG data
    gastrocnemius += 0.1 * np.random.randn(num_frames)
    tibialis += 0.1 * np.random.randn(num_frames)
    quadriceps += 0.1 * np.random.randn(num_frames)
    hamstrings += 0.1 * np.random.randn(num_frames)
    
    # Ensure EMG values are between 0 and 1
    gastrocnemius = np.clip(gastrocnemius, 0, 1)
    tibialis = np.clip(tibialis, 0, 1)
    quadriceps = np.clip(quadriceps, 0, 1)
    hamstrings = np.clip(hamstrings, 0, 1)
    
    # Create DataFrame with all data
    data = pd.DataFrame({
        'time': time,
        'hip_angle': hip_angle,
        'knee_angle': knee_angle,
        'ankle_angle': ankle_angle,
        'vertical_force': vgrf,
        'anterior_posterior_force': apgrf,
        'medial_lateral_force': mlgrf,
        'gastrocnemius': gastrocnemius,
        'tibialis_anterior': tibialis,
        'quadriceps': quadriceps,
        'hamstrings': hamstrings,
        'heel_z': 5 * np.sin(2 * np.pi * time + np.pi) + 10,
        'toe_z': 5 * np.sin(2 * np.pi * time) + 10
    })
    
    return data


def print_opensim_info():
    """Print information about the OpenSim installation."""
    print(f"OpenSim Version: {osim.GetVersionAndDate()}")
    # OpenSim 4.5.2 doesn't have GetInstallDir function
    # print(f"OpenSim Library Path: {osim.GetInstallDir()}")


def main():
    """Main function to demonstrate walking simulation and visualization."""
    print("\n" + "="*80)
    print("WALKING SIMULATION AND VISUALIZATION USING OPENSIM AND BAF")
    print("="*80)
    
    # Print OpenSim information
    print("\nOpenSim Information:")
    print_opensim_info()
    
    # Generate walking data
    print("\nGenerating walking simulation data...")
    data = generate_walking_data(num_frames=100)
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    data.to_csv('output/walking_simulation_data.csv', index=False)
    print(f"Walking data saved to 'output/walking_simulation_data.csv'")
    
    # Detect gait events
    print("\nDetecting gait events...")
    events = {
        'foot_strike_1': 0,
        'toe_off': int(0.6 * len(data)),  # Toe-off at 60% of gait cycle
        'foot_strike_2': len(data) - 1
    }
    print(f"Detected events: {events}")
    
    # Normalize data to gait cycle
    print("\nNormalizing data to gait cycle...")
    normalized_data = normalize_gait_cycle(data, events)
    
    # Compute gait metrics
    print("\nComputing gait metrics...")
    grf_data = pd.DataFrame({
        'vertical_force': normalized_data['vertical_force'],
        'anterior_posterior_force': normalized_data['anterior_posterior_force'],
        'medial_lateral_force': normalized_data['medial_lateral_force']
    })
    
    joint_angles = pd.DataFrame({
        'hip_angle': normalized_data['hip_angle'],
        'knee_angle': normalized_data['knee_angle'],
        'ankle_angle': normalized_data['ankle_angle']
    })
    
    metrics = compute_gait_metrics(joint_angles, grf_data, events)
    print("\nGait Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Joint Angles Plot
    print("  - Joint angles plot")
    joint_plotter = JointPlotter(figsize=(12, 4))
    fig, axes = joint_plotter.plot_joint_angles(
        normalized_data,
        joint_cols={"hip": "hip_angle", "knee": "knee_angle", "ankle": "ankle_angle"},
        gait_events={"toe_off": 60},
        title="Joint Angles During Gait Cycle"
    )
    joint_plotter.save_figure('output/joint_angles.png')
    
    # 2. Ground Reaction Forces Plot
    print("  - Ground reaction forces plot")
    grf_plotter = GRFPlotter(figsize=(10, 6))
    fig, axes = grf_plotter.plot_grf(
        normalized_data,
        vertical_col='vertical_force',
        ap_col='anterior_posterior_force',
        ml_col='medial_lateral_force',
        gait_events={"toe_off": 60},
        title="Ground Reaction Forces During Gait Cycle"
    )
    grf_plotter.save_figure('output/ground_reaction_forces.png')
    
    # 3. EMG Activity Plot
    print("  - EMG activity plot")
    emg_plotter = EMGPlotter(figsize=(10, 6))
    fig, axes = emg_plotter.plot_emg(
        normalized_data,
        muscle_cols={
            "gastrocnemius": "gastrocnemius",
            "tibialis_anterior": "tibialis_anterior",
            "quadriceps": "quadriceps",
            "hamstrings": "hamstrings"
        },
        gait_events={"toe_off": 60},
        title="Muscle Activity During Gait Cycle"
    )
    emg_plotter.save_figure('output/emg_activity.png')
    
    # 4. Combined Gait Analysis Plot
    print("  - Combined gait analysis plot")
    comparative_plotter = ComparativePlotter(figsize=(12, 10))
    fig, axes = comparative_plotter.plot_gait_analysis(
        normalized_data,
        joint_cols={
            "hip": "hip_angle",
            "knee": "knee_angle",
            "ankle": "ankle_angle"
        },
        grf_cols={
            "vertical": "vertical_force",
            "anterior_posterior": "anterior_posterior_force",
            "medial_lateral": "medial_lateral_force"
        },
        emg_cols={
            "gastrocnemius": "gastrocnemius",
            "tibialis_anterior": "tibialis_anterior",
            "quadriceps": "quadriceps",
            "hamstrings": "hamstrings"
        },
        gait_events={"toe_off": 60},
        title="Comprehensive Gait Analysis"
    )
    comparative_plotter.save_figure('output/combined_gait_analysis.png')
    
    print("\nAll visualizations saved to the 'output' directory")
    print("\nDisplaying plots (close windows to continue)...")
    
    # Show plots
    plt.show()
    
    print("\nWalking simulation and visualization completed successfully!")


if __name__ == "__main__":
    main() 