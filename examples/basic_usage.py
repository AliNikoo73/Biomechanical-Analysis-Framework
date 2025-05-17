#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic Usage Example

This script demonstrates basic usage of the BAF library for analyzing and
visualizing biomechanical data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import BAF modules
import baf
from baf.utils.data_processing import normalize_gait_cycle, detect_gait_events, compute_gait_metrics
from baf.visualization.joint_plots import JointPlotter


def generate_sample_data(num_frames=100, noise_level=0.1):
    """Generate sample gait data for demonstration."""
    # Time vector
    time = np.linspace(0, 1, num_frames)
    
    # Generate joint angle data
    hip_angle = 20 * np.sin(2 * np.pi * time) + 10  # Hip flexion/extension
    knee_angle = 45 * np.sin(2 * np.pi * time - np.pi/3) + 15  # Knee flexion/extension
    ankle_angle = 15 * np.sin(2 * np.pi * time - np.pi/2)  # Ankle dorsi/plantarflexion
    
    # Add some noise
    hip_angle += noise_level * np.random.randn(num_frames)
    knee_angle += noise_level * np.random.randn(num_frames)
    ankle_angle += noise_level * np.random.randn(num_frames)
    
    # Generate ground reaction force data
    vgrf = np.zeros(num_frames)
    vgrf[:int(0.6*num_frames)] = np.sin(np.pi * time[:int(0.6*num_frames)] / 0.6) * 800
    vgrf += noise_level * 50 * np.random.randn(num_frames)
    vgrf = np.maximum(vgrf, 0)  # Force can't be negative
    
    apgrf = np.zeros(num_frames)
    apgrf[:int(0.3*num_frames)] = -100 * np.sin(np.pi * time[:int(0.3*num_frames)] / 0.3)
    apgrf[int(0.3*num_frames):int(0.6*num_frames)] = 100 * np.sin(np.pi * (time[int(0.3*num_frames):int(0.6*num_frames)] - 0.3) / 0.3)
    apgrf += noise_level * 20 * np.random.randn(num_frames)
    
    mlgrf = 30 * np.sin(4 * np.pi * time) + noise_level * 10 * np.random.randn(num_frames)
    
    # Create DataFrame
    data = pd.DataFrame({
        'time': time,
        'hip_angle': hip_angle,
        'knee_angle': knee_angle,
        'ankle_angle': ankle_angle,
        'vertical_force': vgrf,
        'anterior_posterior_force': apgrf,
        'medial_lateral_force': mlgrf,
        'heel_z': 5 * np.sin(2 * np.pi * time + np.pi) + 10,
        'toe_z': 5 * np.sin(2 * np.pi * time) + 10
    })
    
    return data


def main():
    """Main function to demonstrate BAF usage."""
    print("Biomechanical Analysis Framework (BAF) Basic Usage Example")
    print("=" * 60)
    
    # Generate sample data
    print("\nGenerating sample gait data...")
    data = generate_sample_data(num_frames=100)
    
    # Save sample data
    os.makedirs('output', exist_ok=True)
    data.to_csv('output/sample_gait_data.csv', index=False)
    print(f"Sample data saved to 'output/sample_gait_data.csv'")
    
    # Detect gait events
    print("\nDetecting gait events...")
    events = detect_gait_events(
        marker_data=data,
        force_data=data['vertical_force'],
        method='force',
        threshold=50.0
    )
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
    
    # Visualize joint angles
    print("\nVisualizing joint angles...")
    plotter = JointPlotter(figsize=(12, 5))
    fig, axes = plotter.plot_joint_angles(
        normalized_data,
        joint_cols={"hip": "hip_angle", "knee": "knee_angle", "ankle": "ankle_angle"},
        gait_events={"toe_off": 60},  # Assuming toe-off at 60%
        title="Joint Angles During Gait Cycle"
    )
    
    # Save the figure
    plotter.save_figure('output/joint_angles.png')
    print(f"Joint angle plot saved to 'output/joint_angles.png'")
    
    # Show the figure
    print("\nDisplaying plot (close window to continue)...")
    plotter.show()
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main() 