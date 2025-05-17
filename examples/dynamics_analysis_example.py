#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dynamics Analysis Example

This script demonstrates how to use the BAF dynamics analysis module
to calculate joint moments, powers, and work during gait.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import BAF modules
from baf.analysis.dynamics import calculate_joint_moments, calculate_joint_powers, calculate_joint_work
from baf.utils.data_processing import normalize_gait_cycle
from baf.visualization.comparative_plots import ComparativePlotter


def generate_sample_data(num_frames=100):
    """Generate sample kinematics and GRF data for dynamics analysis."""
    # Time vector (0 to 1 second)
    time = np.linspace(0, 1, num_frames)
    
    # Generate joint angle data
    hip_angle = 30 * np.sin(2 * np.pi * time - np.pi/6) - 5
    knee_angle = 5 + 60 * (0.5 - 0.5 * np.cos(2 * np.pi * time + np.pi/4))
    knee_angle -= 20 * np.exp(-((time - 0.3) ** 2) / 0.02)  # Extension in mid-stance
    ankle_angle = 10 * np.sin(2 * np.pi * time + np.pi/2)
    ankle_angle -= 15 * np.exp(-((time - 0.6) ** 2) / 0.01)  # Plantarflexion at push-off
    
    # Add some noise
    noise_level = 0.05
    hip_angle += noise_level * np.random.randn(num_frames) * np.max(np.abs(hip_angle))
    knee_angle += noise_level * np.random.randn(num_frames) * np.max(np.abs(knee_angle))
    ankle_angle += noise_level * np.random.randn(num_frames) * np.max(np.abs(ankle_angle))
    
    # Calculate angular velocities (deg/s)
    dt = 1.0 / num_frames
    hip_angular_velocity = np.gradient(hip_angle, dt)
    knee_angular_velocity = np.gradient(knee_angle, dt)
    ankle_angular_velocity = np.gradient(ankle_angle, dt)
    
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
    
    # Add noise to GRF data
    vgrf += noise_level * 50 * np.random.randn(num_frames)
    apgrf += noise_level * 20 * np.random.randn(num_frames)
    
    # Create DataFrames
    kinematics = pd.DataFrame({
        'time': time,
        'hip_angle': hip_angle,
        'knee_angle': knee_angle,
        'ankle_angle': ankle_angle,
        'hip_angular_velocity': hip_angular_velocity,
        'knee_angular_velocity': knee_angular_velocity,
        'ankle_angular_velocity': ankle_angular_velocity
    })
    
    grf = pd.DataFrame({
        'time': time,
        'vertical_force': vgrf,
        'anterior_posterior_force': apgrf
    })
    
    return kinematics, grf


def main():
    """Main function to demonstrate dynamics analysis."""
    print("\n" + "="*80)
    print("DYNAMICS ANALYSIS EXAMPLE USING BAF")
    print("="*80)
    
    # Generate sample data
    print("\nGenerating sample data...")
    kinematics, grf = generate_sample_data(num_frames=100)
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Define anthropometric data for a typical adult
    anthropometry = {
        'mass': 70.0,  # kg
        'height': 1.75,  # m
    }
    
    print("\nCalculating joint moments...")
    moments = calculate_joint_moments(kinematics, grf, anthropometry)
    
    print("\nCalculating joint powers...")
    powers = calculate_joint_powers(kinematics, moments)
    
    print("\nCalculating joint work...")
    work = calculate_joint_work(powers, time_col='time')
    
    # Print joint work results
    print("\nJoint Work Summary:")
    for joint, work_data in work.items():
        print(f"\n{joint.title()} Joint:")
        print(f"  Positive Work: {work_data['positive_work']:.2f} J")
        print(f"  Negative Work: {work_data['negative_work']:.2f} J")
        print(f"  Net Work: {work_data['net_work']:.2f} J")
    
    # Normalize data to gait cycle for visualization
    print("\nNormalizing data to gait cycle...")
    events = {
        'foot_strike_1': 0,
        'toe_off': int(0.6 * len(kinematics)),  # Toe-off at 60% of gait cycle
        'foot_strike_2': len(kinematics) - 1
    }
    
    # Combine all data for normalization
    all_data = kinematics.copy()
    all_data['vertical_force'] = grf['vertical_force']
    all_data['anterior_posterior_force'] = grf['anterior_posterior_force']
    all_data['ankle_moment'] = moments['ankle_moment']
    all_data['knee_moment'] = moments['knee_moment']
    all_data['hip_moment'] = moments['hip_moment']
    all_data['ankle_power'] = powers['ankle_power']
    all_data['knee_power'] = powers['knee_power']
    all_data['hip_power'] = powers['hip_power']
    
    normalized_data = normalize_gait_cycle(all_data, events)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Joint Moments Plot
    print("  - Joint moments plot")
    plotter = ComparativePlotter(figsize=(12, 6))
    fig, axes = plotter.plot_gait_analysis(
        normalized_data,
        joint_cols={
            "hip": "hip_angle",
            "knee": "knee_angle",
            "ankle": "ankle_angle"
        },
        grf_cols={
            "vertical": "vertical_force",
            "anterior_posterior": "anterior_posterior_force"
        },
        gait_events={"toe_off": 60},
        title="Kinematics and Ground Reaction Forces"
    )
    plotter.save_figure('output/kinematics_grf.png')
    
    # 2. Joint Moments Plot
    print("  - Joint moments plot")
    moment_plotter = ComparativePlotter(figsize=(10, 6))
    fig, axes = moment_plotter.plot_condition_comparison(
        {'Ankle': normalized_data, 'Knee': normalized_data, 'Hip': normalized_data},
        plot_type='joint_angle',
        column={
            'Ankle': 'ankle_moment',
            'Knee': 'knee_moment',
            'Hip': 'hip_moment'
        },
        gait_events={"toe_off": 60},
        title="Joint Moments During Gait Cycle"
    )
    
    # Customize y-axis label
    axes.set_ylabel('Moment (Nm)')
    plt.tight_layout()
    plt.savefig('output/joint_moments.png')
    
    # 3. Joint Powers Plot
    print("  - Joint powers plot")
    power_plotter = ComparativePlotter(figsize=(10, 6))
    fig, axes = power_plotter.plot_condition_comparison(
        {'Ankle': normalized_data, 'Knee': normalized_data, 'Hip': normalized_data},
        plot_type='joint_angle',
        column={
            'Ankle': 'ankle_power',
            'Knee': 'knee_power',
            'Hip': 'hip_power'
        },
        gait_events={"toe_off": 60},
        title="Joint Powers During Gait Cycle"
    )
    
    # Customize y-axis label
    axes.set_ylabel('Power (W)')
    plt.tight_layout()
    plt.savefig('output/joint_powers.png')
    
    print("\nAll visualizations saved to the 'output' directory")
    
    # Show plots
    plt.show()
    
    print("\nDynamics analysis completed successfully!")


if __name__ == "__main__":
    main() 