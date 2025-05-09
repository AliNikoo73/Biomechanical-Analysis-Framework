#!/usr/bin/env python3
"""
Generate plots for joint angles, GRF, and EMG data for a complete walking cycle
using simulated data (since we don't have direct access to OpenSim Python API).

This script creates visualizations similar to what you would get from running
an OpenSim simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# Create output directory if it doesn't exist
os.makedirs('simulation_results', exist_ok=True)

# Simulate a complete gait cycle (0-100%)
gait_cycle = np.linspace(0, 100, 101)  # 0-100% of gait cycle

# ---------- Simulate Joint Angle Data ----------
# Hip flexion/extension
hip_angle = 30 * np.sin((gait_cycle - 15) * np.pi / 50) - 10
# Knee flexion/extension
knee_angle = 60 * np.exp(-((gait_cycle - 15) / 10) ** 2) + 5 * np.sin(gait_cycle * np.pi / 50)
# Ankle dorsi/plantarflexion
ankle_angle = -15 * np.sin((gait_cycle - 60) * np.pi / 50) - 5

# ---------- Simulate Ground Reaction Forces ----------
# Vertical GRF (% body weight)
grf_vertical = np.zeros_like(gait_cycle)
grf_vertical[:60] = 120 * np.sin(gait_cycle[:60] * np.pi / 60) 
grf_vertical[60:] = 80 * np.sin((gait_cycle[60:] - 60) * np.pi / 40)
# Ensure GRF is zero at beginning and end of cycle
grf_vertical[:5] = grf_vertical[:5] * np.linspace(0, 1, 5)
grf_vertical[-5:] = grf_vertical[-5:] * np.linspace(1, 0, 5)

# Anterior-posterior GRF (% body weight)
grf_ap = np.zeros_like(gait_cycle)
grf_ap[:50] = -15 * np.sin(gait_cycle[:50] * np.pi / 50)
grf_ap[50:] = 25 * np.sin((gait_cycle[50:] - 50) * np.pi / 50)
# Ensure GRF is zero at beginning and end of cycle
grf_ap[:5] = grf_ap[:5] * np.linspace(0, 1, 5)
grf_ap[-5:] = grf_ap[-5:] * np.linspace(1, 0, 5)

# Medial-lateral GRF (% body weight)
grf_ml = 5 * np.sin(gait_cycle * np.pi / 25)
# Ensure GRF is zero at beginning and end of cycle
grf_ml[:5] = grf_ml[:5] * np.linspace(0, 1, 5)
grf_ml[-5:] = grf_ml[-5:] * np.linspace(1, 0, 5)

# ---------- Simulate EMG Data ----------
# Create noisy EMG signals for major muscles during gait
# Gastrocnemius (active during push-off)
gastroc_emg = np.zeros_like(gait_cycle)
gastroc_emg[30:60] = 0.8 * np.sin((gait_cycle[30:60] - 30) * np.pi / 30)
gastroc_emg += 0.1 * np.random.rand(len(gait_cycle))
gastroc_emg = np.clip(gastroc_emg, 0, 1)

# Tibialis Anterior (active during swing and initial contact)
tibialis_emg = np.zeros_like(gait_cycle)
tibialis_emg[:20] = 0.7 * np.sin(gait_cycle[:20] * np.pi / 20)
tibialis_emg[60:] = 0.6 * np.sin((gait_cycle[60:] - 60) * np.pi / 40)
tibialis_emg += 0.1 * np.random.rand(len(gait_cycle))
tibialis_emg = np.clip(tibialis_emg, 0, 1)

# Quadriceps (active during initial stance)
quad_emg = np.zeros_like(gait_cycle)
quad_emg[:30] = 0.9 * np.sin(gait_cycle[:30] * np.pi / 30)
quad_emg += 0.1 * np.random.rand(len(gait_cycle))
quad_emg = np.clip(quad_emg, 0, 1)

# Hamstrings (active during late swing and initial stance)
hamstring_emg = np.zeros_like(gait_cycle)
hamstring_emg[:15] = 0.5 * np.sin(gait_cycle[:15] * np.pi / 15)
hamstring_emg[70:] = 0.8 * np.sin((gait_cycle[70:] - 70) * np.pi / 30)
hamstring_emg += 0.1 * np.random.rand(len(gait_cycle))
hamstring_emg = np.clip(hamstring_emg, 0, 1)

# ---------- Create Plots ----------

# 1. Joint Angles Plot
plt.figure(figsize=(10, 6))
plt.plot(gait_cycle, hip_angle, 'r-', linewidth=2, label='Hip Flexion/Extension')
plt.plot(gait_cycle, knee_angle, 'g-', linewidth=2, label='Knee Flexion/Extension')
plt.plot(gait_cycle, ankle_angle, 'b-', linewidth=2, label='Ankle Dorsi/Plantarflexion')
plt.axvline(x=60, color='k', linestyle='--', alpha=0.5, label='Toe-Off')
plt.grid(True, alpha=0.3)
plt.title('Joint Angles During Gait Cycle', fontsize=16)
plt.xlabel('Gait Cycle (%)', fontsize=12)
plt.ylabel('Joint Angle (degrees)', fontsize=12)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('simulation_results/joint_angles.png', dpi=300)

# 2. Ground Reaction Forces Plot
plt.figure(figsize=(10, 6))
plt.plot(gait_cycle, grf_vertical, 'r-', linewidth=2, label='Vertical GRF')
plt.plot(gait_cycle, grf_ap, 'g-', linewidth=2, label='Anterior-Posterior GRF')
plt.plot(gait_cycle, grf_ml, 'b-', linewidth=2, label='Medial-Lateral GRF')
plt.axvline(x=60, color='k', linestyle='--', alpha=0.5, label='Toe-Off')
plt.grid(True, alpha=0.3)
plt.title('Ground Reaction Forces During Gait Cycle', fontsize=16)
plt.xlabel('Gait Cycle (%)', fontsize=12)
plt.ylabel('Force (% Body Weight)', fontsize=12)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('simulation_results/ground_reaction_forces.png', dpi=300)

# 3. EMG Plot
plt.figure(figsize=(10, 8))
plt.subplot(4, 1, 1)
plt.plot(gait_cycle, gastroc_emg, 'r-', linewidth=1.5)
plt.fill_between(gait_cycle, 0, gastroc_emg, color='r', alpha=0.3)
plt.title('Gastrocnemius', fontsize=12)
plt.ylim(0, 1.1)
plt.axvline(x=60, color='k', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)

plt.subplot(4, 1, 2)
plt.plot(gait_cycle, tibialis_emg, 'g-', linewidth=1.5)
plt.fill_between(gait_cycle, 0, tibialis_emg, color='g', alpha=0.3)
plt.title('Tibialis Anterior', fontsize=12)
plt.ylim(0, 1.1)
plt.axvline(x=60, color='k', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)

plt.subplot(4, 1, 3)
plt.plot(gait_cycle, quad_emg, 'b-', linewidth=1.5)
plt.fill_between(gait_cycle, 0, quad_emg, color='b', alpha=0.3)
plt.title('Quadriceps', fontsize=12)
plt.ylim(0, 1.1)
plt.axvline(x=60, color='k', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)

plt.subplot(4, 1, 4)
plt.plot(gait_cycle, hamstring_emg, 'purple', linewidth=1.5)
plt.fill_between(gait_cycle, 0, hamstring_emg, color='purple', alpha=0.3)
plt.title('Hamstrings', fontsize=12)
plt.ylim(0, 1.1)
plt.axvline(x=60, color='k', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)

plt.xlabel('Gait Cycle (%)', fontsize=12)
plt.tight_layout()
plt.savefig('simulation_results/emg_activity.png', dpi=300)

# 4. Combined Visualization
plt.figure(figsize=(15, 10))
gs = GridSpec(3, 2)

# Joint angles
ax1 = plt.subplot(gs[0, :])
ax1.plot(gait_cycle, hip_angle, 'r-', linewidth=2, label='Hip')
ax1.plot(gait_cycle, knee_angle, 'g-', linewidth=2, label='Knee')
ax1.plot(gait_cycle, ankle_angle, 'b-', linewidth=2, label='Ankle')
ax1.axvline(x=60, color='k', linestyle='--', alpha=0.5, label='Toe-Off')
ax1.set_title('Joint Angles', fontsize=14)
ax1.set_ylabel('Degrees', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# Ground reaction forces
ax2 = plt.subplot(gs[1, :])
ax2.plot(gait_cycle, grf_vertical, 'r-', linewidth=2, label='Vertical')
ax2.plot(gait_cycle, grf_ap, 'g-', linewidth=2, label='A-P')
ax2.plot(gait_cycle, grf_ml, 'b-', linewidth=2, label='M-L')
ax2.axvline(x=60, color='k', linestyle='--', alpha=0.5)
ax2.set_title('Ground Reaction Forces', fontsize=14)
ax2.set_ylabel('% Body Weight', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')

# EMG - Left side
ax3 = plt.subplot(gs[2, 0])
ax3.plot(gait_cycle, gastroc_emg, 'r-', linewidth=1.5, label='Gastrocnemius')
ax3.plot(gait_cycle, tibialis_emg, 'g-', linewidth=1.5, label='Tibialis Anterior')
ax3.fill_between(gait_cycle, 0, gastroc_emg, color='r', alpha=0.3)
ax3.fill_between(gait_cycle, 0, tibialis_emg, color='g', alpha=0.3)
ax3.axvline(x=60, color='k', linestyle='--', alpha=0.5)
ax3.set_title('EMG - Lower Leg Muscles', fontsize=14)
ax3.set_ylabel('Activation', fontsize=12)
ax3.set_xlabel('Gait Cycle (%)', fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper right')

# EMG - Right side
ax4 = plt.subplot(gs[2, 1])
ax4.plot(gait_cycle, quad_emg, 'b-', linewidth=1.5, label='Quadriceps')
ax4.plot(gait_cycle, hamstring_emg, 'purple', linewidth=1.5, label='Hamstrings')
ax4.fill_between(gait_cycle, 0, quad_emg, color='b', alpha=0.3)
ax4.fill_between(gait_cycle, 0, hamstring_emg, color='purple', alpha=0.3)
ax4.axvline(x=60, color='k', linestyle='--', alpha=0.5)
ax4.set_title('EMG - Upper Leg Muscles', fontsize=14)
ax4.set_ylabel('Activation', fontsize=12)
ax4.set_xlabel('Gait Cycle (%)', fontsize=12)
ax4.grid(True, alpha=0.3)
ax4.legend(loc='upper right')

plt.tight_layout()
plt.savefig('simulation_results/combined_gait_analysis.png', dpi=300)

# Print completion message
print("\nPlots have been generated and saved to the 'simulation_results' directory:")
print("1. Joint Angles (joint_angles.png)")
print("2. Ground Reaction Forces (ground_reaction_forces.png)")
print("3. EMG Activity (emg_activity.png)")
print("4. Combined Gait Analysis (combined_gait_analysis.png)")

# Show all plots
plt.show() 