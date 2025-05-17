# Biomechanical Analysis Framework (BAF)

A comprehensive Python framework for biomechanical data analysis, visualization, and assistive device optimization.

## Overview

The Biomechanical Analysis Framework (BAF) provides a modular, extensible platform for analyzing human movement data, visualizing biomechanical parameters, and optimizing assistive devices. It integrates with OpenSim for musculoskeletal modeling and simulation.

## Features

- **Data Processing**: Tools for processing, filtering, and normalizing biomechanical data
- **Kinematics Analysis**: Joint angle calculation and segment trajectory analysis
- **Dynamics Analysis**: Inverse dynamics for calculating joint moments, powers, and work
- **Visualization**: Comprehensive plotting tools for joint angles, GRF, EMG, and combined analyses
- **Assistive Device Modeling**: Framework for modeling and optimizing exoskeletons, prosthetics, and orthoses
- **OpenSim Integration**: Seamless integration with OpenSim for musculoskeletal modeling and simulation
- **GUI Application**: User-friendly interface for common biomechanical analysis tasks

## Research Results

### Gait Kinematics

![Joint Angles](output/joint_angles.png)

The joint angle analysis reveals characteristic patterns throughout the gait cycle:
- **Hip**: Exhibits ~40° of total excursion, with peak flexion occurring at initial contact and terminal swing, and peak extension at terminal stance
- **Knee**: Shows biphasic pattern with flexion peaks during loading response (~15°) and swing phase (~60°), with near full extension at terminal stance
- **Ankle**: Demonstrates controlled dorsiflexion during stance phase followed by rapid plantarflexion at push-off (~20°), transitioning to dorsiflexion during swing to ensure foot clearance

### Ground Reaction Forces

![Ground Reaction Forces](output/ground_reaction_forces.png)

Ground reaction force analysis demonstrates:
- **Vertical GRF**: Classic double-peak pattern with first peak (~110% BW) representing weight acceptance and second peak (~110% BW) representing push-off, with mid-stance valley (~80% BW)
- **Anterior-Posterior GRF**: Initial braking forces (negative values, ~20% BW) transitioning to propulsive forces (positive values, ~20% BW) at ~50% of stance
- **Medial-Lateral GRF**: Smaller magnitude forces (<10% BW) reflecting medio-lateral control during stance phase

### Muscle Activation Patterns

![EMG Activity](output/emg_activity.png)

Electromyography analysis reveals coordinated muscle activation patterns:
- **Gastrocnemius**: Primary activation during mid-to-late stance (30-60% gait cycle) with peak activity at terminal stance for push-off
- **Tibialis Anterior**: Biphasic activation with peaks during initial contact (0-10% gait cycle) for controlled foot placement and during swing (60-100% gait cycle) for ankle dorsiflexion
- **Quadriceps**: Highest activity during loading response (0-20% gait cycle) to control knee flexion and provide stability
- **Hamstrings**: Active during terminal swing and initial contact (90-10% gait cycle) to decelerate the limb and control hip/knee extension

### Joint Dynamics

![Joint Moments](output/joint_moments.png)

Joint moment analysis shows:
- **Ankle**: Progressive increase in plantarflexor moment during stance, peaking at ~1.5 Nm/kg at terminal stance (50% gait cycle)
- **Knee**: Initial extensor moment (~0.5 Nm/kg) during loading response, transitioning to flexor moment during mid-stance, with second extensor peak during terminal stance
- **Hip**: Extensor moment (~1.0 Nm/kg) during early stance for stability, transitioning to flexor moment during pre-swing and early swing

![Joint Powers](output/joint_powers.png)

Joint power analysis reveals energy flow patterns:
- **Ankle**: Power absorption (A1, negative power) during mid-stance as the tibia rotates over the foot, followed by substantial power generation (A2, positive power, ~3.5 W/kg) during push-off
- **Knee**: Primary power absorption during loading response (K1), mid-stance (K2), and terminal swing (K4), with brief power generation during pre-swing (K3)
- **Hip**: Power generation in early stance (H1, ~1.0 W/kg) to assist forward progression and in terminal swing (H3, ~0.5 W/kg) to decelerate the limb, with power absorption during pre-swing (H2)

### Joint Work Analysis

Joint work analysis quantifies energy generation and absorption:

| Joint | Positive Work (J) | Negative Work (J) | Net Work (J) | Functional Interpretation |
|-------|-------------------|-------------------|--------------|---------------------------|
| Ankle | 241.66 | -377.88 | -136.22 | Significant power generation at push-off despite net energy absorption |
| Knee | 1729.91 | -2766.41 | -1036.50 | Primarily functions as an energy absorber during gait |
| Hip | 935.10 | -6871.29 | -5936.20 | Substantial energy absorption with targeted power generation phases |

*Note: Positive work represents energy generation (concentric muscle action), while negative work represents energy absorption (eccentric muscle action).*

### Integrated Biomechanical Analysis

![Combined Gait Analysis](output/combined_gait_analysis.png)

The integrated analysis demonstrates the temporal coordination between:
- Joint kinematics (top panel)
- Ground reaction forces (middle panel)
- Muscle activation patterns (bottom panel)

This comprehensive view reveals the synergistic relationships between neural control (EMG), movement outcomes (kinematics), and external forces (GRF) throughout the gait cycle.

### Clinical Implications

The dynamics analysis provides several key insights for clinical applications:

1. **Energy Transfer**: The distal-to-proximal energy flow during gait, with ankle power generation (A2 burst) initiating swing phase, highlights the importance of ankle function in efficient locomotion
2. **Joint Specialization**: The knee primarily functions as an energy absorber (negative work), while the hip and ankle contribute to energy generation at specific gait phases
3. **Metabolic Efficiency**: The predominantly eccentric muscle action (negative work) across joints reflects the metabolically efficient nature of human gait
4. **Inter-joint Coordination**: The synchronized timing of power bursts demonstrates energy transfer through the kinetic chain, with implications for rehabilitation strategies targeting specific gait phases

These biomechanical insights provide an evidence-based foundation for clinical assessment, rehabilitation planning, and assistive device design.

## Installation

### Prerequisites

- Python 3.7+
- OpenSim 4.5.2+

### Installing OpenSim

For Mac with Arm64 processors (M1/M2):

```bash
conda create -n opensim python=3.9
conda activate opensim
conda install -c opensim-org opensim=4.5.2
```

For other platforms, see the [OpenSim documentation](https://simtk-confluence.stanford.edu/display/OpenSim/Installing+OpenSim).

### Installing BAF

```bash
git clone https://github.com/yourusername/biomechanical-analysis-framework.git
cd biomechanical-analysis-framework
pip install -e .
```

## Usage

### Basic Usage

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from baf.utils.data_processing import normalize_gait_cycle, detect_gait_events
from baf.visualization import JointPlotter, GRFPlotter, EMGPlotter, ComparativePlotter

# Load data
data = pd.read_csv('your_gait_data.csv')

# Detect gait events
events = detect_gait_events(data)

# Normalize to gait cycle
normalized_data = normalize_gait_cycle(data, events)

# Create visualizations
joint_plotter = JointPlotter()
fig, ax = joint_plotter.plot_joint_angles(
    normalized_data,
    joint_cols={"hip": "hip_angle", "knee": "knee_angle", "ankle": "ankle_angle"},
    gait_events={"toe_off": 60}
)
plt.show()
```

### Dynamics Analysis

```python
from baf.analysis.dynamics import calculate_joint_moments, calculate_joint_powers, calculate_joint_work

# Define anthropometric data
anthropometry = {
    'mass': 70.0,  # kg
    'height': 1.75,  # m
}

# Calculate joint kinetics
moments = calculate_joint_moments(kinematics, grf, anthropometry)
powers = calculate_joint_powers(kinematics, moments)
work = calculate_joint_work(powers)

# Visualize results
plotter = ComparativePlotter()
fig, axes = plotter.plot_condition_comparison(
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
```

### GUI Application

Launch the GUI application:

```bash
python -m baf.gui
```

Or use the provided entry point:

```bash
baf-gui
```

## Examples

The `examples` directory contains example scripts demonstrating various features of the framework:

- `walking_simulation.py`: Generate and visualize walking data
- `dynamics_analysis_example.py`: Calculate and visualize joint kinetics
- More examples coming soon...

## Documentation

Comprehensive documentation is available in the `docs` directory.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenSim team for their musculoskeletal modeling software
- Contributors to the biomechanics research community 