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
from baf.analysis.dynamics import calculate_joint_moments, calculate_joint_powers

# Define anthropometric data
anthropometry = {
    'mass': 70.0,  # kg
    'height': 1.75,  # m
}

# Calculate joint kinetics
moments = calculate_joint_moments(kinematics, grf, anthropometry)
powers = calculate_joint_powers(kinematics, moments)

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