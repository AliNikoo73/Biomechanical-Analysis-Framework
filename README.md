# Biomechanics Analysis Framework

A comprehensive framework for biomechanical analysis using OpenSim and Moco, with specialized modules for assistive device design and optimization. This framework enables researchers and clinicians to perform sophisticated biomechanical simulations, analyze human movement, and design optimized assistive devices.

## 🔍 Overview

The Biomechanics Analysis Framework provides a modular, extensible platform for:
- Analyzing human movement through inverse kinematics and dynamics
- Optimizing muscle forces and control strategies
- Designing and evaluating assistive devices such as exoskeletons and prosthetics
- Visualizing and processing biomechanical data
- Implementing advanced optimization algorithms for rehabilitation

The framework builds on OpenSim's powerful biomechanical modeling capabilities and extends them with specialized modules for assistive device optimization, advanced visualization, and machine learning integration.

## ✨ Features

- **Biomechanical Analysis**
  - Inverse Kinematics and Dynamics Analysis
  - Muscle Force Optimization
  - Forward Dynamics Simulation
  - Joint Torque and Power Analysis
  - Gait Analysis and Evaluation

- **Assistive Device Design and Optimization**
  - Exoskeleton Parameter Optimization
  - Prosthetic Device Design Tools
  - Assistive Force Prediction
  - Patient-Specific Customization
  - Novel Optimization Algorithms (including ADOHRL)

- **Visualization and Analysis**
  - Joint Angle Visualization
  - Ground Reaction Force Analysis
  - EMG Signal Processing and Visualization
  - Motion Animation and Recording
  - Comparative Analysis Tools

- **Data Processing Utilities**
  - Gait Cycle Normalization
  - Event Detection
  - Signal Processing
  - Data Import/Export
  - Statistical Analysis

## 📊 Simulation Results

Our framework produces detailed biomechanical analyses, including:

### Joint Angle Analysis

<img src="./simulation_results/joint_angles.png" alt="Joint Angles" width="700">

**Description:** This plot shows the joint angles (in degrees) for the hip, knee, and ankle throughout a complete gait cycle (0-100%). Key features include:
- **Hip Flexion/Extension (red):** Shows hip joint movement, with positive values indicating flexion and negative values indicating extension
- **Knee Flexion/Extension (green):** Shows the characteristic knee flexion pattern during stance and swing phases
- **Ankle Dorsi/Plantarflexion (blue):** Shows ankle movement, with upward deflection indicating dorsiflexion and downward indicating plantarflexion
- **Vertical dashed line:** Marks the transition from stance to swing phase (toe-off) at approximately 60% of the gait cycle

### Ground Reaction Forces

<img src="./simulation_results/ground_reaction_forces.png" alt="Ground Reaction Forces" width="700">

**Description:** This plot displays the ground reaction forces (as % of body weight) throughout the gait cycle:
- **Vertical GRF (red):** Shows the characteristic double-peak pattern of vertical force during stance phase, with the first peak representing weight acceptance and the second peak representing push-off
- **Anterior-Posterior GRF (green):** Shows braking forces (negative values) in early stance and propulsive forces (positive values) in late stance
- **Medial-Lateral GRF (blue):** Shows the smaller medial-lateral forces during walking
- **Forces reduce to zero** during swing phase (after 60%) when the foot is no longer in contact with the ground

### Muscle Activation Patterns

<img src="./simulation_results/emg_activity.png" alt="EMG Activity" width="700">

**Description:** This plot shows the electromyography (EMG) activity patterns of key lower limb muscles during the gait cycle:
- **Gastrocnemius:** Primarily active during mid to late stance phase, with peak activity during push-off
- **Tibialis Anterior:** Shows two main activity periods - during initial contact to control foot placement and during swing phase for ankle dorsiflexion
- **Quadriceps:** Most active during early stance phase for weight acceptance and knee stability
- **Hamstrings:** Active during late swing and early stance phases to decelerate the limb and control knee extension

### Comprehensive Gait Analysis

<img src="./simulation_results/combined_gait_analysis.png" alt="Combined Gait Analysis" width="700">

**Description:** This comprehensive visualization integrates all key biomechanical parameters to provide a complete view of the gait cycle:
- **Top panel:** Joint angles showing coordinated movement patterns of hip, knee, and ankle
- **Middle panel:** Ground reaction forces showing the interaction between the foot and ground
- **Bottom panels:** Muscle activation patterns divided into lower leg muscles (left) and upper leg muscles (right)

This integrated view allows researchers to examine the relationships between joint kinematics, kinetics, and muscle activity throughout the gait cycle.

## 🧩 Project Structure

```
biomech-analysis-framework/
├── src/                        # Source code for the framework
│   ├── kinematics/             # Joint and segment kinematics analysis
│   ├── dynamics/               # Force and torque calculations
│   ├── optimization/           # Optimization algorithms
│   │   └── adaptive_dual_objective_rl.py  # ADOHRL algorithm
│   ├── visualization/          # Data visualization tools
│   ├── assistive_devices/      # Assistive device models
│   │   ├── exoskeleton/        # Exoskeleton models and controllers
│   │   └── prosthetics/        # Prosthetic device models
│   └── utils/                  # Utility functions
├── examples/                   # Example scripts and tutorials
│   ├── tracking/               # Motion tracking examples
│   ├── gait_analysis/          # Gait analysis examples
│   ├── muscle_optimization/    # Muscle force optimization examples
│   └── assistive_devices/      # Assistive device examples
│       ├── exoskeleton_optimization.py  # Exoskeleton optimization example
├── docs/                       # Documentation
├── tests/                      # Unit and integration tests
├── simulation_results/         # Generated simulation visualizations
└── data/                       # Sample data and models
    ├── models/                 # OpenSim models
    └── raw/                    # Raw experimental data
```

## 📋 Prerequisites

- OpenSim 4.4 or later
- OpenSim Moco 1.0.0 or later
- Python 3.8+
- NumPy (>= 1.21.0)
- SciPy (>= 1.7.0)
- Matplotlib (>= 3.4.0)
- Pandas (>= 1.3.0)
- PyTorch (for machine learning components)

## 🔧 Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/biomech-analysis-framework.git
cd biomech-analysis-framework
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install OpenSim and Moco following the [official installation guide](https://opensim.stanford.edu/install/).

## 💻 Usage

### Basic Simulation

For a basic simulation, run:

```bash
python examples/gait_analysis/gait_simulation.py
```

### Exoskeleton Optimization

To run the exoskeleton optimization example using our novel ADOHRL algorithm:

```bash
python examples/assistive_devices/exoskeleton_optimization.py
```

### Data Visualization

To generate visualization plots of gait analysis:

```bash
python generate_plots.py
```

This will create the simulation plots in the `simulation_results` directory, which can be viewed with any image viewer.

## 🌟 Key Innovations

### Adaptive Dual-Objective Hybrid Reinforcement Learning (ADOHRL)

Our framework introduces a novel optimization algorithm for assistive device control:

- **Dual-Objective Optimization:** Simultaneously optimizes for user comfort and energy efficiency
- **Adaptive User Preference Learning:** Learns and adapts to individual user preferences over time
- **Biomechanical Constraint Integration:** Incorporates physiological constraints into the optimization
- **Hybrid Reward Formulation:** Combines immediate physical rewards with long-term adaptation benefits

For more details, see [ADOHRL Documentation](docs/algorithms/adaptive_dual_objective_rl.md).

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [Tutorials](docs/tutorials/)
- [API Reference](docs/api/)
- [Algorithm Documentation](docs/algorithms/)

## 🎯 Examples

1. [Basic Moco Example](examples/basic_moco_example/): Introduction to trajectory optimization
2. [Gait Analysis](examples/gait_analysis/): Analysis of human walking
3. [Muscle Optimization](examples/muscle_optimization/): Optimizing muscle forces
4. [Exoskeleton Optimization](examples/assistive_devices/exoskeleton_optimization.py): Optimizing exoskeleton parameters

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and contribute to the project.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenSim Team at Stanford University
- Contributors to the OpenSim and Moco projects
- Biomechanics research community

## 📧 Contact

For questions and support, please [open an issue](https://github.com/yourusername/biomech-analysis-framework/issues) or contact the maintainers. 