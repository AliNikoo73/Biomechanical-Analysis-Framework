# Installation Guide

This guide will help you set up the Biomechanics Analysis Framework on your system.

## System Requirements

- Operating System: Windows 10/11, macOS 10.15+, or Linux
- Python 3.8 or later
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space

## Step 1: Install OpenSim and Moco

1. Download OpenSim 4.4 or later from the [official website](https://opensim.stanford.edu/download/)
2. Follow the installation instructions for your operating system
3. Verify the installation by running the OpenSim GUI
4. Install Moco following the [Moco documentation](https://opensim.stanford.edu/moco/)

## Step 2: Set Up Python Environment

```bash
# Create a new virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

## Step 3: Configure OpenSim Python Bindings

1. Locate your OpenSim installation directory
2. Add the OpenSim libraries to your system PATH:
   - Windows: Add `<OpenSim_Install_Dir>/bin` to PATH
   - macOS: Add to `DYLD_LIBRARY_PATH`
   - Linux: Add to `LD_LIBRARY_PATH`

## Step 4: Verify Installation

Run the test suite to verify everything is working:

```bash
pytest tests/
```

## Common Issues and Solutions

### Missing OpenSim Libraries

If you encounter "ImportError: No module named opensim", ensure:
- OpenSim is properly installed
- System PATH variables are correctly set
- Python version matches OpenSim build

### Moco Integration Issues

If Moco tools are not available:
1. Verify Moco installation in OpenSim GUI
2. Check Moco binary location is in system PATH
3. Reinstall OpenSim with Moco if necessary

## Getting Help

If you encounter any issues:
1. Check our [FAQ](./faq.md)
2. Search existing [GitHub Issues](https://github.com/yourusername/biomech-analysis-framework/issues)
3. Create a new issue with detailed information about your problem

## Next Steps

- Read the [Getting Started Guide](./tutorials/getting_started.md)
- Try the [Basic Examples](../examples/basic_moco_example/)
- Explore the [API Documentation](./api/) 