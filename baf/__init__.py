"""
Biomechanical Analysis Framework (BAF)
=====================================

A comprehensive framework for biomechanical analysis, simulation, and assistive device optimization.
"""

__version__ = "0.1.0"

# Import main submodules for easier access
from . import analysis
from . import optimization
from . import visualization
from . import assistive_devices
from . import utils
from . import models

# Define what gets imported with "from baf import *"
__all__ = [
    "analysis",
    "optimization",
    "visualization",
    "assistive_devices",
    "utils",
    "models",
] 