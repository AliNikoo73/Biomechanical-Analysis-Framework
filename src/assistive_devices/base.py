from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import numpy as np
import opensim as osim

class AssistiveDevice(ABC):
    """Base class for all assistive devices in the framework."""
    
    def __init__(self, name: str, mass: float = 0.0):
        """
        Initialize the assistive device.
        
        Args:
            name: Name of the device
            mass: Total mass of the device in kg
        """
        self.name = name
        self.mass = mass
        self._model: Optional[osim.Model] = None
        self._actuators: List[osim.Actuator] = []
        
    @abstractmethod
    def build_device(self) -> osim.Model:
        """Build the OpenSim model of the device."""
        pass
    
    @abstractmethod
    def optimize_parameters(self, 
                          target_motion: str,
                          objective_function: str,
                          constraints: Dict[str, Union[float, List[float]]]) -> Dict[str, float]:
        """
        Optimize device parameters for a given motion.
        
        Args:
            target_motion: Path to the target motion file
            objective_function: Objective function to minimize
            constraints: Dictionary of optimization constraints
            
        Returns:
            Dictionary of optimized parameters
        """
        pass
    
    def attach_to_model(self, base_model: osim.Model) -> osim.Model:
        """
        Attach the device to an existing OpenSim model.
        
        Args:
            base_model: The OpenSim model to attach the device to
            
        Returns:
            Combined model with the device attached
        """
        if self._model is None:
            self._model = self.build_device()
            
        # Create a copy of the base model
        combined_model = osim.Model(base_model)
        
        # Add device components to the model
        for component in self._model.getComponentsList():
            combined_model.addComponent(component.clone())
            
        return combined_model
    
    def compute_assistance_metrics(self, 
                                 simulation_results: Dict[str, np.ndarray]
                                 ) -> Dict[str, float]:
        """
        Compute metrics to evaluate device assistance.
        
        Args:
            simulation_results: Dictionary containing simulation data
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {
            'metabolic_cost_reduction': 0.0,
            'peak_assistance_force': 0.0,
            'average_assistance': 0.0,
            'assistance_symmetry': 0.0
        }
        
        # Implementation will depend on specific device type
        return metrics 