from typing import Dict, List, Union
import numpy as np
import opensim as osim
from ..base import AssistiveDevice

class AnkleExoskeleton(AssistiveDevice):
    """Ankle exoskeleton assistive device implementation."""
    
    def __init__(self, name: str, mass: float = 1.0,
                 max_torque: float = 50.0,
                 attachment_points: Dict[str, List[float]] = None):
        """
        Initialize the ankle exoskeleton.
        
        Args:
            name: Name of the device
            mass: Mass of the device in kg
            max_torque: Maximum torque the device can provide in Nm
            attachment_points: Dictionary of attachment points coordinates
        """
        super().__init__(name, mass)
        self.max_torque = max_torque
        self.attachment_points = attachment_points or {
            'shank': [0.0, 0.0, 0.0],
            'foot': [0.0, 0.0, 0.0]
        }
        
    def build_device(self) -> osim.Model:
        """Build the OpenSim model of the ankle exoskeleton."""
        model = osim.Model()
        model.setName(f"{self.name}_model")
        
        # Create main physical components
        shank_frame = osim.PhysicalOffsetFrame()
        shank_frame.setName("shank_attachment")
        
        foot_frame = osim.PhysicalOffsetFrame()
        foot_frame.setName("foot_attachment")
        
        # Create the actuator
        actuator = osim.TorqueActuator()
        actuator.setName("ankle_assistance")
        actuator.set_max_torque(self.max_torque)
        actuator.setBodyA(shank_frame)
        actuator.setBodyB(foot_frame)
        
        # Add components to model
        model.addComponent(shank_frame)
        model.addComponent(foot_frame)
        model.addComponent(actuator)
        
        self._model = model
        self._actuators.append(actuator)
        
        return model
    
    def optimize_parameters(self,
                          target_motion: str,
                          objective_function: str = "metabolic_cost",
                          constraints: Dict[str, Union[float, List[float]]] = None
                          ) -> Dict[str, float]:
        """
        Optimize exoskeleton parameters for a given motion.
        
        Args:
            target_motion: Path to the target motion file
            objective_function: Objective function to minimize
            constraints: Dictionary of optimization constraints
            
        Returns:
            Dictionary of optimized parameters
        """
        if constraints is None:
            constraints = {
                'max_torque': self.max_torque,
                'timing_bounds': [0.0, 100.0],
                'power_limit': 150.0
            }
            
        # Setup optimization problem
        study = osim.MocoStudy()
        problem = study.updProblem()
        
        # Add optimization parameters
        torque_pattern = osim.MocoParameter()
        torque_pattern.setName("assistance_pattern")
        problem.addParameter(torque_pattern)
        
        # Add goals
        if objective_function == "metabolic_cost":
            goal = osim.MocoMetabolicCost()
        elif objective_function == "mechanical_work":
            goal = osim.MocoControlGoal()
        else:
            raise ValueError(f"Unsupported objective function: {objective_function}")
            
        problem.addGoal(goal)
        
        # Solve the optimization
        solution = study.solve()
        
        # Extract optimized parameters
        optimized_params = {
            'peak_torque': float(solution.getParameter("assistance_pattern")),
            'timing': [0.0, 100.0],  # Placeholder for actual timing parameters
            'metabolic_reduction': float(solution.getObjective())
        }
        
        return optimized_params
    
    def compute_assistance_metrics(self,
                                 simulation_results: Dict[str, np.ndarray]
                                 ) -> Dict[str, float]:
        """
        Compute metrics specific to ankle exoskeleton assistance.
        
        Args:
            simulation_results: Dictionary containing simulation data
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = super().compute_assistance_metrics(simulation_results)
        
        # Add exoskeleton-specific metrics
        if 'ankle_angle' in simulation_results and 'exo_torque' in simulation_results:
            angle = simulation_results['ankle_angle']
            torque = simulation_results['exo_torque']
            
            metrics.update({
                'peak_assistance_torque': float(np.max(np.abs(torque))),
                'average_power': float(np.mean(np.abs(torque * np.gradient(angle)))),
                'assistance_timing': float(np.argmax(torque) / len(torque) * 100)
            })
            
        return metrics 