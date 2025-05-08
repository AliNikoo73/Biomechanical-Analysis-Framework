from typing import Dict, List, Union
import numpy as np
import opensim as osim
from ..base import AssistiveDevice

class HipExoskeleton(AssistiveDevice):
    """Hip exoskeleton assistive device implementation."""
    
    def __init__(self, name: str, mass: float = 2.0,
                 max_torque: float = 70.0,
                 bilateral: bool = True,
                 attachment_points: Dict[str, List[float]] = None):
        """
        Initialize the hip exoskeleton.
        
        Args:
            name: Name of the device
            mass: Mass of the device in kg
            max_torque: Maximum torque the device can provide in Nm
            bilateral: Whether to create a bilateral device
            attachment_points: Dictionary of attachment points coordinates
        """
        super().__init__(name, mass)
        self.max_torque = max_torque
        self.bilateral = bilateral
        self.attachment_points = attachment_points or {
            'pelvis': [0.0, 0.0, 0.0],
            'thigh_r': [0.0, -0.15, 0.0],  # 15cm below hip joint
            'thigh_l': [0.0, -0.15, 0.0]
        }
        
    def build_device(self) -> osim.Model:
        """Build the OpenSim model of the hip exoskeleton."""
        model = osim.Model()
        model.setName(f"{self.name}_model")
        
        # Create main physical components
        pelvis_frame = osim.PhysicalOffsetFrame()
        pelvis_frame.setName("pelvis_attachment")
        
        # Create components for right side
        thigh_frame_r = osim.PhysicalOffsetFrame()
        thigh_frame_r.setName("thigh_attachment_r")
        
        actuator_r = osim.TorqueActuator()
        actuator_r.setName("hip_assistance_r")
        actuator_r.set_max_torque(self.max_torque)
        actuator_r.setBodyA(pelvis_frame)
        actuator_r.setBodyB(thigh_frame_r)
        
        # Add right side components
        model.addComponent(pelvis_frame)
        model.addComponent(thigh_frame_r)
        model.addComponent(actuator_r)
        self._actuators.append(actuator_r)
        
        # Create components for left side if bilateral
        if self.bilateral:
            thigh_frame_l = osim.PhysicalOffsetFrame()
            thigh_frame_l.setName("thigh_attachment_l")
            
            actuator_l = osim.TorqueActuator()
            actuator_l.setName("hip_assistance_l")
            actuator_l.set_max_torque(self.max_torque)
            actuator_l.setBodyA(pelvis_frame)
            actuator_l.setBodyB(thigh_frame_l)
            
            # Add left side components
            model.addComponent(thigh_frame_l)
            model.addComponent(actuator_l)
            self._actuators.append(actuator_l)
        
        self._model = model
        return model
    
    def optimize_parameters(self,
                          target_motion: str,
                          objective_function: str = "metabolic_cost",
                          constraints: Dict[str, Union[float, List[float]]] = None
                          ) -> Dict[str, float]:
        """
        Optimize hip exoskeleton parameters for a given motion.
        
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
                'power_limit': 200.0,
                'symmetry_threshold': 0.1  # 10% asymmetry allowed
            }
            
        # Setup optimization problem
        study = osim.MocoStudy()
        problem = study.updProblem()
        
        # Add optimization parameters
        for side in ['r', 'l'] if self.bilateral else ['r']:
            torque_pattern = osim.MocoParameter()
            torque_pattern.setName(f"assistance_pattern_{side}")
            problem.addParameter(torque_pattern)
        
        # Add goals
        if objective_function == "metabolic_cost":
            goal = osim.MocoMetabolicCost()
        elif objective_function == "mechanical_work":
            goal = osim.MocoControlGoal()
        elif objective_function == "tracking_error":
            goal = osim.MocoStateTrackingGoal()
        else:
            raise ValueError(f"Unsupported objective function: {objective_function}")
            
        problem.addGoal(goal)
        
        # Add symmetry constraint for bilateral devices
        if self.bilateral:
            symmetry_constraint = osim.MocoControlBoundConstraint()
            problem.addPathConstraint(symmetry_constraint)
        
        # Solve the optimization
        solution = study.solve()
        
        # Extract optimized parameters
        optimized_params = {
            'peak_torque_r': float(solution.getParameter("assistance_pattern_r")),
            'timing_r': [0.0, 100.0]
        }
        
        if self.bilateral:
            optimized_params.update({
                'peak_torque_l': float(solution.getParameter("assistance_pattern_l")),
                'timing_l': [0.0, 100.0],
                'symmetry_index': float(solution.getObjective())
            })
        
        return optimized_params
    
    def compute_assistance_metrics(self,
                                 simulation_results: Dict[str, np.ndarray]
                                 ) -> Dict[str, float]:
        """
        Compute metrics specific to hip exoskeleton assistance.
        
        Args:
            simulation_results: Dictionary containing simulation data
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = super().compute_assistance_metrics(simulation_results)
        
        # Process right side metrics
        if all(k in simulation_results for k in ['hip_angle_r', 'exo_torque_r']):
            angle_r = simulation_results['hip_angle_r']
            torque_r = simulation_results['exo_torque_r']
            
            metrics.update({
                'peak_assistance_torque_r': float(np.max(np.abs(torque_r))),
                'average_power_r': float(np.mean(np.abs(torque_r * np.gradient(angle_r)))),
                'assistance_timing_r': float(np.argmax(torque_r) / len(torque_r) * 100)
            })
        
        # Process left side metrics if bilateral
        if self.bilateral and all(k in simulation_results for k in ['hip_angle_l', 'exo_torque_l']):
            angle_l = simulation_results['hip_angle_l']
            torque_l = simulation_results['exo_torque_l']
            
            metrics.update({
                'peak_assistance_torque_l': float(np.max(np.abs(torque_l))),
                'average_power_l': float(np.mean(np.abs(torque_l * np.gradient(angle_l)))),
                'assistance_timing_l': float(np.argmax(torque_l) / len(torque_l) * 100)
            })
            
            # Compute symmetry metrics for bilateral devices
            metrics['torque_symmetry'] = float(
                np.mean(np.abs(torque_r - torque_l)) / np.mean(np.abs(torque_r + torque_l))
            )
            metrics['power_symmetry'] = float(
                np.mean(np.abs(angle_r * torque_r - angle_l * torque_l)) /
                np.mean(np.abs(angle_r * torque_r + angle_l * torque_l))
            )
        
        return metrics 