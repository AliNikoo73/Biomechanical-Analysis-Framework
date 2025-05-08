from typing import Dict, List, Union, Optional
import numpy as np
import opensim as osim
from ..base import AssistiveDevice

class ProstheticLeg(AssistiveDevice):
    """Prosthetic leg implementation with configurable joints and components."""
    
    def __init__(self, name: str, mass: float = 3.0,
                 amputation_level: str = "transtibial",
                 side: str = "right",
                 joint_properties: Dict[str, Dict[str, float]] = None):
        """
        Initialize the prosthetic leg.
        
        Args:
            name: Name of the device
            mass: Mass of the device in kg
            amputation_level: Level of amputation ('transtibial' or 'transfemoral')
            side: Side of amputation ('right' or 'left')
            joint_properties: Dictionary of joint properties (stiffness, damping, etc.)
        """
        super().__init__(name, mass)
        self.amputation_level = amputation_level.lower()
        self.side = side.lower()
        
        if self.amputation_level not in ["transtibial", "transfemoral"]:
            raise ValueError("Amputation level must be 'transtibial' or 'transfemoral'")
        
        if self.side not in ["right", "left"]:
            raise ValueError("Side must be 'right' or 'left'")
        
        # Default joint properties
        self.joint_properties = joint_properties or {
            'ankle': {
                'stiffness': 400.0,  # Nm/rad
                'damping': 5.0,      # Nms/rad
                'range_min': -30.0,  # degrees
                'range_max': 30.0    # degrees
            },
            'knee': {
                'stiffness': 300.0,
                'damping': 3.0,
                'range_min': 0.0,
                'range_max': 120.0
            }
        }
        
        self._segments: List[osim.Body] = []
        self._joints: List[osim.Joint] = []
        
    def build_device(self) -> osim.Model:
        """Build the OpenSim model of the prosthetic leg."""
        model = osim.Model()
        model.setName(f"{self.name}_model")
        
        # Create ground reference
        ground = model.getGround()
        
        # Build segments based on amputation level
        if self.amputation_level == "transtibial":
            # Create pylon and foot segments
            pylon = self._create_segment("pylon", 1.0, [0.05, 0.2, 0.05])
            foot = self._create_segment("foot", 0.8, [0.22, 0.05, 0.08])
            
            # Create ankle joint
            ankle = self._create_prosthetic_joint(
                "ankle", pylon, foot, 
                self.joint_properties['ankle']
            )
            
            # Add components to model
            model.addBody(pylon)
            model.addBody(foot)
            model.addJoint(ankle)
            
        else:  # transfemoral
            # Create thigh, pylon, and foot segments
            thigh = self._create_segment("thigh", 1.2, [0.08, 0.25, 0.08])
            shank = self._create_segment("shank", 1.0, [0.05, 0.2, 0.05])
            foot = self._create_segment("foot", 0.8, [0.22, 0.05, 0.08])
            
            # Create knee and ankle joints
            knee = self._create_prosthetic_joint(
                "knee", thigh, shank,
                self.joint_properties['knee']
            )
            ankle = self._create_prosthetic_joint(
                "ankle", shank, foot,
                self.joint_properties['ankle']
            )
            
            # Add components to model
            model.addBody(thigh)
            model.addBody(shank)
            model.addBody(foot)
            model.addJoint(knee)
            model.addJoint(ankle)
        
        self._model = model
        return model
    
    def _create_segment(self, name: str, mass: float, dimensions: List[float]) -> osim.Body:
        """Create a prosthetic segment with specified properties."""
        segment = osim.Body()
        segment.setName(f"{name}_{self.side}")
        segment.setMass(mass)
        
        # Create and attach geometry
        geometry = osim.Brick()
        geometry.setHalfLengths(osim.Vec3(dimensions[0], dimensions[1], dimensions[2]))
        
        segment.attachGeometry(geometry)
        self._segments.append(segment)
        
        return segment
    
    def _create_prosthetic_joint(self, name: str, parent: osim.Body, child: osim.Body,
                                properties: Dict[str, float]) -> osim.CustomJoint:
        """Create a prosthetic joint with specified properties."""
        joint = osim.CustomJoint(f"{name}_{self.side}")
        joint.connectSocket_parent(parent)
        joint.connectSocket_child(child)
        
        # Add spring and damper forces
        spring = osim.SpringGeneralizedForce()
        spring.setName(f"{name}_spring_{self.side}")
        spring.setStiffness(properties['stiffness'])
        spring.setRestLength(0.0)
        
        damper = osim.DampingGeneralizedForce()
        damper.setName(f"{name}_damper_{self.side}")
        damper.setDamping(properties['damping'])
        
        # Add coordinate limits
        coord = joint.updCoordinate()
        coord.setName(f"{name}_angle_{self.side}")
        coord.setRangeMin(np.deg2rad(properties['range_min']))
        coord.setRangeMax(np.deg2rad(properties['range_max']))
        
        self._joints.append(joint)
        return joint
    
    def optimize_parameters(self,
                          target_motion: str,
                          objective_function: str = "tracking_error",
                          constraints: Dict[str, Union[float, List[float]]] = None
                          ) -> Dict[str, float]:
        """
        Optimize prosthetic parameters for a given motion.
        
        Args:
            target_motion: Path to the target motion file
            objective_function: Objective function to minimize
            constraints: Dictionary of optimization constraints
            
        Returns:
            Dictionary of optimized parameters
        """
        if constraints is None:
            constraints = {
                'min_stiffness': 100.0,
                'max_stiffness': 1000.0,
                'min_damping': 0.1,
                'max_damping': 20.0
            }
            
        # Setup optimization problem
        study = osim.MocoStudy()
        problem = study.updProblem()
        
        # Add parameters to optimize
        for joint in self._joints:
            name = joint.getName().split('_')[0]  # Get base joint name
            
            # Stiffness parameter
            stiffness = osim.MocoParameter()
            stiffness.setName(f"{name}_stiffness")
            stiffness.setBounds([constraints['min_stiffness'], 
                               constraints['max_stiffness']])
            problem.addParameter(stiffness)
            
            # Damping parameter
            damping = osim.MocoParameter()
            damping.setName(f"{name}_damping")
            damping.setBounds([constraints['min_damping'],
                             constraints['max_damping']])
            problem.addParameter(damping)
        
        # Add optimization goal
        if objective_function == "tracking_error":
            goal = osim.MocoStateTrackingGoal()
            goal.setReference(target_motion)
        elif objective_function == "metabolic_cost":
            goal = osim.MocoMetabolicCost()
        elif objective_function == "symmetry":
            goal = osim.MocoControlGoal()
        else:
            raise ValueError(f"Unsupported objective function: {objective_function}")
            
        problem.addGoal(goal)
        
        # Solve the optimization
        solution = study.solve()
        
        # Extract optimized parameters
        optimized_params = {}
        for joint in self._joints:
            name = joint.getName().split('_')[0]
            optimized_params[f"{name}_stiffness"] = float(
                solution.getParameter(f"{name}_stiffness")
            )
            optimized_params[f"{name}_damping"] = float(
                solution.getParameter(f"{name}_damping")
            )
        
        return optimized_params
    
    def compute_assistance_metrics(self,
                                 simulation_results: Dict[str, np.ndarray]
                                 ) -> Dict[str, float]:
        """
        Compute metrics specific to prosthetic leg performance.
        
        Args:
            simulation_results: Dictionary containing simulation data
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = super().compute_assistance_metrics(simulation_results)
        
        # Process joint-specific metrics
        for joint in self._joints:
            name = joint.getName().split('_')[0]
            angle_key = f"{name}_angle_{self.side}"
            moment_key = f"{name}_moment_{self.side}"
            
            if angle_key in simulation_results and moment_key in simulation_results:
                angle = simulation_results[angle_key]
                moment = simulation_results[moment_key]
                
                metrics.update({
                    f"{name}_peak_moment": float(np.max(np.abs(moment))),
                    f"{name}_range_of_motion": float(np.ptp(angle)),
                    f"{name}_average_power": float(
                        np.mean(np.abs(moment * np.gradient(angle)))
                    )
                })
        
        # Compute additional prosthetic-specific metrics
        if 'ground_reaction_force' in simulation_results:
            grf = simulation_results['ground_reaction_force']
            metrics.update({
                'peak_vertical_force': float(np.max(grf[:, 1])),
                'loading_rate': float(np.max(np.gradient(grf[:, 1]))),
                'stance_duration': float(
                    np.sum(grf[:, 1] > 20) / len(grf) * 100  # % of gait cycle
                )
            })
        
        return metrics 