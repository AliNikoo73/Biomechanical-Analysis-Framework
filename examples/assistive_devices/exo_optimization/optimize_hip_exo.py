"""
Example script demonstrating hip exoskeleton optimization for walking assistance.
"""

import os
import numpy as np
import opensim as osim
from src.assistive_devices.exoskeleton.hip_exo import HipExoskeleton

def main():
    # Create the hip exoskeleton device
    exo = HipExoskeleton(
        name="hip_exo_v1",
        mass=2.5,  # kg
        max_torque=70.0,  # Nm
        bilateral=True,
        attachment_points={
            'pelvis': [0.0, 0.0, 0.0],
            'thigh_r': [0.0, -0.15, 0.0],  # 15cm below hip joint
            'thigh_l': [0.0, -0.15, 0.0]
        }
    )
    
    # Load the base human model
    model_path = os.path.join(os.path.dirname(__file__), 
                             "../../../data/models/gait2392_simbody.osim")
    base_model = osim.Model(model_path)
    
    # Attach the exoskeleton to the model
    combined_model = exo.attach_to_model(base_model)
    
    # Load example walking data
    motion_path = os.path.join(os.path.dirname(__file__),
                              "../../../data/motions/normal_walking.mot")
    
    # Define optimization constraints
    constraints = {
        'max_torque': 70.0,  # Nm
        'timing_bounds': [0.0, 100.0],  # % gait cycle
        'power_limit': 200.0,  # W
        'symmetry_threshold': 0.1  # 10% asymmetry allowed
    }
    
    # Optimize the exoskeleton parameters
    optimized_params = exo.optimize_parameters(
        target_motion=motion_path,
        objective_function="metabolic_cost",
        constraints=constraints
    )
    
    # Print optimization results
    print("\nOptimization Results:")
    print(f"Right Hip Peak Torque: {optimized_params['peak_torque_r']:.1f} Nm")
    print(f"Left Hip Peak Torque: {optimized_params['peak_torque_l']:.1f} Nm")
    print(f"Symmetry Index: {optimized_params['symmetry_index']:.3f}")
    
    # Run a forward simulation with optimized parameters
    state = combined_model.initSystem()
    manager = osim.Manager(combined_model)
    manager.setInitialTime(0)
    manager.setFinalTime(1.0)  # Simulate one gait cycle
    
    # Collect simulation results
    time = np.linspace(0, 1.0, 101)
    hip_angle_r = np.zeros(len(time))
    hip_angle_l = np.zeros(len(time))
    exo_torque_r = np.zeros(len(time))
    exo_torque_l = np.zeros(len(time))
    
    for i, t in enumerate(time):
        state = manager.getState()
        
        # Get right side data
        hip_angle_r[i] = combined_model.getCoordinate("hip_flexion_r").getValue(state)
        exo_torque_r[i] = exo._actuators[0].getForce(state)
        
        # Get left side data
        hip_angle_l[i] = combined_model.getCoordinate("hip_flexion_l").getValue(state)
        exo_torque_l[i] = exo._actuators[1].getForce(state)
        
        manager.integrate(t)
    
    # Compute assistance metrics
    simulation_results = {
        'hip_angle_r': hip_angle_r,
        'hip_angle_l': hip_angle_l,
        'exo_torque_r': exo_torque_r,
        'exo_torque_l': exo_torque_l
    }
    
    metrics = exo.compute_assistance_metrics(simulation_results)
    
    print("\nAssistance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

if __name__ == "__main__":
    main() 