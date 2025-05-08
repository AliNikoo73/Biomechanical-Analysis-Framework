"""
Example script demonstrating ankle exoskeleton optimization for walking assistance.
"""

import os
import numpy as np
import opensim as osim
from src.assistive_devices.exoskeleton.ankle_exo import AnkleExoskeleton

def main():
    # Create the exoskeleton device
    exo = AnkleExoskeleton(
        name="ankle_exo_v1",
        mass=1.2,  # kg
        max_torque=45.0,  # Nm
        attachment_points={
            'shank': [0.0, -0.2, 0.0],  # 20cm below knee
            'foot': [0.15, 0.02, 0.0]   # 15cm anterior to ankle
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
        'max_torque': 45.0,  # Nm
        'timing_bounds': [0.0, 60.0],  # % gait cycle
        'power_limit': 120.0  # W
    }
    
    # Optimize the exoskeleton parameters
    optimized_params = exo.optimize_parameters(
        target_motion=motion_path,
        objective_function="metabolic_cost",
        constraints=constraints
    )
    
    # Print optimization results
    print("\nOptimization Results:")
    print(f"Peak Torque: {optimized_params['peak_torque']:.1f} Nm")
    print(f"Assistance Timing: {optimized_params['timing'][0]:.1f}% - {optimized_params['timing'][1]:.1f}% gait cycle")
    print(f"Metabolic Cost Reduction: {optimized_params['metabolic_reduction']:.1f}%")
    
    # Run a forward simulation with optimized parameters
    state = combined_model.initSystem()
    manager = osim.Manager(combined_model)
    manager.setInitialTime(0)
    manager.setFinalTime(1.0)  # Simulate one gait cycle
    
    # Collect simulation results
    time = np.linspace(0, 1.0, 101)
    ankle_angle = np.zeros(len(time))
    exo_torque = np.zeros(len(time))
    
    for i, t in enumerate(time):
        state = manager.getState()
        ankle_angle[i] = combined_model.getCoordinate("ankle_angle_r").getValue(state)
        exo_torque[i] = exo._actuators[0].getForce(state)
        manager.integrate(t)
    
    # Compute assistance metrics
    simulation_results = {
        'ankle_angle': ankle_angle,
        'exo_torque': exo_torque
    }
    
    metrics = exo.compute_assistance_metrics(simulation_results)
    
    print("\nAssistance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

if __name__ == "__main__":
    main() 