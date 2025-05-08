"""
Example script demonstrating prosthetic leg optimization for walking.
"""

import os
import numpy as np
import opensim as osim
from src.assistive_devices.prosthetics.prosthetic_leg import ProstheticLeg

def main():
    # Create the prosthetic leg device
    prosthesis = ProstheticLeg(
        name="transtibial_prosthesis_v1",
        mass=3.0,  # kg
        amputation_level="transtibial",
        side="right",
        joint_properties={
            'ankle': {
                'stiffness': 400.0,  # Nm/rad
                'damping': 5.0,      # Nms/rad
                'range_min': -30.0,  # degrees
                'range_max': 30.0    # degrees
            }
        }
    )
    
    # Load the base human model
    model_path = os.path.join(os.path.dirname(__file__), 
                             "../../../data/models/gait2392_simbody.osim")
    base_model = osim.Model(model_path)
    
    # Attach the prosthesis to the model
    combined_model = prosthesis.attach_to_model(base_model)
    
    # Load example walking data
    motion_path = os.path.join(os.path.dirname(__file__),
                              "../../../data/motions/normal_walking.mot")
    
    # Define optimization constraints
    constraints = {
        'min_stiffness': 200.0,  # Nm/rad
        'max_stiffness': 800.0,  # Nm/rad
        'min_damping': 1.0,      # Nms/rad
        'max_damping': 10.0      # Nms/rad
    }
    
    # Optimize the prosthesis parameters
    optimized_params = prosthesis.optimize_parameters(
        target_motion=motion_path,
        objective_function="tracking_error",
        constraints=constraints
    )
    
    # Print optimization results
    print("\nOptimized Parameters:")
    for param, value in optimized_params.items():
        print(f"{param}: {value:.2f}")
    
    # Run a forward simulation with optimized parameters
    state = combined_model.initSystem()
    manager = osim.Manager(combined_model)
    manager.setInitialTime(0)
    manager.setFinalTime(1.0)  # Simulate one gait cycle
    
    # Collect simulation results
    time = np.linspace(0, 1.0, 101)
    ankle_angle = np.zeros(len(time))
    ankle_moment = np.zeros(len(time))
    grf = np.zeros((len(time), 3))  # 3D ground reaction force
    
    for i, t in enumerate(time):
        state = manager.getState()
        
        # Get joint kinematics and kinetics
        ankle_angle[i] = combined_model.getCoordinate("ankle_angle_r").getValue(state)
        ankle_moment[i] = prosthesis._joints[0].getCoordiante(0).getForce(state)
        
        # Get ground reaction forces
        for j in range(3):
            grf[i, j] = combined_model.getForceSet().get("ground_force_r").getRecordValues(state)[j]
        
        manager.integrate(t)
    
    # Compute performance metrics
    simulation_results = {
        'ankle_angle_right': ankle_angle,
        'ankle_moment_right': ankle_moment,
        'ground_reaction_force': grf
    }
    
    metrics = prosthesis.compute_assistance_metrics(simulation_results)
    
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
    
    # Additional analysis for prosthetic-specific metrics
    print("\nGait Analysis:")
    print(f"Stance Duration: {metrics['stance_duration']:.1f}% of gait cycle")
    print(f"Peak Loading Rate: {metrics['loading_rate']:.1f} N/s")
    print(f"Range of Motion: {metrics['ankle_range_of_motion']:.1f} degrees")

if __name__ == "__main__":
    main() 