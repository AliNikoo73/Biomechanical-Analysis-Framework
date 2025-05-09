"""
Example: Exoskeleton Optimization using ADOHRL Algorithm

This example demonstrates how to use the Adaptive Dual-Objective Hybrid Reinforcement Learning
algorithm to optimize an exoskeleton control policy that balances user comfort and efficiency.
"""

import numpy as np
import matplotlib.pyplot as plt
import opensim as osim
from typing import List, Dict, Tuple
import os
import sys
import torch

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from optimization.adaptive_dual_objective_rl import (
    AdaptiveDualObjectiveHRL, 
    ADOHRLConfig, 
    BiomechanicalConstraint
)
from assistive_devices.exoskeleton.knee_exo import KneeExoskeleton
from utils.visualization import plot_training_curves, visualize_gait_cycle
from utils.data_processing import normalize_joint_angles, filter_signals


def create_biomechanical_constraints() -> List[BiomechanicalConstraint]:
    """Create biomechanical constraints for the optimization."""
    
    def knee_angle_constraint(state):
        # Extract knee angle from state
        knee_angle = state[2]  # Assuming knee angle is the 3rd element
        return knee_angle
    
    def knee_moment_constraint(state):
        # Extract knee moment from state
        knee_moment = state[5]  # Assuming knee moment is the 6th element
        return knee_moment
    
    def metabolic_cost_constraint(state):
        # This would be a more complex function in reality
        # Here we use a simplified approach
        velocity = state[3]  # Assuming velocity is the 4th element
        force = state[4]     # Assuming force is the 5th element
        return velocity * force  # Power as a proxy for metabolic cost
    
    constraints = [
        BiomechanicalConstraint(
            name="Knee Angle",
            min_value=-0.1,  # Radians
            max_value=1.5,   # Radians
            weight=1.0,
            measurement_function=knee_angle_constraint
        ),
        BiomechanicalConstraint(
            name="Knee Moment",
            min_value=-50.0,  # Nm
            max_value=50.0,   # Nm
            weight=0.8,
            measurement_function=knee_moment_constraint
        ),
        BiomechanicalConstraint(
            name="Metabolic Cost",
            min_value=0.0,    # W
            max_value=300.0,  # W
            weight=0.6,
            measurement_function=metabolic_cost_constraint
        )
    ]
    
    return constraints


def create_opensim_model() -> osim.Model:
    """Create an OpenSim model for simulation."""
    
    model = osim.Model()
    model.setName("knee_exo_model")
    
    # Ground
    ground = model.getGround()
    
    # Create bodies
    thigh = osim.Body("thigh", 10.0, osim.Vec3(0), osim.Inertia(1.0, 1.0, 1.0))
    shank = osim.Body("shank", 5.0, osim.Vec3(0), osim.Inertia(0.5, 0.5, 0.5))
    foot = osim.Body("foot", 1.0, osim.Vec3(0), osim.Inertia(0.1, 0.1, 0.1))
    
    # Add bodies to model
    model.addBody(thigh)
    model.addBody(shank)
    model.addBody(foot)
    
    # Create joints
    hip_joint = osim.PinJoint(
        "hip", 
        ground, 
        osim.Vec3(0, 1.0, 0), 
        osim.Vec3(0), 
        thigh, 
        osim.Vec3(0, 0.5, 0), 
        osim.Vec3(0)
    )
    
    knee_joint = osim.PinJoint(
        "knee", 
        thigh, 
        osim.Vec3(0, -0.5, 0), 
        osim.Vec3(0), 
        shank, 
        osim.Vec3(0, 0.5, 0), 
        osim.Vec3(0)
    )
    
    ankle_joint = osim.PinJoint(
        "ankle", 
        shank, 
        osim.Vec3(0, -0.5, 0), 
        osim.Vec3(0), 
        foot, 
        osim.Vec3(0, 0.1, -0.1), 
        osim.Vec3(0)
    )
    
    # Add joints to model
    model.addJoint(hip_joint)
    model.addJoint(knee_joint)
    model.addJoint(ankle_joint)
    
    # Create muscles
    hip_flexor = osim.Millard2012EquilibriumMuscle(
        "hip_flexor", 
        1000.0,   # max isometric force
        0.1,      # optimal fiber length
        0.2,      # tendon slack length
        0.0       # pennation angle
    )
    
    hip_extensor = osim.Millard2012EquilibriumMuscle(
        "hip_extensor", 
        1500.0,   # max isometric force
        0.1,      # optimal fiber length
        0.2,      # tendon slack length
        0.0       # pennation angle
    )
    
    knee_flexor = osim.Millard2012EquilibriumMuscle(
        "knee_flexor", 
        1000.0,   # max isometric force
        0.1,      # optimal fiber length
        0.2,      # tendon slack length
        0.0       # pennation angle
    )
    
    knee_extensor = osim.Millard2012EquilibriumMuscle(
        "knee_extensor", 
        1500.0,   # max isometric force
        0.1,      # optimal fiber length
        0.2,      # tendon slack length
        0.0       # pennation angle
    )
    
    # Add path points to muscles
    hip_flexor.addNewPathPoint("hip_flexor_origin", ground, osim.Vec3(0, 1.0, 0.1))
    hip_flexor.addNewPathPoint("hip_flexor_insertion", thigh, osim.Vec3(0, 0.4, 0.1))
    
    hip_extensor.addNewPathPoint("hip_extensor_origin", ground, osim.Vec3(0, 1.0, -0.1))
    hip_extensor.addNewPathPoint("hip_extensor_insertion", thigh, osim.Vec3(0, 0.4, -0.1))
    
    knee_flexor.addNewPathPoint("knee_flexor_origin", thigh, osim.Vec3(0, -0.4, 0.1))
    knee_flexor.addNewPathPoint("knee_flexor_insertion", shank, osim.Vec3(0, 0.4, 0.1))
    
    knee_extensor.addNewPathPoint("knee_extensor_origin", thigh, osim.Vec3(0, -0.4, -0.1))
    knee_extensor.addNewPathPoint("knee_extensor_insertion", shank, osim.Vec3(0, 0.4, -0.1))
    
    # Add muscles to model
    model.addForce(hip_flexor)
    model.addForce(hip_extensor)
    model.addForce(knee_flexor)
    model.addForce(knee_extensor)
    
    # Exoskeleton actuator
    knee_exo_actuator = osim.CoordinateActuator("knee/knee_angle")
    knee_exo_actuator.setName("knee_exo_actuator")
    knee_exo_actuator.setOptimalForce(500.0)
    model.addForce(knee_exo_actuator)
    
    # Finalize model
    model.finalizeConnections()
    
    return model


def compute_comfort_reward(state: np.ndarray, action: np.ndarray) -> float:
    """
    Compute comfort reward based on user state and exoskeleton action.
    
    Args:
        state: Current state array
        action: Action array
    
    Returns:
        Comfort reward value
    """
    # Extract relevant state variables
    knee_angle = state[2]           # Knee angle
    knee_velocity = state[3]        # Knee angular velocity
    interaction_force = state[6]    # Human-exoskeleton interaction force
    
    # Extract action
    exo_torque = action[0]
    
    # Compute rewards
    alignment_reward = -np.abs(knee_angle - 0.3) * 2.0  # Reward being close to comfortable angle
    smoothness_reward = -np.abs(knee_velocity) * 0.5    # Reward smooth movements
    force_reward = -np.abs(interaction_force) * 0.3     # Penalize high interaction forces
    effort_reward = -np.abs(exo_torque) * 0.1           # Small penalty for high torques
    
    # Combined comfort reward
    comfort_reward = alignment_reward + smoothness_reward + force_reward + effort_reward
    
    # Normalize to [0, 1] range (assuming worst case is -5.0)
    normalized_reward = np.clip((comfort_reward + 5.0) / 5.0, 0.0, 1.0)
    
    return normalized_reward


def compute_efficiency_reward(state: np.ndarray, action: np.ndarray) -> float:
    """
    Compute efficiency reward based on user state and exoskeleton action.
    
    Args:
        state: Current state array
        action: Action array
    
    Returns:
        Efficiency reward value
    """
    # Extract relevant state variables
    knee_velocity = state[3]      # Knee angular velocity
    muscle_activation = state[7]  # Muscle activation (proxy for metabolic cost)
    
    # Extract action
    exo_torque = action[0]
    
    # Compute rewards
    metabolic_reward = -muscle_activation * 2.0        # Reward low muscle activation
    power_reward = knee_velocity * exo_torque * 0.5    # Reward positive power assistance
    consumption_reward = -np.abs(exo_torque) * 0.05    # Small penalty for energy consumption
    
    # Combined efficiency reward
    efficiency_reward = metabolic_reward + power_reward + consumption_reward
    
    # Normalize to [0, 1] range (assuming worst case is -3.0)
    normalized_reward = np.clip((efficiency_reward + 3.0) / 3.0, 0.0, 1.0)
    
    return normalized_reward


def simulate_step(model: osim.Model, state: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Simulate one step of the exoskeleton model.
    
    Args:
        model: OpenSim model
        state: Current state
        action: Control action
    
    Returns:
        Tuple of (next_state, done)
    """
    # In a real implementation, this would use OpenSim's ForwardDynamicsTool
    # For this example, we use a simplified simulation
    
    # Unpack state (example format - would be model-specific)
    # [time, hip_angle, knee_angle, knee_velocity, grf, knee_moment, interaction_force, muscle_activation]
    time = state[0]
    hip_angle = state[1]
    knee_angle = state[2]
    knee_velocity = state[3]
    grf = state[4]
    knee_moment = state[5]
    interaction_force = state[6]
    muscle_activation = state[7]
    
    # Apply action
    exo_torque = action[0]
    
    # Simple physics model (very simplified for example purposes)
    dt = 0.01  # Time step
    
    # Update knee kinematics
    knee_acceleration = (exo_torque + knee_moment - 0.1 * knee_velocity) / 0.5  # Simple dynamics
    knee_velocity_new = knee_velocity + knee_acceleration * dt
    knee_angle_new = knee_angle + knee_velocity_new * dt
    
    # Apply constraints
    knee_angle_new = np.clip(knee_angle_new, -0.1, 1.5)
    
    # Update hip (simplified)
    hip_angle_new = 0.5 * np.sin(time * 2.0)
    
    # Update other state variables
    time_new = time + dt
    interaction_force_new = 0.8 * interaction_force + 0.2 * np.abs(exo_torque - knee_moment)
    muscle_activation_new = 0.9 * muscle_activation + 0.1 * (1.0 - np.clip(np.abs(exo_torque / knee_moment), 0, 1))
    grf_new = grf * 0.9 + 0.1 * np.random.normal(500, 50)
    knee_moment_new = 0.9 * knee_moment + 0.1 * np.random.normal(20, 5)
    
    # Assemble next state
    next_state = np.array([
        time_new, 
        hip_angle_new, 
        knee_angle_new, 
        knee_velocity_new, 
        grf_new, 
        knee_moment_new, 
        interaction_force_new, 
        muscle_activation_new
    ])
    
    # Check if episode is done
    done = time_new >= 5.0  # End episode after 5 seconds
    
    return next_state, done


def run_optimization():
    """Run the exoskeleton optimization using ADOHRL algorithm."""
    print("Starting exoskeleton optimization using ADOHRL algorithm...")
    
    # Create OpenSim model
    model = create_opensim_model()
    
    # Define state and action dimensions
    state_dim = 8  # [time, hip_angle, knee_angle, knee_velocity, grf, knee_moment, interaction_force, muscle_activation]
    action_dim = 1  # [exo_torque]
    
    # Create biomechanical constraints
    constraints = create_biomechanical_constraints()
    
    # Create configuration for ADOHRL
    config = ADOHRLConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        adaptation_rate=0.02,
        comfort_coefficient=0.6,  # Prioritize comfort slightly over efficiency
        efficiency_coefficient=0.4,
        hidden_dims=[128, 128],
        update_interval=10,
        batch_size=32,
        learning_rate=1e-4
    )
    
    # Create ADOHRL agent
    agent = AdaptiveDualObjectiveHRL(
        config=config,
        biomechanical_constraints=constraints,
        opensim_model=model
    )
    
    # Training loop
    total_episodes = 50
    max_steps_per_episode = 500
    
    episode_rewards = []
    comfort_rewards = []
    efficiency_rewards = []
    user_preferences = []
    constraint_violations = []
    
    for episode in range(total_episodes):
        # Initialize state
        state = np.array([0.0, 0.0, 0.2, 0.0, 500.0, 20.0, 10.0, 0.5])
        
        episode_reward = 0
        episode_comfort = 0
        episode_efficiency = 0
        episode_violations = 0
        
        for step in range(max_steps_per_episode):
            # Select action
            action = agent.select_action(state)
            
            # Apply action and get next state
            next_state, done = simulate_step(model, state, action)
            
            # Calculate rewards
            comfort_reward = compute_comfort_reward(state, action)
            efficiency_reward = compute_efficiency_reward(state, action)
            
            # Calculate constraint values
            constraint_values = [
                constraint.measurement_function(state) 
                for constraint in constraints
            ]
            
            # Check for constraint violations
            violations = 0
            for i, constraint in enumerate(constraints):
                value = constraint_values[i]
                if value < constraint.min_value or value > constraint.max_value:
                    violations += 1
            
            episode_violations += violations
            
            # Add experience to agent's memory
            agent.add_experience(
                state=state,
                action=action,
                comfort_reward=comfort_reward,
                efficiency_reward=efficiency_reward,
                next_state=next_state,
                done=done,
                constraint_values=constraint_values
            )
            
            # Update state
            state = next_state
            
            # Update metrics
            combined_reward = (
                config.comfort_coefficient * comfort_reward + 
                config.efficiency_coefficient * efficiency_reward
            )
            episode_reward += combined_reward
            episode_comfort += comfort_reward
            episode_efficiency += efficiency_reward
            
            # Simulate user feedback every 50 steps
            if step % 50 == 0:
                # Simulated user feedback (could come from real user in practice)
                # Here we simulate a user who prefers comfort at the beginning
                # but gradually shifts to efficiency
                user_comfort_feedback = 0.8 - 0.4 * (episode / total_episodes)
                user_efficiency_feedback = 0.2 + 0.4 * (episode / total_episodes)
                
                # Update user preference
                agent.update_preference(user_comfort_feedback, user_efficiency_feedback)
            
            if done:
                break
        
        # Update metrics
        avg_reward = episode_reward / (step + 1)
        avg_comfort = episode_comfort / (step + 1)
        avg_efficiency = episode_efficiency / (step + 1)
        avg_violations = episode_violations / (step + 1)
        
        episode_rewards.append(avg_reward)
        comfort_rewards.append(avg_comfort)
        efficiency_rewards.append(avg_efficiency)
        constraint_violations.append(avg_violations)
        user_preferences.append(agent.network.user_preference.item())
        
        # Print progress
        print(f"Episode {episode+1}/{total_episodes}: " + 
              f"Reward={avg_reward:.3f}, " +
              f"Comfort={avg_comfort:.3f}, " +
              f"Efficiency={avg_efficiency:.3f}, " +
              f"Violations={avg_violations:.3f}, " +
              f"Preference={agent.network.user_preference.item():.3f}")
    
    # Save model
    agent.save("exoskeleton_control_policy.pt")
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Combined Reward')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    plt.subplot(2, 2, 2)
    plt.plot(comfort_rewards, label='Comfort')
    plt.plot(efficiency_rewards, label='Efficiency')
    plt.title('Comfort vs Efficiency')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(user_preferences)
    plt.title('User Preference (Comfort Weight)')
    plt.xlabel('Episode')
    plt.ylabel('Preference')
    
    plt.subplot(2, 2, 4)
    plt.plot(constraint_violations)
    plt.title('Constraint Violations')
    plt.xlabel('Episode')
    plt.ylabel('Average Violations per Step')
    
    plt.tight_layout()
    plt.savefig('exoskeleton_optimization_results.png')
    
    print("Optimization complete. Results saved.")


if __name__ == "__main__":
    run_optimization() 