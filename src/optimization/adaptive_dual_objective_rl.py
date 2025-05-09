"""
Adaptive Dual-Objective Hybrid Reinforcement Learning (ADOHRL)

A patentable algorithm for personalized assistive device control that combines 
reinforcement learning with biomechanical constraints and adaptation mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import opensim as osim
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass

@dataclass
class BiomechanicalConstraint:
    """Represents a biomechanical constraint for the optimization process"""
    name: str
    min_value: float
    max_value: float
    weight: float
    measurement_function: Callable


class ADOHRLConfig:
    """Configuration for the ADOHRL algorithm"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        adaptation_rate: float = 0.01,
        entropy_coefficient: float = 0.01,
        comfort_coefficient: float = 0.5,
        efficiency_coefficient: float = 0.5,
        hidden_dims: List[int] = [256, 256],
        update_interval: int = 5,
        memory_capacity: int = 10000,
        batch_size: int = 64,
        learning_rate: float = 3e-4,
    ):
        """
        Initialize the ADOHRL configuration.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            adaptation_rate: Rate at which the algorithm adapts to user-specific parameters
            entropy_coefficient: Weight for the entropy term in the objective function
            comfort_coefficient: Weight for the comfort objective
            efficiency_coefficient: Weight for the efficiency objective
            hidden_dims: Dimensions of hidden layers in the neural networks
            update_interval: Interval (in steps) between policy updates
            memory_capacity: Capacity of experience replay buffer
            batch_size: Batch size for neural network updates
            learning_rate: Learning rate for neural network optimization
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.adaptation_rate = adaptation_rate
        self.entropy_coefficient = entropy_coefficient
        self.comfort_coefficient = comfort_coefficient
        self.efficiency_coefficient = efficiency_coefficient
        self.hidden_dims = hidden_dims
        self.update_interval = update_interval
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.learning_rate = learning_rate


class DualObjectiveNetwork(nn.Module):
    """
    Neural network architecture for ADOHRL that optimizes both comfort and efficiency.
    """
    
    def __init__(self, config: ADOHRLConfig):
        """
        Initialize the dual-objective network.
        
        Args:
            config: Configuration for the network
        """
        super().__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential()
        prev_dim = config.state_dim
        for i, dim in enumerate(config.hidden_dims[:1]):
            self.feature_extractor.add_module(f"shared_linear_{i}", nn.Linear(prev_dim, dim))
            self.feature_extractor.add_module(f"shared_relu_{i}", nn.ReLU())
            prev_dim = dim
        
        # Comfort branch
        self.comfort_branch = nn.Sequential()
        prev_dim = config.hidden_dims[0]
        for i, dim in enumerate(config.hidden_dims[1:]):
            self.comfort_branch.add_module(f"comfort_linear_{i}", nn.Linear(prev_dim, dim))
            self.comfort_branch.add_module(f"comfort_relu_{i}", nn.ReLU())
            prev_dim = dim
        
        # Efficiency branch
        self.efficiency_branch = nn.Sequential()
        prev_dim = config.hidden_dims[0]
        for i, dim in enumerate(config.hidden_dims[1:]):
            self.efficiency_branch.add_module(f"efficiency_linear_{i}", nn.Linear(prev_dim, dim))
            self.efficiency_branch.add_module(f"efficiency_relu_{i}", nn.ReLU())
            prev_dim = dim
        
        # Output layers
        self.comfort_action_mean = nn.Linear(prev_dim, config.action_dim)
        self.efficiency_action_mean = nn.Linear(prev_dim, config.action_dim)
        self.log_std = nn.Parameter(torch.zeros(config.action_dim))
        
        # Value function
        self.comfort_value = nn.Linear(prev_dim, 1)
        self.efficiency_value = nn.Linear(prev_dim, 1)
        
        # User adaptation parameters
        self.user_preference = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: Current state tensor
            
        Returns:
            Dictionary containing action means and values for both objectives
        """
        shared_features = self.feature_extractor(state)
        
        comfort_features = self.comfort_branch(shared_features)
        efficiency_features = self.efficiency_branch(shared_features)
        
        comfort_action_mean = self.comfort_action_mean(comfort_features)
        efficiency_action_mean = self.efficiency_action_mean(efficiency_features)
        
        comfort_value = self.comfort_value(comfort_features)
        efficiency_value = self.efficiency_value(efficiency_features)
        
        # Adaptive action mean based on user preference
        preference = torch.sigmoid(self.user_preference)
        combined_action_mean = (
            preference * comfort_action_mean + 
            (1 - preference) * efficiency_action_mean
        )
        
        return {
            "comfort_action_mean": comfort_action_mean,
            "efficiency_action_mean": efficiency_action_mean,
            "combined_action_mean": combined_action_mean,
            "comfort_value": comfort_value,
            "efficiency_value": efficiency_value,
            "user_preference": preference
        }
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Sample an action from the policy.
        
        Args:
            state: Current state tensor
            deterministic: Whether to return deterministic action
            
        Returns:
            Sampled action tensor
        """
        output = self.forward(state)
        
        if deterministic:
            return output["combined_action_mean"]
        
        std = torch.exp(self.log_std)
        normal = torch.distributions.Normal(output["combined_action_mean"], std)
        action = normal.sample()
        return action


class AdaptiveDualObjectiveHRL:
    """
    Implementation of the ADOHRL algorithm for personalized assistive device control.
    """
    
    def __init__(
        self, 
        config: ADOHRLConfig,
        biomechanical_constraints: List[BiomechanicalConstraint],
        opensim_model: Optional[osim.Model] = None
    ):
        """
        Initialize the ADOHRL algorithm.
        
        Args:
            config: Algorithm configuration
            biomechanical_constraints: List of biomechanical constraints
            opensim_model: OpenSim model for biomechanical simulations
        """
        self.config = config
        self.constraints = biomechanical_constraints
        self.opensim_model = opensim_model
        
        # Initialize neural network
        self.network = DualObjectiveNetwork(config)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), 
            lr=config.learning_rate
        )
        
        # Experience replay buffer
        self.memory = []
        
        # Learning statistics
        self.comfort_losses = []
        self.efficiency_losses = []
        self.constraint_violations = []
        self.user_preferences = []
        
        # Adaptation parameters
        self.adaptation_steps = 0
        self.preference_history = []
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select an action based on the current state.
        
        Args:
            state: Current state array
            deterministic: Whether to select a deterministic action
            
        Returns:
            Selected action array
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action = self.network.get_action(state_tensor, deterministic)
        
        return action.numpy().squeeze()
    
    def update_preference(self, comfort_feedback: float, efficiency_feedback: float) -> None:
        """
        Update user preference based on explicit or implicit feedback.
        
        Args:
            comfort_feedback: Feedback score for comfort (0-1)
            efficiency_feedback: Feedback score for efficiency (0-1)
        """
        with torch.no_grad():
            current_pref = torch.sigmoid(self.network.user_preference)
            pref_delta = self.config.adaptation_rate * (comfort_feedback - efficiency_feedback)
            new_pref = torch.clamp(current_pref + pref_delta, 0.1, 0.9)
            
            # Update via log-odds for bounded preference
            self.network.user_preference.data = torch.log(new_pref / (1 - new_pref))
            self.preference_history.append(new_pref.item())
    
    def update_policy(self) -> Dict[str, float]:
        """
        Update the policy network based on collected experience.
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.memory) < self.config.batch_size:
            return {"comfort_loss": 0, "efficiency_loss": 0, "constraint_loss": 0}
        
        # Sample batch
        indices = np.random.choice(len(self.memory), self.config.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        # Unpack batch
        states = torch.FloatTensor([item["state"] for item in batch])
        actions = torch.FloatTensor([item["action"] for item in batch])
        comfort_rewards = torch.FloatTensor([item["comfort_reward"] for item in batch])
        efficiency_rewards = torch.FloatTensor([item["efficiency_reward"] for item in batch])
        next_states = torch.FloatTensor([item["next_state"] for item in batch])
        dones = torch.FloatTensor([item["done"] for item in batch])
        
        # Forward pass
        outputs = self.network(states)
        next_outputs = self.network(next_states)
        
        # Compute target values
        comfort_target = comfort_rewards + (1 - dones) * 0.99 * next_outputs["comfort_value"].squeeze()
        efficiency_target = efficiency_rewards + (1 - dones) * 0.99 * next_outputs["efficiency_value"].squeeze()
        
        # Compute value losses
        comfort_value_loss = F.mse_loss(outputs["comfort_value"].squeeze(), comfort_target.detach())
        efficiency_value_loss = F.mse_loss(outputs["efficiency_value"].squeeze(), efficiency_target.detach())
        
        # Compute action losses
        std = torch.exp(self.network.log_std)
        comfort_dist = torch.distributions.Normal(outputs["comfort_action_mean"], std)
        efficiency_dist = torch.distributions.Normal(outputs["efficiency_action_mean"], std)
        combined_dist = torch.distributions.Normal(outputs["combined_action_mean"], std)
        
        comfort_log_probs = comfort_dist.log_prob(actions).sum(dim=-1)
        efficiency_log_probs = efficiency_dist.log_prob(actions).sum(dim=-1)
        combined_log_probs = combined_dist.log_prob(actions).sum(dim=-1)
        
        # Compute advantages
        comfort_advantage = comfort_rewards - outputs["comfort_value"].squeeze().detach()
        efficiency_advantage = efficiency_rewards - outputs["efficiency_value"].squeeze().detach()
        
        # Compute policy losses
        comfort_policy_loss = -(comfort_log_probs * comfort_advantage).mean()
        efficiency_policy_loss = -(efficiency_log_probs * efficiency_advantage).mean()
        combined_policy_loss = -(combined_log_probs * (
            self.config.comfort_coefficient * comfort_advantage + 
            self.config.efficiency_coefficient * efficiency_advantage
        )).mean()
        
        # Compute entropy bonus
        entropy = combined_dist.entropy().mean()
        
        # Compute constraint loss
        constraint_loss = torch.zeros(1)
        for i, constraint in enumerate(self.constraints):
            constraint_values = torch.FloatTensor([
                item["constraint_values"][i] if "constraint_values" in item else 0.0
                for item in batch
            ])
            min_violation = F.relu(constraint.min_value - constraint_values)
            max_violation = F.relu(constraint_values - constraint.max_value)
            constraint_loss += constraint.weight * (min_violation + max_violation).mean()
        
        # Compute total loss
        value_loss = comfort_value_loss + efficiency_value_loss
        policy_loss = combined_policy_loss
        total_loss = value_loss + policy_loss - self.config.entropy_coefficient * entropy + constraint_loss
        
        # Update network
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Store metrics
        self.comfort_losses.append(comfort_policy_loss.item())
        self.efficiency_losses.append(efficiency_policy_loss.item())
        self.constraint_violations.append(constraint_loss.item())
        self.user_preferences.append(outputs["user_preference"].mean().item())
        
        return {
            "comfort_loss": comfort_policy_loss.item(),
            "efficiency_loss": efficiency_policy_loss.item(),
            "constraint_loss": constraint_loss.item(),
            "entropy": entropy.item(),
            "user_preference": outputs["user_preference"].mean().item()
        }
    
    def add_experience(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        comfort_reward: float,
        efficiency_reward: float,
        next_state: np.ndarray,
        done: bool,
        constraint_values: Optional[List[float]] = None
    ) -> None:
        """
        Add experience to the replay buffer.
        
        Args:
            state: Current state
            action: Selected action
            comfort_reward: Reward for comfort
            efficiency_reward: Reward for efficiency
            next_state: Next state
            done: Whether the episode is done
            constraint_values: Values of biomechanical constraints
        """
        experience = {
            "state": state,
            "action": action,
            "comfort_reward": comfort_reward,
            "efficiency_reward": efficiency_reward,
            "next_state": next_state,
            "done": done
        }
        
        if constraint_values is not None:
            experience["constraint_values"] = constraint_values
        
        if len(self.memory) >= self.config.memory_capacity:
            self.memory.pop(0)
        
        self.memory.append(experience)
        
        self.adaptation_steps += 1
        if self.adaptation_steps % self.config.update_interval == 0:
            self.update_policy()
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'preference_history': self.preference_history,
            'comfort_losses': self.comfort_losses,
            'efficiency_losses': self.efficiency_losses,
            'constraint_violations': self.constraint_violations,
            'user_preferences': self.user_preferences
        }, path)
    
    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.preference_history = checkpoint['preference_history']
        self.comfort_losses = checkpoint['comfort_losses']
        self.efficiency_losses = checkpoint['efficiency_losses']
        self.constraint_violations = checkpoint['constraint_violations']
        self.user_preferences = checkpoint['user_preferences'] 