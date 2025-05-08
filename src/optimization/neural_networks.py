"""
Neural network implementations for exoskeleton optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

class PPONetwork(nn.Module):
    """
    Implementation of PPO policy network for exoskeleton control.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        """
        Initialize the PPO network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        # Actor network
        actor_layers = []
        prev_dim = state_dim
        for dim in hidden_dims:
            actor_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU()
            ])
            prev_dim = dim
        actor_layers.append(nn.Linear(prev_dim, action_dim))
        self.actor = nn.Sequential(*actor_layers)
        
        # Critic network
        critic_layers = []
        prev_dim = state_dim
        for dim in hidden_dims:
            critic_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU()
            ])
            prev_dim = dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)
        
        # Action distribution parameters
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: Current state tensor
            
        Returns:
            Tuple of (action_mean, state_value)
        """
        action_mean = self.actor(state)
        state_value = self.critic(state)
        return action_mean, state_value
    
    def get_action(self, state: torch.Tensor, 
                   deterministic: bool = False) -> torch.Tensor:
        """
        Sample an action from the policy.
        
        Args:
            state: Current state tensor
            deterministic: Whether to return deterministic action
            
        Returns:
            Sampled action tensor
        """
        action_mean, _ = self.forward(state)
        
        if deterministic:
            return action_mean
        
        std = torch.exp(self.log_std)
        normal = torch.distributions.Normal(action_mean, std)
        action = normal.sample()
        return action
    
    def evaluate_actions(self, state: torch.Tensor, 
                        action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Args:
            state: Current state tensor
            action: Action tensor to evaluate
            
        Returns:
            Tuple of (log_prob, entropy, state_value)
        """
        action_mean, state_value = self.forward(state)
        std = torch.exp(self.log_std)
        
        normal = torch.distributions.Normal(action_mean, std)
        log_prob = normal.log_prob(action).sum(dim=-1)
        entropy = normal.entropy().sum(dim=-1)
        
        return log_prob, entropy, state_value

class MetabolicCostNet(nn.Module):
    """
    Neural network for predicting metabolic cost.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        """
        Initialize the metabolic cost prediction network.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
            
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor containing state, action, and parameter information
            
        Returns:
            Predicted metabolic cost
        """
        return self.network(x)
    
class MovementClassifier(nn.Module):
    """
    Neural network for classifying movement patterns.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int):
        """
        Initialize the movement pattern classifier.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of movement classes
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(0.3)
            ])
            prev_dim = dim
            
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor containing movement features
            
        Returns:
            Class logits
        """
        return self.network(x)
    
class StabilityPredictor(nn.Module):
    """
    Neural network for predicting stability metrics.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 num_metrics: int = 3):
        """
        Initialize the stability prediction network.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            num_metrics: Number of stability metrics to predict
        """
        super().__init__()
        
        # Temporal feature extraction
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dims[0],
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Stability prediction
        fc_layers = []
        prev_dim = hidden_dims[0]
        for dim in hidden_dims[1:]:
            fc_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
            
        fc_layers.append(nn.Linear(prev_dim, num_metrics))
        self.fc = nn.Sequential(*fc_layers)
        
    def forward(self, x: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor containing temporal stability features
            hidden: Optional initial hidden state for LSTM
            
        Returns:
            Tuple of (stability_metrics, hidden_state)
        """
        # Process temporal features
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Predict stability metrics
        stability = self.fc(lstm_out[:, -1, :])  # Use last temporal state
        return stability, hidden 