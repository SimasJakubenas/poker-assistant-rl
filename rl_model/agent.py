import numpy as np
import random
import os
from collections import deque

# For reinforcement learning
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    has_torch = True
except ImportError:
    has_torch = False
    print("PyTorch not found. RL training will not be available.")

# Example RL agent if PyTorch is available
if has_torch:
    class DQNAgent:
        """Deep Q-Network agent for Poker."""
        
        def __init__(self, state_size: int, action_size: int, player_id: int):
            self.state_size = state_size
            self.action_size = action_size
            self.player_id = player_id
            self.memory = deque(maxlen=10000)
            self.gamma = 0.95  # Discount factor
            self.epsilon = 1.0  # Exploration rate
            self.epsilon_min = 0.1
            self.epsilon_decay = 0.995
            self.learning_rate = 0.001
            self.model = self._build_model()
            self.target_model = self._build_model()
            self.update_target_model()
            
        def _build_model(self):
            """Build a neural network model."""
            model = nn.Sequential(
                nn.Linear(self.state_size, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, self.action_size)
            )
            self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            return model
            
        def update_target_model(self):
            """Update target model to match the main model."""
            self.target_model.load_state_dict(self.model.state_dict())
            
        def act(self, observation, legal_actions=None):
            """Choose an action."""
            if self.player_id not in observation:
                return None
                
            player_obs = observation[self.player_id]
            
            if "legal_actions" not in player_obs:
                # Not this player's turn
                return None
                
            # Extract state representation
            state = self._process_state(player_obs)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Choose action
            if random.random() <= self.epsilon:
                # Exploration: random action
                return random.choice(list(range(self.action_size)))
            else:
                # Exploitation: use the model
                self.model.eval()
                with torch.no_grad():
                    q_values = self.model(state_tensor)
                    
                    # Filter illegal actions
                    legal_indices = list(range(min(len(player_obs["legal_actions"]), self.action_size)))
                    
                    if not legal_indices:
                        # No legal actions available
                        return None
                    legal_q_values = q_values[0][legal_indices]
                    
                    return legal_indices[torch.argmax(legal_q_values).item()]
                    
        def _process_state(self, observation):
            """
            Process the observation into a state vector.
            This is a simple example - in practice, you'd want to create a more
            sophisticated state representation.
            """
            state = []
            
            # Player info
            player_info = observation["players"][self.player_id]
            state.append(player_info["stack"] / 100.0)  # Normalize
            state.append(1.0 if player_info["is_active"] else 0.0)
            state.append(1.0 if player_info["has_folded"] else 0.0)
            state.append(1.0 if player_info["is_all_in"] else 0.0)
            state.append(player_info["current_bet"] / 100.0)  # Normalize
            
            # Community cards
            community_cards = observation["community_cards"]
            num_community = len(community_cards)
            state.append(num_community / 5.0)  # Normalize
            
            # Pot and current bet
            state.append(observation["pot"] / 200.0)  # Normalize
            state.append(observation["current_bet"] / 100.0)  # Normalize
            
            # Betting round
            round_map = {"PREFLOP": 0, "FLOP": 1, "TURN": 2, "RIVER": 3, None: 4}
            betting_round = observation["betting_round"]
            round_idx = round_map.get(betting_round, 0)
            for i in range(5):
                state.append(1.0 if round_idx == i else 0.0)
                
            # Add additional features as needed
            
            return state
            
        def remember(self, state, action, reward, next_state, done):
            """Store experience in memory."""
            self.memory.append((state, action, reward, next_state, done))
            
        def replay(self, batch_size=32):
            """Train the model with experiences from memory."""
            if len(self.memory) < batch_size:
                return
                
            # Sample batch from memory
            batch = random.sample(self.memory, batch_size)
            
            states = []
            targets = []
            
            self.model.train()
            for state, action, reward, next_state, done in batch:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                
                # Calculate target Q-value
                with torch.no_grad():
                    target = self.model(state_tensor).data.clone()
                    if done:
                        target[0][action] = reward
                    else:
                        next_q = self.target_model(next_state_tensor).data
                        target[0][action] = reward + self.gamma * torch.max(next_q)
                
                states.append(state)
                targets.append(target.squeeze().numpy())
            
            # Convert to tensors and train
            states_tensor = torch.FloatTensor(np.array(states))
            targets_tensor = torch.FloatTensor(np.array(targets))
            
            # Compute loss and update weights
            self.optimizer.zero_grad()
            outputs = self.model(states_tensor)
            loss = F.mse_loss(outputs, targets_tensor)
            loss.backward()
            self.optimizer.step()
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        
        def save(self, filepath):
            """Save model to file."""
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'target_model_state_dict': self.target_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon
            }, filepath)
            
        def load(self, filepath):
            """Load model from file."""
            if os.path.exists(filepath):
                checkpoint = torch.load(filepath)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint['epsilon']