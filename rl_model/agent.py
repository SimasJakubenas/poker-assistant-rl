import numpy as np
import random
import os
from collections import deque, namedtuple

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

# Define a Hand object to store complete hand sequences
Hand = namedtuple('Hand', ['states', 'actions', 'rewards', 'next_states', 'dones', 'player_id'])

# Example RL agent if PyTorch is available
if has_torch:
    class DQNAgent:
        """Deep Q-Network agent for Poker."""
        
        def __init__(self, state_size: int, action_size: int, player_id: int):
            self.state_size = state_size
            self.action_size = action_size
            self.player_id = player_id
            self.hand_memory = deque(maxlen=1000)
            self.gamma = 0.95  # Discount factor
            self.epsilon = 1.0  # Exploration rate
            self.epsilon_min = 0.1
            self.epsilon_decay = 0.995
            self.learning_rate = 0.001
            self.model = self._build_model()
            self.target_model = self._build_model()
            self.update_target_model()
            
            # Current hand being played
            self.current_hand = None
            self.reset_current_hand()
            self.total = 0.0
            
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
        
        def reset_current_hand(self):
            """Reset the current hand being collected."""
            self.current_hand = {
                'states': [],
                'actions': [],
                'rewards': [],
                'next_states': [],
                'dones': []
            }
        
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
            round_map = {"PREFLOP": 0, "FLOP": 1, "TURN": 2, "RIVER": 3}
            betting_round = observation["betting_round"]
            round_idx = round_map.get(betting_round, 0)
            for i in range(4):
                state.append(1.0 if round_idx == i else 0.0)
            
            # Player position
            position_map = {'BTN': 0, 'SB': 1, 'BB': 2, 'UTG': 3, 'MP': 4, 'CO': 5}
            position = position_map.get(player_info.get('position'), 0) / 5.0  # Normalize
            state.append(position)

            # Pot size relative to player stack
            pot_to_stack = observation['pot'] / max(1.0, player_info.get('stack', 1.0))
            pot_to_stack = min(3.0, pot_to_stack) / 3.0  # Cap at 3x stack and normalize
            state.append(pot_to_stack)

            # Current bet relative to pot
            if observation['pot'] > 0:
                bet_to_pot = observation['current_bet'] / observation['pot']
                bet_to_pot = min(2.0, bet_to_pot) / 2.0  # Cap at 2x pot and normalize
            else:
                bet_to_pot = 0.0
            state.append(bet_to_pot)

            # Number of active players
            active_players = sum(1 for p in observation['players'].values() 
                            if p.get('is_all_in', False) and not p.get('has_folded', False))
            active_ratio = active_players / 6.0  # Normalize by max players
            state.append(active_ratio)
            # Add additional features as needed
            
            return state
            
        def remember_action(self, state, action, reward, next_state, done):
            """Store an action from the current hand."""
            self.current_hand['states'].append(state)
            self.current_hand['actions'].append(action)
            self.current_hand['rewards'].append(reward)
            self.current_hand['next_states'].append(next_state)
            self.current_hand['dones'].append(done)
        
        def complete_hand(self, final_reward):
            """Complete the current hand and store it in memory."""
            if not self.current_hand['states']:
                # No actions were taken by this agent in this hand
                self.reset_current_hand()
                return
            
            # Replace the last reward with the final hand outcome
            if self.current_hand['rewards']:
                self.current_hand['rewards'][-1] = final_reward
            
            # Create a Hand object and add to memory
            hand = Hand(
                states=self.current_hand['states'],
                actions=self.current_hand['actions'],
                rewards=self.current_hand['rewards'],
                next_states=self.current_hand['next_states'],
                dones=self.current_hand['dones'],
                player_id=self.player_id
            )
            self.hand_memory.append(hand)
            
            # Reset for the next hand
            self.reset_current_hand()
        
        def replay(self, batch_size=4):
            """Train the model with hand-level experience replay."""
            if len(self.hand_memory) < batch_size:
                return
            
            # Sample a batch of hands
            hands = random.sample(self.hand_memory, batch_size)
            
            # Process each hand
            for hand in hands:
                # Calculate returns with Monte Carlo (using the actual final outcome)
                self._train_on_hand(hand)
         
        def _train_on_hand(self, hand):
            """Train on a single complete hand."""
            self.model.train()
            
            # Get the final reward of the hand (typically the stack change)
            final_reward = hand.rewards[-1]
            
            # Process all state-action pairs in the hand
            for i in range(len(hand.states)):
                state = hand.states[i]
                action = hand.actions[i]
                
                # Convert state to tensor
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                # Get current Q-values
                q_values = self.model(state_tensor)
                
                # Create target by copying predictions
                target = q_values.clone().detach()
                
                # Update Q-value for the taken action
                target[0, action] = final_reward
                
                # Compute loss and update weights
                self.optimizer.zero_grad()
                loss = nn.MSELoss()(q_values, target)
                loss.backward()
                self.optimizer.step()
            
            # Decay epsilon after training on a hand
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        
        def save(self, filepath, player_total):
            """Save model to file."""
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'target_model_state_dict': self.target_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'total': player_total
            }, filepath)
            
        def load(self, filepath):
            """Load model from file."""
            if os.path.exists(filepath):
                checkpoint = torch.load(filepath)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint['epsilon']
                self.total = checkpoint['total']