import random


class RandomAgent:
    """Simple random agent for testing the poker environment."""
    
    def __init__(self, player_id: int):
        self.player_id = player_id
        
    def act(self, observation, legal_actions=None):
        """
        Select a random legal action.
        
        Args:
            observation: Current state observation
            legal_actions: Optional dictionary of legal actions
            
        Returns:
            Integer action index
        """
        player_obs = observation[self.player_id]
        
        if "legal_actions" not in player_obs:
            # Not this player's turn
            return None
            
        # Get legal actions
        legal_actions = player_obs["legal_actions"]
        
        # Create action mapping
        action_mapping = {}
        action_idx = 0
        
        # Map each legal action to an index
        for action_type, amounts in legal_actions.items():
            if not amounts:  # Actions like FOLD, CHECK
                action_mapping[action_idx] = (action_type, None)
                action_idx += 1
            else:
                for amount in amounts:
                    action_mapping[action_idx] = (action_type, amount)
                    action_idx += 1
                    
        if not action_mapping:
            return None
            
        # Choose random action index
        action_idx = random.choice(list(action_mapping.keys()))
        return action_idx