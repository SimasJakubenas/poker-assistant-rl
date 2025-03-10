from typing import Dict, List, Tuple, Optional
from game.essentials import  PokerAction
from game.table import PokerTable


class PokerEnv:
    """
    RL environment for No-Limit Texas Hold'em Poker.
    Follows the standard Gym environment interface.
    """
    
    def __init__(self, n_players: int = 6, small_blind: float = 0.5, big_blind: float = 1.0, 
                 initial_stack: float = 100.0):
        """
        Initialize the poker environment.
        
        Args:
            n_players: Number of players (2-6)
            small_blind: Small blind amount
            big_blind: Big blind amount
            initial_stack: Initial stack size for all players
        """
        self.table = PokerTable(n_players=n_players, 
                              small_blind=small_blind, 
                              big_blind=big_blind,
                              initial_stack=initial_stack)
        self.n_players = n_players
        self.current_player = None
        self.terminal = False
        self.rewards = {i: 0.0 for i in range(n_players)}
        
        # Define action space
        self.action_space_mapping = {
            0: (PokerAction.FOLD, None),
            1: (PokerAction.CHECK, None),
            2: (PokerAction.CALL, None),
            # Following actions are for bet sizing (index 3 and above)
            # These will be dynamically determined based on legal actions
        }
    
    def reset(self):
        """
        Reset the environment and return initial state.
        
        Returns:
            Initial state dictionary
        """
        self.table.start_new_hand()
        self.current_player = self.table.current_player_idx
        self.terminal = False
        self.rewards = {i: 0.0 for i in range(self.n_players)}
        
        # Return the initial state
        return self._get_observation()
    
    def step(self, action_idx: int):
        """
        Take a step in the environment by performing an action.
        
        Args:
            action_idx: Index of the action to perform
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        self.render()
        if self.terminal:
            return self._get_observation(), 0.0, True, {"info": "Hand already complete"}
            
        # Get action and amount from action index
        legal_actions = self.table.get_legal_actions(self.current_player)
        action_mapping = self._create_action_mapping(legal_actions)
        
        if action_idx >= len(action_mapping):
            # Invalid action, return current state with a penalty
            return self._get_observation(), -1.0, False, {"info": "Invalid action"}
            
        action, amount = action_mapping[action_idx]
        
        # Execute the action
        next_state = self.table.act(self.current_player, action, amount)
        
        # Store previous player for reward calculation
        prev_player = self.current_player
        
        # Update current player
        self.current_player = self.table.current_player_idx
        
        # Calculate immediate rewards (based on money put in the pot)
        immediate_reward = 0.0
        if action in [PokerAction.CALL, PokerAction.BET, PokerAction.RAISE, PokerAction.ALL_IN]:
            # Negative reward for putting money in the pot
            immediate_reward = -amount
            
        # Check if hand is complete
        if self.current_player is None or self.table.hand_complete:
            self.terminal = True
        else:
            # Immediate reward for the action taken
            self.rewards[prev_player] = immediate_reward
            
        return self._get_observation(), self.rewards[prev_player], self.terminal, {}
    
    def calculate_final_rewards(self):
        # Calculate final rewards (based on stack changes)
        for i in range(self.n_players):
            # Final reward is the stack change
            self.rewards[i] = self.table.players[i].stack - self.table.initial_stack
    
    def _get_observation(self):
        """
        Get the current state observation.
        
        Returns:
            State observation dictionary
        """
        if self.current_player is None:
            # Game is over or not started
            return {i: self.table.get_state() for i in range(self.n_players)}
            
        # Return player-specific observations
        return {i: self.table.get_player_state(i) for i in range(self.n_players)}
    
    def _create_action_mapping(self, legal_actions: Dict[PokerAction, List[float]]) -> Dict[int, Tuple[PokerAction, Optional[float]]]:
        """
        Creates a mapping from action indices to (PokerAction, amount) pairs.
        
        Args:
            legal_actions: Dictionary of legal actions from the poker table
            
        Returns:
            Dictionary mapping action indices to (action, amount) pairs
        """
        action_mapping = {}
        action_idx = 0
        
        # Basic actions (FOLD, CHECK, CALL)
        if PokerAction.FOLD in legal_actions:
            action_mapping[action_idx] = (PokerAction.FOLD, None)
            action_idx += 1
            
        if PokerAction.CHECK in legal_actions:
            action_mapping[action_idx] = (PokerAction.CHECK, None)
            action_idx += 1
            
        if PokerAction.CALL in legal_actions:
            call_amount = legal_actions[PokerAction.CALL][0]
            action_mapping[action_idx] = (PokerAction.CALL, call_amount)
            action_idx += 1
            
        # BET actions (including raises and all-ins)
        for action in [PokerAction.BET, PokerAction.RAISE]:
            if action in legal_actions:
                for amount in legal_actions[action]:
                    action_mapping[action_idx] = (action, amount)
                    action_idx += 1
                    
        # ALL_IN (add as a separate action if not included in bet/raise)
        if PokerAction.ALL_IN in legal_actions:
            all_in_amount = legal_actions[PokerAction.ALL_IN][0]
            if not any(a == PokerAction.ALL_IN and amt == all_in_amount for a, amt in action_mapping.values()):
                action_mapping[action_idx] = (PokerAction.ALL_IN, all_in_amount)
                action_idx += 1
                
        return action_mapping
    
    def render(self, mode='human'):
        """
        Render the current state of the environment.
        
        Args:
            mode: Rendering mode ('human' for console output)
            
        Returns:
            None
        """
        if mode != 'human':
            return
            
        state = self.table.get_state()
        
        print("=" * 80)
        print(f"Poker Hand - Betting Round: {state['betting_round']}")
        print(f"Pot: {state['pot']:.2f}  Current Bet: {state['current_bet']:.2f}")
        print(f"Button Position: {state['button_pos']}  Current Player: {state['current_player']}")
        print(f"Community Cards: {' '.join(state['community_cards'])}")
        print("-" * 80)
        
        # Print player information
        for player_id, player_info in state['players'].items():
            position_name = ["BTN", "SB", "BB", "UTG", "MP", "CO"][player_info['position']] if player_info['position'] < 6 else str(player_info['position'])
            status = ""
            if player_info['has_folded']:
                status = "FOLDED"
            elif player_info['is_all_in']:
                status = "ALL-IN"
            elif int(player_id) == state['current_player']:
                status = "TO ACT"
                
            hole_cards = ' '.join(player_info['hole_cards']) if player_info['hole_cards'] else "XX"
            
            print(f"Player {player_id} ({position_name}): Stack=${player_info['stack']:.2f}  Bet=${player_info['current_bet']:.2f}  Cards: {hole_cards}  {status}")
            
        print("=" * 80)
        
        # If hand is complete, show winners
        if state['hand_complete'] and state['hand_history']:
            winners_action = next((a for a in reversed(state['hand_history']) if a['type'] in ['WINNER', 'SIDE_POT_WINNER']), None)
            if winners_action:
                print(f"Winners: {winners_action['winners']}")
                print(f"Amount per winner: ${winners_action['amount_per_winner']:.2f}")
                
        print("=" * 80)