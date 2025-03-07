from typing import Dict, List, Optional
from game.essentials import Deck, PokerAction, BettingRound
from game.player import PokerPlayer
from game.hand_evaluator import PokerHandEvaluator


class PokerTable:
    """Represents a poker table with players and game state."""
    
    def __init__(self, n_players: int = 6, small_blind: float = 0.5, big_blind: float = 1.0, 
                 initial_stack: float = 100.0, max_players: int = 6):
        if n_players < 2 or n_players > max_players:
            raise ValueError(f"Number of players must be between 2 and {max_players}")
            
        self.max_players = max_players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.initial_stack = initial_stack
        
        # Create players
        self.players = {i: PokerPlayer(i, initial_stack) for i in range(n_players)}
        
        # Game state
        self.deck = Deck()
        self.community_cards = []
        self.pot = 0.0
        self.side_pots = []  # List of (amount, eligible_players) tuples
        self.current_bet = 0.0
        self.current_min_raise = big_blind
        self.button_pos = 0
        self.current_player_idx = None
        self.betting_round = None
        self.hand_complete = True
        self.last_aggressor = None
        self.hand_history = []
        
        # Initialize player positions
        self._set_player_positions()
    
    def _set_player_positions(self):
        """Sets player positions based on the button position."""
        active_players = [p_id for p_id, p in self.players.items() if p.is_active]
        if len(active_players) < 2:
            raise ValueError("Need at least 2 active players to set positions")
            
        # Sort active players by their IDs for positional assignments
        active_players.sort()
        
        # Find button position in active players
        button_idx = active_players.index(self.button_pos) if self.button_pos in active_players else 0
        
        # Assign positions (Button is 0, SB is 1, BB is 2, etc.)
        positions = {}
        for i in range(len(active_players)):
            player_idx = (button_idx + i) % len(active_players)
            player_id = active_players[player_idx]
            positions[player_id] = i
            
        # Update player positions
        for player_id, position in positions.items():
            self.players[player_id].position = position
            
    def get_state(self) -> Dict:
        """
        Returns the current state of the poker table.
        
        Returns:
            Dictionary containing the current state of the game.
        """
        state = {
            "hand_complete": self.hand_complete,
            "betting_round": self.betting_round.name if self.betting_round else None,
            "pot": round(self.pot, 2),
            "current_bet": round(self.current_bet, 2),
            "button_pos": self.button_pos,
            "current_player": self.current_player_idx,
            "community_cards": [str(card) for card in self.community_cards],
            "players": {},
            "hand_history": self.hand_history
        }
        
        for player_id, player in self.players.items():
            state["players"][player_id] = {
                "stack": player.stack,
                "balance": player.balance,
                "hole_cards": [str(card) for card in player.hole_cards],
                "position": player.position,
                "is_active": player.is_active,
                "has_folded": player.has_folded,
                "is_all_in": player.is_all_in,
                "current_bet": round(player.current_bet, 2),
                "total_bet_in_hand": round(player.total_bet_in_hand, 2),
                "last_action": str(player.last_action) if player.last_action else None
            }
            
        return state
    
    def get_player_state(self, player_id: int) -> Dict:
        """
        Returns the game state from a specific player's perspective.
        
        Args:
            player_id: The player ID to get the state for
            
        Returns:
            Dictionary with game state information visible to the player
        """
        state = self.get_state()
        
        # Hide hole cards of other players
        for pid, player_info in state["players"].items():
            if int(pid) != player_id:
                player_info["hole_cards"] = []
                
        # Add legal actions for the current player
        if self.current_player_idx == player_id:
            state["legal_actions"] = self.get_legal_actions(player_id)
            
        return state
        
    def start_new_hand(self) -> Dict:
        """
        Starts a new hand of poker, dealing cards and posting blinds.
        
        Returns:
            Dictionary with new hand state
        """
        # if self.hand_complete is False:
        #     raise ValueError("Previous hand is not complete")
            
        # Reset game state
        self.deck.reset()
        self.community_cards = []
        self.pot = 0.0
        self.side_pots = []
        self.current_bet = 0.0
        self.current_min_raise = self.big_blind
        self.betting_round = BettingRound.PREFLOP
        self.hand_complete = False
        self.hand_history = []
        
        # Reset all players for new hand
        for player in self.players.values():
            player.reset_for_new_hand()
            
        # Move the button
        self._move_button()
        
        # Set player positions
        self._set_player_positions()
        
        # Deal hole cards
        active_players = [p_id for p_id, p in self.players.items() if p.is_active]
        for player_id in active_players:
            self.players[player_id].hole_cards = self.deck.deal(2)
            
        # Post blinds
        self._post_blinds()
        
        # Set starting player for preflop (after BB)
        positions = {p.player_id: p.position for p in self.players.values() if p.is_active}
        next_pos = 3 % len(active_players)  # After BB
        self.current_player_idx = [p_id for p_id, pos in positions.items() if pos == next_pos][0]
        
        # Record hand start
        self._record_action("HAND_START", -1, 0, 
                            {"button": self.button_pos, "blinds": (self.small_blind, self.big_blind)})
        
        # Return initial state
        return self.get_state()
    
    def _move_button(self):
        """Moves the button to the next active player."""
        active_players = [p_id for p_id, p in self.players.items() if p.is_active]
        if not active_players:
            return
            
        # Find next active player after button
        next_pos = (self.button_pos + 1) % max(self.players.keys(), default=0) + 1
        while next_pos not in active_players and next_pos != self.button_pos:
            next_pos = (next_pos + 1) % max(self.players.keys(), default=0) + 1
            
        self.button_pos = next_pos
    
    def _post_blinds(self):
        """Posts small and big blinds."""
        active_players = sorted([p_id for p_id, p in self.players.items() if p.is_active])
        if len(active_players) < 2:
            return
            
        # Find SB and BB positions
        positions = {p.player_id: p.position for p in self.players.values() if p.is_active}
        sb_player_id = [p_id for p_id, pos in positions.items() if pos == 1 % len(active_players)][0]
        bb_player_id = [p_id for p_id, pos in positions.items() if pos == 2 % len(active_players)][0]
        
        # Post small blind
        sb_amount = self.players[sb_player_id].bet(self.small_blind)
        self.pot += sb_amount
        self.players[sb_player_id].last_action = (PokerAction.BET, sb_amount)
        self._record_action("SMALL_BLIND", sb_player_id, sb_amount)
        
        # Post big blind
        bb_amount = self.players[bb_player_id].bet(self.big_blind)
        self.pot += bb_amount
        self.current_bet = bb_amount
        self.players[bb_player_id].last_action = (PokerAction.BET, bb_amount)
        self._record_action("BIG_BLIND", bb_player_id, bb_amount)
    
    def _record_action(self, action_type: str, player_id: int, amount: float, extra_info: Dict = None):
        """Records an action in the hand history."""
        action = {
            "type": action_type,
            "player": player_id,
            "amount": amount,
            "betting_round": self.betting_round.name if self.betting_round else None,
            "pot": self.pot,
            "timestamp": len(self.hand_history)
        }
        
        if extra_info:
            action.update(extra_info)
            
        self.hand_history.append(action)
        
        return action
    
    def get_legal_actions(self, player_id: int) -> Dict[PokerAction, List[float]]:
        """
        Gets the legal actions and bet amounts for a player.
        
        Args:
            player_id: The player ID to get legal actions for
            
        Returns:
            Dictionary mapping PokerAction to list of valid bet amounts (empty list for FOLD, CHECK)
        """
        player = self.players[player_id]
        
        if player.has_folded or player.is_all_in or not player.is_active:
            return {}
            
        legal_actions = {}
        
        # FOLD is always legal unless player can check
        if self.current_bet > player.current_bet:
            legal_actions[PokerAction.FOLD] = []
            
        # CHECK is legal if no current bet or player has matched it
        if self.current_bet <= player.current_bet:
            legal_actions[PokerAction.CHECK] = []
        
        # CALL is legal if there's a bet to call
        call_amount = round(self.current_bet - player.current_bet, 2)
        if call_amount > 0:
            # If call amount is >= player's stack, it becomes an all-in call
            if call_amount >= player.stack:
                legal_actions[PokerAction.ALL_IN] = [player.stack]
            else:
                legal_actions[PokerAction.CALL] = [call_amount]
                
                # ALL_IN is also available
                legal_actions[PokerAction.ALL_IN] = [player.stack]
        
        # BET/RAISE are legal if player has enough chips
        min_bet_or_raise = max(self.current_min_raise, self.big_blind)
        
        if self.current_bet == 0:  # No current bet, so it's a bet
            if player.stack >= min_bet_or_raise:
                legal_actions[PokerAction.BET] = self._get_bet_sizing_options(
                    min_bet_or_raise, player.stack
                )
        else:  # Current bet exists, so it's a raise
            min_raise_to = round(self.current_bet + min_bet_or_raise, 2)
            raise_amount = round(min_raise_to - player.current_bet, 2)
            
            if player.stack >= raise_amount:
                legal_actions[PokerAction.RAISE] = self._get_bet_sizing_options(
                    min_raise_to, round(player.stack + player.current_bet, 2)
                )
                
        return legal_actions
    
    def _get_bet_sizing_options(self, min_amount: float, max_amount: float) -> List[float]:
        """
        Gets common bet sizing options between min and max amount.
        
        Args:
            min_amount: Minimum bet amount
            max_amount: Maximum bet amount (player's stack)
            
        Returns:
            List of bet sizing options
        """
        # For RL environment, return a discretized set of bet sizes
        pot_size = self.pot
        options = []
        
        # Min bet
        options.append(min_amount)
        
        # 1/2 pot
        half_pot = max(min_amount, round(pot_size * 0.5, 2))
        if min_amount <= half_pot <= max_amount:
            options.append(half_pot)
            
        # 2/3 pot
        two_thirds_pot = max(min_amount, round(pot_size * 0.67, 2))
        if min_amount <= two_thirds_pot <= max_amount and abs(two_thirds_pot - half_pot) > 0.5:
            options.append(two_thirds_pot)
            
        # Pot-sized bet
        pot_bet = max(min_amount, pot_size)
        if min_amount <= pot_bet <= max_amount and abs(pot_bet - two_thirds_pot) > 0.5:
            options.append(pot_bet)
            
        # 2x pot
        double_pot = max(min_amount, pot_size * 2)
        if min_amount <= double_pot <= max_amount and abs(double_pot - pot_bet) > 1.0:
            options.append(double_pot)
            
        # All-in
        if max_amount not in options:
            options.append(max_amount)
            
        return sorted(options)
    
    def act(self, player_id: int, action: PokerAction, amount: Optional[float] = None) -> Dict:
        """
        Execute a player action and return the updated game state.
        
        Args:
            player_id: The player taking the action
            action: The PokerAction to take
            amount: The bet amount (required for BET, RAISE, ALL_IN)
            
        Returns:
            Updated game state dictionary
        """
        if self.hand_complete:
            raise ValueError("Hand is already complete")
            
        if player_id != self.current_player_idx:
            raise ValueError(f"Not {player_id}'s turn to act")
            
        player = self.players[player_id]
        legal_actions = self.get_legal_actions(player_id)
        
        if action not in legal_actions:
            raise ValueError(f"Action {action} is not legal for player {player_id}")
            
        if action in [PokerAction.BET, PokerAction.RAISE, PokerAction.ALL_IN] and amount is None:
            raise ValueError(f"Amount required for action {action}")
            
        if action in [PokerAction.BET, PokerAction.RAISE, PokerAction.ALL_IN]:
            valid_amounts = legal_actions[action]
            if amount not in valid_amounts and action != PokerAction.ALL_IN:
                # Allow for small floating point differences
                valid_amount = next((a for a in valid_amounts if abs(a - amount) < 0.01), None)
                if valid_amount is None:
                    raise ValueError(f"Amount {amount} not valid for action {action}. Valid amounts: {valid_amounts}")
                amount = valid_amount
                
        # Execute the action
        if action == PokerAction.FOLD:
            player.fold()
            self._record_action("FOLD", player_id, 0)
            
        elif action == PokerAction.CHECK:
            self._record_action("CHECK", player_id, 0)
            
        elif action == PokerAction.CALL:
            call_amount = round(self.current_bet - player.current_bet, 2)
            call_amount = min(call_amount, player.stack)  # In case player can't cover the call
            
            actual_bet = player.bet(call_amount)
            self.pot += actual_bet
            
            self._record_action("CALL", player_id, actual_bet)
            
        elif action == PokerAction.BET:
            if self.current_bet > 0:
                raise ValueError("Cannot BET when there's already a bet; use RAISE")
                
            actual_bet = player.bet(amount)
            self.pot += actual_bet
            self.current_bet = actual_bet
            self.current_min_raise = actual_bet
            self.last_aggressor = player_id
            
            self._record_action("BET", player_id, actual_bet)
            
        elif action == PokerAction.RAISE:
            raise_to = amount
            actual_raise = round(raise_to - player.current_bet, 2)
            
            # Adjust for player's stack
            actual_bet = player.bet(actual_raise)
            self.pot += actual_bet
            
            # Update current bet and minimum raise
            old_bet = self.current_bet
            self.current_bet = player.current_bet
            self.current_min_raise = round(self.current_bet - old_bet, 2)
            self.last_aggressor = player_id
            
            self._record_action("RAISE", player_id, actual_bet, {"raise_to": player.current_bet})
            
        elif action == PokerAction.ALL_IN:
            # If the all-in amount is greater than the current bet, it's treated as a raise
            all_in_amount = player.stack
            is_raise = round(player.current_bet + all_in_amount, 2) > self.current_bet
            
            actual_bet = player.bet(all_in_amount)
            self.pot += actual_bet
            
            if is_raise:
                old_bet = self.current_bet
                self.current_bet = player.current_bet
                self.current_min_raise = max(self.big_blind, self.current_bet - old_bet)
                self.last_aggressor = player_id
                
            self._record_action("ALL_IN_INFO", player_id, all_in_amount,
                            {"stack_size": player.stack, "total_pot": self.pot})
            
        player.last_action = (action, amount if amount is not None else 0)
        
        # Move to next player or next betting round
        self._advance_game()
        
        # Return updated state
        return self.get_state()

    def _end_hand(self, winners: List[int]):
        """Ends the current hand and distributes the pot to the winners."""
        # Calculate the amount each winner receives
        amount_per_winner = round(self.pot / len(winners), 2)
        
        for winner_id in winners:
            self.players[winner_id].win_amount(amount_per_winner)
        
        # Record the winners in the hand history
        self._record_action("WINNER", -1, 0, {"winners": winners, "amount_per_winner": amount_per_winner})
        # Mark the hand as complete
        self.hand_complete = True
        
        for i, player in self.players.items():
            player.update_balance()
    
    def _advance_game(self):
        """Advances the game to the next player or next betting round."""
        # Check if hand is complete
        active_players = [p for p in self.players.values() if not p.has_folded and p.is_active]
        
        # If only one player remains, they win
        if len(active_players) == 1:
            self._end_hand([active_players[0].player_id])
            return
            
        # Find next player to act
        next_player = self._get_next_player()
        
        # If next player is None or betting round is complete, move to next betting round
        if next_player is None or self._is_betting_round_complete():
            self._move_to_next_betting_round()
        else:
            self.current_player_idx = next_player
    
    def _move_to_next_betting_round(self):
        """Moves to the next betting round or ends the hand if all rounds are complete."""
        if self.betting_round == BettingRound.RIVER:
            # Showdown and determine the winners
            self._showdown_with_side_pots()
        else:
            # Move to the next betting round
            self.check_and_deal_next_round()
            self.current_bet = 0.0
            self.current_min_raise = self.big_blind
            self.current_player_idx = self._get_next_player()
            self._record_action("NEW_ROUND", -1, 0, {"betting_round": self.betting_round.name})

    
    def _get_next_player(self) -> Optional[int]:
        """Gets the ID of the next player to act."""
        # Get list of active players
        active_players = [p for p in self.players.values() 
                         if not p.has_folded and p.is_active and not p.is_all_in]
        if not active_players:
            return None
            
        # Get positions of active players
        positions = {p.player_id: p.position for p in active_players}
        # Find current player position
        current_pos = positions.get(self.current_player_idx)
        
        if current_pos is None:
            # Find the first player after the button
            sorted_positions = sorted(positions.items(), key=lambda x: x[0]) # Sort the keys
            for position in sorted_positions:
                if position[0] >= self.current_player_idx:
                    return position[0]  # Return the next highest key
            return sorted_positions[0][0]
            
        # Find the next player by position
        next_players = [(p_id, pos) for p_id, pos in positions.items() 
                       if pos > current_pos]
        
        if next_players:
            # Sort by position and return the first
            next_players.sort(key=lambda x: x[1])
            return next_players[0][0]
        else:
            # Wrap around to the first position
            sorted_positions = sorted(positions.items(), key=lambda x: x[1])
            return sorted_positions[0][0]
    
    def _is_betting_round_complete(self) -> bool:
        """Checks if the current betting round is complete."""
        active_players = [p for p in self.players.values() 
                         if not p.has_folded and p.is_active and not p.is_all_in]
        if not active_players:
            return True
            
        # Betting round is complete if all active players have the same bet amount
        # or have folded, and at least one player has acted
        bet_amounts = [p.current_bet for p in active_players]
        
        if len(set(bet_amounts)) != 1:
            return False
            
        # If there's no last aggressor or they have already acted,
        # and everyone has had a chance to act since the last bet/raise
        if self.last_aggressor is None:
            # In preflop, the big blind acts last
            if self.betting_round == BettingRound.PREFLOP:
                bb_player = next((p for p in self.players.values() if p.position == 2 % len(active_players)), None)
                if bb_player and not bb_player.has_folded and not bb_player.is_all_in:
                    return self.current_player_idx == bb_player.player_id
            
            # For the betting round to be complete, all players must have acted at least once
            if self.hand_history:
                acted_players = {a["player"] for a in self.hand_history 
                                if a["betting_round"] == self.betting_round.name
                                and a["type"] not in ["HAND_START", "SMALL_BLIND", "BIG_BLIND"]}
                return len(acted_players) == len(active_players)
        
        # If there is a last aggressor, check if all players have acted after them
        last_aggressor_timestamp = None
        for action in reversed(self.hand_history):
            if action["player"] == self.last_aggressor and action["betting_round"] == self.betting_round.name:
                last_aggressor_timestamp = action["timestamp"]
                break
                
        if last_aggressor_timestamp is not None:
            # Check if all active players have acted after the last aggressor
            for player in active_players:
                player_acted_after_aggressor = False
                for action in self.hand_history:
                    if action["timestamp"] > last_aggressor_timestamp and action["player"] == player.player_id:
                        player_acted_after_aggressor = True
                        break
                        
                if not player_acted_after_aggressor:
                    return False
                    
        return True
    
    def deal_community_cards(self, num_cards=None, betting_round=None):
        """
        Deal community cards for the specified betting round and add them to the game state.
        
        Args:
            num_cards: Number of cards to deal (default: based on betting round)
            betting_round: The betting round to deal for (flop, turn, river)
        
        Returns:
            List of newly dealt card strings
        """
        # Default to dealing flop if not specified
        if num_cards is None:
            num_cards = 3
        
        # Deal new cards from the deck
        new_cards = []
        for _ in range(num_cards):
            card = self.deck.deal(1)[0]  # Deal one card from the deck
            new_cards.append(card)  # Convert Card object to string

        # Update the community cards
        self.community_cards += new_cards
        
        # If this is a specific round, update the betting round in the game state
        if betting_round is not None:
            if betting_round == 'FLOP':
                self.betting_round = BettingRound.FLOP
            elif betting_round == 'TURN':
                self.betting_round = BettingRound.TURN
            elif betting_round == 'RIVER':
                self.betting_round = BettingRound.RIVER

        return new_cards

    # You might also want to add methods for dealing flop, turn, and river specifically:

    def deal_flop(self):
        """Deal the flop (3 community cards)."""
        return self.deal_community_cards(num_cards=3, betting_round="FLOP")

    def deal_turn(self):
        """Deal the turn (1 community card)."""
        return self.deal_community_cards(num_cards=1, betting_round="TURN")

    def deal_river(self):
        """Deal the river (1 community card)."""
        return self.deal_community_cards(num_cards=1, betting_round="RIVER")


    def check_and_deal_next_round(self):
        """
        Check if we need to deal community cards for the next betting round.
        """
        # Check if we need to deal community cards
        if self.betting_round.name == "PREFLOP" and len(self.community_cards) == 0:
            # All players have acted, deal the flop
            if self._all_players_acted():
                self.deal_flop()
                
        elif self.betting_round.name == "FLOP" and len(self.community_cards) == 3:
            # All players have acted on the flop, deal the turn
            if self._all_players_acted():
                self.deal_turn()

        elif self.betting_round.name == "TURN" and len(self.community_cards) == 4:
            # All players have acted on the turn, deal the river
            if self._all_players_acted():
                self.deal_river()
        
    def _all_players_acted(self):
        """
        Helper method to check if all active players have acted in the current round.
        
        Returns:
            True if all players have acted, False otherwise
        """
        active_players = [i for i, player in self.players.items()
                         if not player.has_folded and player.is_active and not player.is_all_in]
        if len(active_players) > 1:
            return False
        else:
            return True
    
    def _calculate_side_pots(self):
        """
        Calculates side pots when there are all-in players.
        
        Returns:
            List of tuples (pot_amount, set of eligible_player_ids)
        """
        # Reset side pots
        self.side_pots = []
        
        # Get all players who are active (not folded)
        active_players = [p for p in self.players.values() if not p.has_folded and p.is_active]
        if not active_players:
            return []
            
        # Sort players by their total bet in the hand
        sorted_players = sorted(active_players, key=lambda p: p.total_bet_in_hand)
        
        # Calculate side pots
        prev_bet = 0
        eligible_players = set()
        
        for player in sorted_players:
            current_bet = player.total_bet_in_hand
            if current_bet > prev_bet:
                this_split_eligible_players = set()
                # Calculate side pot for this bet level
                pot_amount = 0
                for p in self.players.values():
                    contribution = min(p.total_bet_in_hand, current_bet) - prev_bet

                    if contribution > 0:
                        pot_amount += contribution
                        if not p.has_folded:
                            this_split_eligible_players.add(p.player_id)
                    
                if pot_amount > 0 and this_split_eligible_players:
                    self.side_pots.append((pot_amount, this_split_eligible_players.copy()))
                    
            # Add player to eligible players for higher pots
            eligible_players.add(player.player_id)
            prev_bet = current_bet
         
        # Add the final main pot with all remaining players
        remaining_pot = self.pot - sum(amount for amount, _ in self.side_pots)
        if remaining_pot > 0 and eligible_players:
            self.side_pots.append((remaining_pot, eligible_players))
        
        # Log side pots for debugging
        for i, (amount, players) in enumerate(self.side_pots):
            pot_name = "Main Pot" if 0 else f"Side Pot {i+1}"
            self._record_action("POT_INFO", -1, amount, 
                            {"pot_name": pot_name, "eligible_players": list(players)})
        
        return self.side_pots

    def _showdown_with_side_pots(self):
        """
        Performs a showdown with side pots to determine winners and distribute the pot.
        """
        # Determine player hands for showdown
        active_player_hands = {}
        for player_id, player in self.players.items():
            if not player.has_folded and player.is_active:
                active_player_hands[player_id] = player.hole_cards
        
        # Calculate side pots if there are all-in players
        all_in_players = [p for p in self.players.values() if p.is_all_in]
        if all_in_players:
            self._calculate_side_pots()
            
            # Distribute each side pot to the winner(s)
            for i, (pot_amount, eligible_players) in enumerate(self.side_pots):
                # Get only eligible hands (players who can win this pot)
                eligible_hands = {p_id: active_player_hands[p_id] for p_id in eligible_players
                                if p_id in active_player_hands}
                
                if not eligible_hands:
                    continue  # Skip if no eligible players for this pot
                    
                # Determine winners for this pot
                winners = PokerHandEvaluator.determine_winners(eligible_hands, self.community_cards)
                
                # Split the pot among winners
                pot_per_winner = pot_amount / len(winners)
                for winner_id in winners:
                    self.players[winner_id].win_amount(pot_per_winner)
                    
                # Record winner action for this pot
                pot_name = "Main Pot" if pot_amount == self.side_pots[-1][0] else "Side Pot"
                action = self._record_action("POT_WINNER", -1, pot_amount, 
                                {"pot_name": pot_name, "winners": winners, 
                                "amount_per_winner": pot_per_winner})
                
                self.side_pots[i] += (action,)
                
        else:
            # Single pot - standard showdown
            winners = PokerHandEvaluator.determine_winners(active_player_hands, self.community_cards)
            pot_per_winner = self.pot / len(winners)
            
            for winner_id in winners:
                self.players[winner_id].win_amount(pot_per_winner)
                
            self._record_action("WINNER", -1, self.pot, 
                            {"winners": winners, "amount_per_winner": pot_per_winner})
        
        # Update balance
        for i, player in self.players.items():
            player.update_balance()
        
        # Mark hand as complete
        self.hand_complete = True
    