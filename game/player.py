class PokerPlayer:
    """Represents a poker player in the game."""
    
    def __init__(self, player_id: int, stack: float = 100.0):
        self.player_id = player_id
        self.stack = stack
        self.hole_cards = []
        self.is_active = True
        self.has_folded = False
        self.current_bet = 0.0
        self.total_bet_in_hand = 0.0
        self.is_all_in = False
        self.last_action = None
        self.position = None  # Button position is 0, SB is 1, BB is 2, etc.
    
    def reset_for_new_hand(self):
        """Resets player state for a new hand."""
        self.hole_cards = []
        self.is_active = True
        self.has_folded = False
        self.current_bet = 0.0
        self.total_bet_in_hand = 0.0
        self.is_all_in = False
        self.last_action = None
    
    def bet(self, amount: float) -> float:
        """Player bets a specific amount and returns the actual amount bet."""
        amount = min(amount, self.stack)
        self.stack -= amount
        self.current_bet += amount
        self.total_bet_in_hand += amount
        
        if self.stack == 0:
            self.is_all_in = True
            
        return amount
    
    def fold(self):
        """Player folds their hand."""
        self.has_folded = True
        self.is_active = False
    
    def win_amount(self, amount: float):
        """Player wins a specific amount."""
        self.stack += amount