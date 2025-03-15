class PokerPlayer:
    """Represents a poker player in the game."""
    
    def __init__(self, player_id: int, stack: float = 100.0):
        self.player_id = player_id
        self.stack = stack
        self.starting_stack = self.stack
        self.balance = 0
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
        self.update_balance()
        self.starting_stack = self.stack
    
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
    
    def update_balance(self):
        """Balance get updated at the end of hand"""
        if self.stack < 100:
            self.balance = self.balance - (100 - self.stack)
            self.stack = 100
        
        # Stores money to balance when stack gets too large
        if self.stack > 1500:
            self.balance = self.balance + self.stack - 100
            self.stack = 100