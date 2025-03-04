import random
from enum import Enum
from typing import List


class Card:
    """A playing card with suit and rank."""
    SUITS = ['h', 'd', 'c', 's']  # hearts, diamonds, clubs, spades
    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    
    def __init__(self, rank: str, suit: str):
        if rank not in self.RANKS or suit not in self.SUITS:
            raise ValueError(f"Invalid card: {rank}{suit}")
        self.rank = rank
        self.suit = suit
        self.rank_index = self.RANKS.index(rank)
    
    def __str__(self):
        return f"{self.rank}{self.suit}"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit


class Deck:
    """A standard deck of 52 playing cards."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Resets the deck to its initial state."""
        self.cards = [Card(rank, suit) for suit in Card.SUITS for rank in Card.RANKS]
        self.shuffle()
    
    def shuffle(self):
        """Shuffles the deck."""
        random.shuffle(self.cards)
    
    def deal(self, n: int = 1) -> List[Card]:
        """Deals n cards from the deck."""
        if n > len(self.cards):
            raise ValueError(f"Cannot deal {n} cards, only {len(self.cards)} remaining")
        return [self.cards.pop() for _ in range(n)]


class HandRank(Enum):
    """Enum for poker hand rankings."""
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    ROYAL_FLUSH = 9
    

class PokerAction(Enum):
    """Enum for available poker actions."""
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET = 3
    RAISE = 4
    ALL_IN = 5


class BettingRound(Enum):
    """Enum for betting rounds in Texas Hold'em."""
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3