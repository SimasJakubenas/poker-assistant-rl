from collections import defaultdict
from typing import Dict, List, Tuple
from game.essentials import Card, HandRank


class PokerHandEvaluator:
    """Evaluates poker hands and determines winners."""
    
    @staticmethod
    def evaluate_hand(hand: List[Card]) -> Tuple[HandRank, List[int]]:
        """
        Evaluates a poker hand and returns its rank and kickers.
        
        Args:
            hand: A list of 5-7 Card objects
            
        Returns:
            Tuple of (HandRank, list of kicker values for tiebreakers)
        """
        if len(hand) < 5:
            raise ValueError("Need at least 5 cards to evaluate a poker hand")
        
        # For 6 or 7 card hands, we need to find the best 5-card combination
        if len(hand) > 5:
            best_rank = (HandRank.HIGH_CARD, [])
            for hand_combo in PokerHandEvaluator._get_all_5_card_combinations(hand):
                rank = PokerHandEvaluator.evaluate_hand(hand_combo)
                if PokerHandEvaluator._compare_hand_ranks(rank, best_rank) > 0:
                    best_rank = rank
            return best_rank
        
        # Get counts of each rank
        rank_counts = defaultdict(int)
        for card in hand:
            rank_counts[card.rank] += 1
        
        # Check for flush
        suits = [card.suit for card in hand]
        is_flush = len(set(suits)) == 1
        
        # Check for straight
        ranks = sorted([Card.RANKS.index(card.rank) for card in hand])
        is_straight = False
        if ranks == [0, 1, 2, 3, 12]:  # A-5 straight (Ace low)
            is_straight = True
            ranks = [-1, 0, 1, 2, 3]  # Rearrange for correct kicker order
        elif ranks[-1] - ranks[0] == 4 and len(set(ranks)) == 5:
            is_straight = True
        
        # Determine hand rank and kickers
        if is_straight and is_flush:
            if ranks == [8, 9, 10, 11, 12]:  # T-J-Q-K-A
                return (HandRank.ROYAL_FLUSH, [])
            return (HandRank.STRAIGHT_FLUSH, [max(ranks)])
            
        if 4 in rank_counts.values():
            quads_rank = [r for r, count in rank_counts.items() if count == 4][0]
            kicker = [r for r, count in rank_counts.items() if count == 1][0]
            return (HandRank.FOUR_OF_A_KIND, [Card.RANKS.index(quads_rank), Card.RANKS.index(kicker)])
            
        if 3 in rank_counts.values() and 2 in rank_counts.values():
            trips_rank = [r for r, count in rank_counts.items() if count == 3][0]
            pair_rank = [r for r, count in rank_counts.items() if count == 2][0]
            return (HandRank.FULL_HOUSE, [Card.RANKS.index(trips_rank), Card.RANKS.index(pair_rank)])
            
        if is_flush:
            kickers = sorted([Card.RANKS.index(card.rank) for card in hand], reverse=True)
            return (HandRank.FLUSH, kickers)
            
        if is_straight:
            return (HandRank.STRAIGHT, [max(ranks)])
            
        if 3 in rank_counts.values():
            trips_rank = [r for r, count in rank_counts.items() if count == 3][0]
            kickers = sorted([Card.RANKS.index(r) for r, count in rank_counts.items() if count == 1], reverse=True)
            return (HandRank.THREE_OF_A_KIND, [Card.RANKS.index(trips_rank)] + kickers)
            
        if list(rank_counts.values()).count(2) == 2:
            pair_ranks = sorted([Card.RANKS.index(r) for r, count in rank_counts.items() if count == 2], reverse=True)
            kicker = [Card.RANKS.index(r) for r, count in rank_counts.items() if count == 1][0]
            return (HandRank.TWO_PAIR, pair_ranks + [kicker])
            
        if 2 in rank_counts.values():
            pair_rank = [r for r, count in rank_counts.items() if count == 2][0]
            kickers = sorted([Card.RANKS.index(r) for r, count in rank_counts.items() if count == 1], reverse=True)
            return (HandRank.PAIR, [Card.RANKS.index(pair_rank)] + kickers)
            
        kickers = sorted([Card.RANKS.index(card.rank) for card in hand], reverse=True)
        return (HandRank.HIGH_CARD, kickers)
    
    @staticmethod
    def _get_all_5_card_combinations(cards: List[Card]) -> List[List[Card]]:
        """Returns all 5-card combinations from a list of 6 or 7 cards."""
        from itertools import combinations
        return list(combinations(cards, 5))
    
    @staticmethod
    def _compare_hand_ranks(rank1: Tuple[HandRank, List[int]], rank2: Tuple[HandRank, List[int]]) -> int:
        """Compares two hand ranks. Returns 1 if rank1 > rank2, -1 if rank1 < rank2, 0 if equal."""
        if rank1[0].value > rank2[0].value:
            return 1
        if rank1[0].value < rank2[0].value:
            return -1
        
        # Compare kickers
        for k1, k2 in zip(rank1[1], rank2[1]):
            if k1 > k2:
                return 1
            if k1 < k2:
                return -1
        
        return 0
    
    @staticmethod
    def determine_winners(player_hands: Dict[int, List[Card]], community_cards: List[Card]) -> List[int]:
        """
        Determines the winner(s) given player hands and community cards.
        
        Args:
            player_hands: Dictionary mapping player_id to their hole cards
            community_cards: List of community cards
            
        Returns:
            List of player_ids who won (can be multiple in case of a tie)
        """
        best_rank = None
        winners = []
        
        for player_id, hole_cards in player_hands.items():
            hand = hole_cards + community_cards
            rank = PokerHandEvaluator.evaluate_hand(hand)
            
            if best_rank is None or PokerHandEvaluator._compare_hand_ranks(rank, best_rank) > 0:
                best_rank = rank
                winners = [player_id]
            elif PokerHandEvaluator._compare_hand_ranks(rank, best_rank) == 0:
                winners.append(player_id)
                
        return winners