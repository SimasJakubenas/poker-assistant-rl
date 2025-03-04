import numpy as np
from typing import Dict, List


class FeatureExtractor:
    """
    Extracts features from the poker game state for RL agents.
    Converts the game state into a normalized vector representation.
    """
    
    @staticmethod
    def extract_features(state: Dict, player_id: int) -> np.ndarray:
        """
        Extract features from the game state for a specific player.
        
        Args:
            state: Game state dictionary
            player_id: Player ID to extract features for
            
        Returns:
            Numpy array of features
        """
        # Get player-specific information
        player_info = state["players"][str(player_id)]
        
        # Basic game state features
        features = []
        
        # Player position (normalized)
        position = player_info["position"] / 5.0  # Normalize by max position (6-max table)
        features.append(position)
        
        # Player stack (normalized)
        stack = player_info["stack"] / 200.0  # Normalize by 2x initial stack
        features.append(stack)
        
        # Pot size relative to player stack
        pot_to_stack = state["pot"] / max(1.0, player_info["stack"])
        pot_to_stack = min(3.0, pot_to_stack) / 3.0  # Cap at 3x stack and normalize
        features.append(pot_to_stack)
        
        # Current bet relative to pot
        if state["pot"] > 0:
            bet_to_pot = state["current_bet"] / state["pot"]
            bet_to_pot = min(2.0, bet_to_pot) / 2.0  # Cap at 2x pot and normalize
        else:
            bet_to_pot = 0.0
        features.append(bet_to_pot)
        
        # Betting round
        round_map = {"PREFLOP": 0, "FLOP": 1, "TURN": 2, "RIVER": 3, None: 4}
        betting_round = round_map.get(state["betting_round"], 0)
        for i in range(5):
            features.append(1.0 if betting_round == i else 0.0)
            
        # Player hole cards (if available)
        hole_cards = player_info["hole_cards"]
        if hole_cards:
            # Extract card features (rank, suit, etc.)
            card_features = FeatureExtractor._extract_card_features(hole_cards)
            features.extend(card_features)
        else:
            # Placeholder for unknown cards
            features.extend([0.0] * 10)  # Adjust size based on your card feature representation
            
        # Community cards
        community_cards = state["community_cards"]
        community_card_features = FeatureExtractor._extract_card_features(community_cards)
        features.extend(community_card_features)
        
        # Number of active players
        active_players = sum(1 for p in state["players"].values() if p["is_active"] and not p["has_folded"])
        active_ratio = active_players / 6.0  # Normalize by max players
        features.append(active_ratio)
        
        # Button distance (how far player is from the button)
        
        button_pos = state["button_pos"]
        player_pos = player_info["position"]
        button_distance = (player_pos - 0) % 6  # Button is position 0
        button_distance = button_distance / 5.0  # Normalize
        features.append(button_distance)
        
        return np.array(features)
    
    @staticmethod
    def _extract_card_features(cards: List[str]) -> List[float]:
        """
        Extract features from a list of cards.
        
        Args:
            cards: List of card strings (e.g., ["Ah", "Kd"])
            
        Returns:
            List of card features
        """
        if not cards:
            return [0.0] * 10  # Return zeros for no cards
            
        # Simple representation: encode rank and suit
        features = []
        
        # Define rank and suit mappings
        rank_map = {r: i / 12.0 for i, r in enumerate("23456789TJQKA")}
        suit_map = {"h": 0.25, "d": 0.5, "c": 0.75, "s": 1.0}
        
        for card_str in cards[:5]:  # Limit to 5 cards (max community cards)
            if len(card_str) < 2:
                features.extend([0.0, 0.0])
                continue
                
            rank = card_str[0]
            suit = card_str[1]
            
            # Add normalized rank and suit
            features.append(rank_map.get(rank, 0.0))
            features.append(suit_map.get(suit, 0.0))
            
        # Pad to fixed length if needed
        while len(features) < 10:
            features.extend([0.0, 0.0])
            
        return features
