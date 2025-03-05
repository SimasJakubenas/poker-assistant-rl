import pygame
import sys
import math
import time
from typing import Dict, List, Tuple, Optional

# Import the poker environment
from rl_model.random_agent import RandomAgent
from rl_model.rl_environment import PokerEnv, PokerAction


class PokerTableUI:
    """User interface for the poker environment using Pygame."""
    
    # Define colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREEN = (0, 128, 0)
    DARK_GREEN = (0, 100, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    GOLD = (212, 175, 55)
    GRAY = (128, 128, 128)
    LIGHT_GRAY = (200, 200, 200)
    
    # Card colors
    CARD_BACK = (53, 101, 77)
    SUIT_COLORS = {
        'h': (255, 0, 0),    # Hearts: Red
        'd': (255, 0, 0),    # Diamonds: Red
        'c': (0, 0, 0),      # Clubs: Black
        's': (0, 0, 0)       # Spades: Black
    }
    
    # Suit symbols (Unicode)
    SUIT_SYMBOLS = {
        'h': '♥',
        'd': '♦',
        'c': '♣',
        's': '♠'
    }
    
    def __init__(self, env: PokerEnv, width: int = 1024, height: int = 768, human_player: Optional[int] = None):
        """
        Initialize the poker table UI.
        
        Args:
            env: Poker environment
            width: Screen width
            height: Screen height
            human_player: ID of human player (None for all AI players)
        """
        self.env = env
        self.width = width
        self.height = height
        self.human_player = human_player
        
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        
        # Create screen
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("No-Limit Texas Hold'em Poker")
        
        # Load fonts
        self.large_font = pygame.font.SysFont('Arial', 32)
        self.medium_font = pygame.font.SysFont('Arial', 24)
        self.small_font = pygame.font.SysFont('Arial', 18)
        self.card_font = pygame.font.SysFont('Arial', 30, bold=True)
        
        # Table dimensions
        self.table_center = (width // 2, height // 2)
        self.table_radius = min(width, height) * 0.35
        
        # Player positions (6-max)
        self.player_positions = self._calculate_player_positions()
        
        # Card dimensions
        self.card_width = 80
        self.card_height = 120
        
        # Animation speeds
        self.animation_speed = 0.5  # seconds
        
        # UI state
        self.selected_action = None
        self.selected_bet_amount = None
        self.action_buttons = []
        self.bet_slider_rect = None
        self.bet_slider_button_rect = None
        self.dragging_slider = False
        
        # Initialize the game
        self.game_state = None
        self.current_player = None
        self.agents = []
        
        # Event tracking
        self.last_action_time = time.time()
        self.action_delay = 1.0  # seconds between AI actions
    
    def _calculate_player_positions(self) -> List[Tuple[int, int]]:
        """
        Calculate the positions of players around the table.
        
        Returns:
            List of (x, y) positions for each player
        """
        positions = []
        center_x, center_y = self.table_center
        radius = self.table_radius * 1.3
        
        # 6-max table positions
        angles = [math.pi/2, 5*math.pi/6, 7*math.pi/6, 3*math.pi/2, 11*math.pi/6, math.pi/6]
        
        for angle in angles:
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            positions.append((int(x), int(y)))
            
        return positions
    
    def start_game(self, n_players: int = 6):
        """
        Start a new poker game.
        
        Args:
            n_players: Number of players (2-6)
        """
        # Create agents for AI players
        self.agents = []
        for i in range(n_players):
            if i == self.human_player:
                self.agents.append(None)  # Human player
            else:
                self.agents.append(RandomAgent(i))
                
        # Reset the environment
        self.game_state = self.env.reset()
        self.current_player = self.env.current_player
        
        # Start the game loop
        self.game_loop()
    
    def game_loop(self):
        """Main game loop. Modified to check for betting round progression."""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                
                # Handle human player actions
                if self.human_player is not None and self.current_player == self.human_player:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        self._handle_mouse_click(event.pos)
                    elif event.type == pygame.MOUSEBUTTONUP:
                        self.dragging_slider = False
                    elif event.type == pygame.MOUSEMOTION and self.dragging_slider:
                        self._handle_mouse_drag(event.pos)
            
            # Execute AI actions with delay
            if self.human_player != self.current_player and self.current_player is not None:
                current_time = time.time()
                if current_time - self.last_action_time >= self.action_delay:
                    self._execute_ai_action()
                    self.last_action_time = current_time
            
            all_players_acted = self.env.table._all_players_acted()
            
            if all_players_acted == True:
                pygame.time.wait(1500)
                self.env.table._advance_game()
            
            if self.env.table.hand_complete == True:
                self.env.terminal = True
            
            # Render the game
            self.render()
            
            # Cap at 60 FPS
            clock.tick(60)
            
            # Check if game is over
            if self.env.terminal:
                # Display the final state for a while
                pygame.time.wait(3000)
                self.game_state = self.env.reset()
                self.current_player = self.env.current_player
                self.last_action_time = time.time()
        
        # Clean up
        pygame.quit()
        sys.exit()
    
    def _execute_ai_action(self):
        """Execute an action for an AI player. Modified to check for round progression."""
        if self.current_player is None or self.env.terminal:
            return
            
        agent = self.agents[self.current_player]
        if agent is None:
            return  # Human player
            
        # Get action from agent
        action_idx = agent.act(self.game_state)
        
        # Execute the action
        self.game_state, reward, done, _ = self.env.step(action_idx)
        self.current_player = self.env.current_player
    
    def _handle_mouse_click(self, pos: Tuple[int, int]):
        """
        Handle mouse click event.
        
        Args:
            pos: Mouse position (x, y)
        """
        # Check action buttons
        for button in self.action_buttons:
            rect, action, amount = button
            if rect.collidepoint(pos):
                self._execute_human_action(action, amount)
                return
                
        # Check bet slider
        if self.bet_slider_rect and self.bet_slider_rect.collidepoint(pos):
            self.dragging_slider = True
            self._update_slider_position(pos[0])
    
    def _handle_mouse_drag(self, pos: Tuple[int, int]):
        """
        Handle mouse drag event for the bet slider.
        
        Args:
            pos: Mouse position (x, y)
        """
        if self.dragging_slider:
            self._update_slider_position(pos[0])
    
    def _update_slider_position(self, x_pos: int):
        """
        Update the bet slider position and selected bet amount.
        
        Args:
            x_pos: X position of the mouse
        """
        if not self.bet_slider_rect:
            return
            
        # Calculate new slider position
        x_min = self.bet_slider_rect.left
        x_max = self.bet_slider_rect.right
        x_pos = max(x_min, min(x_pos, x_max))
        
        # Calculate bet amount based on slider position
        player_info = self.game_state[self.human_player]["players"][self.human_player]
        min_bet = self.env.table.big_blind
        max_bet = player_info["stack"]
        
        if "legal_actions" in self.game_state[self.human_player]:
            legal_actions = self.game_state[self.human_player]["legal_actions"]
            if PokerAction.BET in legal_actions and legal_actions[PokerAction.BET]:
                min_bet = min(legal_actions[PokerAction.BET])
            elif PokerAction.RAISE in legal_actions and legal_actions[PokerAction.RAISE]:
                min_bet = min(legal_actions[PokerAction.RAISE])
        
        # Linear interpolation between min and max bet
        ratio = (x_pos - x_min) / max(1, x_max - x_min)
        bet_amount = min_bet + ratio * (max_bet - min_bet)
        self.selected_bet_amount = max(min_bet, min(max_bet, bet_amount))
        
        # Update slider button position
        if self.bet_slider_button_rect:
            self.bet_slider_button_rect.centerx = x_pos
    
    def _execute_human_action(self, action: PokerAction, amount: Optional[float] = None):
        """Execute an action for the human player. Modified to check for round progression."""
        if self.current_player != self.human_player or self.env.terminal:
            return
            
        # Get legal actions
        legal_actions = self.game_state[self.human_player]["legal_actions"]
        
        # Validate action
        if action not in legal_actions:
            return
            
        # Determine action index
        action_mapping = {}
        action_idx = 0
        
        # Map each legal action to an index
        for action_type, amounts in legal_actions.items():
            if not amounts:  # Actions like FOLD, CHECK
                action_mapping[action_idx] = (action_type, None)
                action_idx += 1
            else:
                for amt in amounts:
                    action_mapping[action_idx] = (action_type, amt)
                    action_idx += 1
        
        # Find matching action
        target_action_idx = None
        for idx, (act, amt) in action_mapping.items():
            if act == action:
                if amt is None or amount is None:
                    target_action_idx = idx
                    break
                elif abs(amt - amount) < 0.01:
                    target_action_idx = idx
                    break
        
        if target_action_idx is None:
            # Find closest bet amount
            if action in [PokerAction.BET, PokerAction.RAISE, PokerAction.ALL_IN]:
                closest_idx = None
                closest_diff = float('inf')
                
                for idx, (act, amt) in action_mapping.items():
                    if act == action and amt is not None:
                        diff = abs(amt - amount)
                        if diff < closest_diff:
                            closest_diff = diff
                            closest_idx = idx
                
                target_action_idx = closest_idx
        
        if target_action_idx is not None:
            # Execute the action
            self.game_state, reward, done, _ = self.env.step(target_action_idx)
            self.current_player = self.env.current_player
            self.selected_action = None
            self.selected_bet_amount = None        
    
    def render(self):
        """Render the game state to the screen."""
        # Fill background
        self.screen.fill(self.WHITE)
        
        # Draw poker table
        self._draw_table()
        
        # Draw community cards
        self._draw_community_cards()
        
        # Draw pot
        self._draw_pot()
        
        # Draw players
        self._draw_players()
        
        # Draw action buttons for human player
        if self.human_player is not None and self.current_player == self.human_player:
            self._draw_action_buttons()
        
        # Draw game info
        self._draw_game_info()
        
        # Update the display
        pygame.display.flip()
    
    def _draw_table(self):
        """Draw the poker table."""
        center_x, center_y = self.table_center
        
        # Draw table
        pygame.draw.circle(self.screen, self.DARK_GREEN, self.table_center, self.table_radius)
        pygame.draw.circle(self.screen, self.GREEN, self.table_center, self.table_radius - 5)
        
        # Draw dealer button
        if self.game_state:
            button_pos = self.game_state[0]["button_pos"]
            seat_positions = self._get_seat_positions()
            if button_pos in seat_positions:
                btn_x, btn_y = seat_positions[button_pos]
                btn_radius = 15
                pygame.draw.circle(self.screen, self.WHITE, (btn_x, btn_y), btn_radius)
                pygame.draw.circle(self.screen, self.BLACK, (btn_x, btn_y), btn_radius - 2, 2)
                dealer_text = self.small_font.render("D", True, self.BLACK)
                self.screen.blit(dealer_text, (btn_x - dealer_text.get_width() // 2, 
                                             btn_y - dealer_text.get_height() // 2))
    
    def _draw_community_cards(self):
        """Draw the community cards in the center of the table."""
        if not self.game_state:
            return

        community_cards = self.env.table.get_state()['community_cards']
        if not community_cards:
            return
            
        center_x, center_y = self.table_center
        card_spacing = self.card_width + 10
        total_width = (len(community_cards) - 1) * card_spacing + self.card_width
        start_x = center_x - total_width // 2
        
        for i, card_str in enumerate(community_cards):
            card_x = start_x + i * card_spacing
            card_y = center_y - self.card_height // 2
            self._draw_card(card_str, card_x, card_y)
    
    def _draw_pot(self):
        """Draw the pot information."""
        if not self.game_state:
            return
            
        pot = self.game_state[0]["pot"]
        current_bet = self.game_state[0]["current_bet"]
        
        center_x, center_y = self.table_center
        pot_y = center_y + self.card_height // 2 + 20
        
        # Draw pot amount
        pot_text = self.medium_font.render(f"Pot: ${pot:.2f}", True, self.WHITE)
        self.screen.blit(pot_text, (center_x - pot_text.get_width() // 2, pot_y))
        
        # Draw current bet
        if current_bet > 0:
            bet_text = self.medium_font.render(f"Current Bet: ${current_bet:.2f}", True, self.WHITE)
            self.screen.blit(bet_text, (center_x - bet_text.get_width() // 2, pot_y + 30))
    
    def _draw_players(self):
        """Draw the players around the table."""
        if not self.game_state:
            return
            
        # Get player information
        players_info = self.game_state[0]["players"]
        
        # Draw each player
        for player_id, info in players_info.items():
            player_id = int(player_id)
            if player_id < len(self.player_positions):
                pos_x, pos_y = self.player_positions[player_id]
                self._draw_player(player_id, info, pos_x, pos_y)
    
    def _draw_player(self, player_id: int, info: Dict, pos_x: int, pos_y: int):
        """
        Draw a single player.
        
        Args:
            player_id: Player ID
            info: Player information dictionary
            pos_x: X position
            pos_y: Y position
        """
        # Draw player background
        rect_width = 200
        rect_height = 150
        rect = pygame.Rect(pos_x - rect_width // 2, pos_y - rect_height // 2, rect_width, rect_height)
        
        # Highlight current player
        if player_id == self.current_player:
            pygame.draw.rect(self.screen, self.GOLD, rect, 0, 10)
            pygame.draw.rect(self.screen, self.BLACK, rect, 2, 10)
        else:
            pygame.draw.rect(self.screen, self.LIGHT_GRAY, rect, 0, 10)
            pygame.draw.rect(self.screen, self.BLACK, rect, 2, 10)
        
        # Player name/ID
        player_text = self.medium_font.render(f"Player {player_id}", True, self.BLACK)
        self.screen.blit(player_text, (pos_x - player_text.get_width() // 2, pos_y - rect_height // 2 + 10))
        
        # Player stack
        stack_text = self.small_font.render(f"Stack: ${info['stack']:.2f}", True, self.BLACK)
        self.screen.blit(stack_text, (pos_x - stack_text.get_width() // 2, pos_y - rect_height // 2 + 40))
        
        # Player cards
        card_y = pos_y - 5
        if info['hole_cards']:
            card1_x = pos_x - self.card_width - 5
            card2_x = pos_x + 5
            
            # Draw the cards
            self._draw_card(info['hole_cards'][0], card1_x, card_y, 0.6)
            self._draw_card(info['hole_cards'][1], card2_x, card_y, 0.6)
        else:
            # Draw card backs for other players
            if info['is_active'] and not info['has_folded']:
                card1_x = pos_x - self.card_width * 0.6 - 5
                card2_x = pos_x + 5
                self._draw_card_back(card1_x, card_y, 0.6)
                self._draw_card_back(card2_x, card_y, 0.6)
        
        # Player status
        status_text = ""
        if info['has_folded']:
            status_text = "FOLDED"
        elif info['is_all_in']:
            status_text = "ALL-IN"
        
        if status_text:
            status_render = self.small_font.render(status_text, True, self.RED)
            self.screen.blit(status_render, (pos_x - status_render.get_width() // 2, pos_y + rect_height // 2 - 30))
        
        # Current bet
        if info['current_bet'] > 0:
            bet_text = self.small_font.render(f"Bet: ${info['current_bet']:.2f}", True, self.BLACK)
            self.screen.blit(bet_text, (pos_x - bet_text.get_width() // 2, pos_y + rect_height // 2 - 50))
        
        # Last action
        if info['last_action']:
            action_text = self.small_font.render(f"{info['last_action']}", True, self.BLACK)
            self.screen.blit(action_text, (pos_x - action_text.get_width() // 2, pos_y + rect_height // 2 - 30))
    
    def _draw_card(self, card_str: str, x: int, y: int, scale: float = 1.0):
        """
        Draw a playing card.
        
        Args:
            card_str: Card string (e.g., "Ah" for Ace of hearts)
            x: X position
            y: Y position
            scale: Scale factor (1.0 is normal size)
        """
        if not card_str or len(card_str) < 2:
            return
            
        # Parse card string
        rank = card_str[0]
        suit = card_str[1]
        
        # Scale card dimensions
        width = int(self.card_width * scale)
        height = int(self.card_height * scale)
        
        # Draw card background
        card_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, self.WHITE, card_rect, 0, 5)
        pygame.draw.rect(self.screen, self.BLACK, card_rect, 2, 5)
        
        # Draw card rank and suit
        suit_symbol = self.SUIT_SYMBOLS.get(suit, suit)
        rank_text = self.card_font.render(rank, True, self.SUIT_COLORS.get(suit, self.BLACK))
        suit_text = self.card_font.render(suit_symbol, True, self.SUIT_COLORS.get(suit, self.BLACK))
        
        # Position the rank and suit
        rank_x = x + 5
        rank_y = y + 5
        suit_x = x + width - suit_text.get_width() - 5
        suit_y = y + height - suit_text.get_height() - 5
        
        self.screen.blit(rank_text, (rank_x, rank_y))
        self.screen.blit(suit_text, (suit_x, suit_y))
        
        # Draw a larger suit symbol in the center
        big_suit_text = pygame.font.SysFont('Arial', int(36 * scale), bold=True).render(suit_symbol, True, self.SUIT_COLORS.get(suit, self.BLACK))
        big_suit_x = x + width // 2 - big_suit_text.get_width() // 2
        big_suit_y = y + height // 2 - big_suit_text.get_height() // 2
        self.screen.blit(big_suit_text, (big_suit_x, big_suit_y))
    
    def _draw_card_back(self, x: int, y: int, scale: float = 1.0):
        """
        Draw the back of a playing card.
        
        Args:
            x: X position
            y: Y position
            scale: Scale factor (1.0 is normal size)
        """
        # Scale card dimensions
        width = int(self.card_width * scale)
        height = int(self.card_height * scale)
        
        # Draw card background
        card_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, self.CARD_BACK, card_rect, 0, 5)
        pygame.draw.rect(self.screen, self.BLACK, card_rect, 2, 5)
        
        # Draw pattern on back
        pattern_rect = pygame.Rect(x + 5, y + 5, width - 10, height - 10)
        pygame.draw.rect(self.screen, self.BLACK, pattern_rect, 2, 3)
    
    def _draw_action_buttons(self):
        """Draw action buttons for the human player."""
        if self.human_player is None or self.current_player != self.human_player:
            return
            
        # Get legal actions
        legal_actions = self.game_state[self.human_player]["legal_actions"]
        if not legal_actions:
            return
            
        # Define button properties
        button_width = 120
        button_height = 40
        button_spacing = 20
        total_width = len(legal_actions) * (button_width + button_spacing) - button_spacing
        start_x = self.width // 2 - total_width // 2
        button_y = self.height - 100
        
        # Reset action buttons
        self.action_buttons = []
        
        # Draw action buttons
        i = 0
        for action, amounts in legal_actions.items():
            button_x = start_x + i * (button_width + button_spacing)
            button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
            
            # Button color and amount
            button_color = self.LIGHT_GRAY
            action_amount = None
            
            if action == PokerAction.FOLD:
                button_text = "FOLD"
            elif action == PokerAction.CHECK:
                button_text = "CHECK"
            elif action == PokerAction.CALL:
                button_text = f"CALL ${amounts[0]:.2f}"
                action_amount = amounts[0]
            elif action == PokerAction.BET:
                if self.selected_bet_amount and action == self.selected_action:
                    button_text = f"BET ${self.selected_bet_amount:.2f}"
                    action_amount = self.selected_bet_amount
                    button_color = self.GOLD
                else:
                    button_text = "BET"
                    self.selected_action = action
                    self.selected_bet_amount = amounts[0]
            elif action == PokerAction.RAISE:
                if self.selected_bet_amount and action == self.selected_action:
                    button_text = f"RAISE ${self.selected_bet_amount:.2f}"
                    action_amount = self.selected_bet_amount
                    button_color = self.GOLD
                else:
                    button_text = "RAISE"
                    self.selected_action = action
                    self.selected_bet_amount = amounts[0]
            elif action == PokerAction.ALL_IN:
                button_text = f"ALL-IN ${amounts[0]:.2f}"
                action_amount = amounts[0]
            
            # Draw button
            pygame.draw.rect(self.screen, button_color, button_rect, 0, 5)
            pygame.draw.rect(self.screen, self.BLACK, button_rect, 2, 5)
            
            # Draw button text
            text = self.small_font.render(button_text, True, self.BLACK)
            text_x = button_x + button_width // 2 - text.get_width() // 2
            text_y = button_y + button_height // 2 - text.get_height() // 2
            self.screen.blit(text, (text_x, text_y))
            
            # Store button for click detection
            self.action_buttons.append((button_rect, action, action_amount))
            
            i += 1
        
        # Draw bet slider for BET, RAISE
        if (PokerAction.BET in legal_actions or PokerAction.RAISE in legal_actions) and self.selected_action in [PokerAction.BET, PokerAction.RAISE]:
            slider_y = button_y + button_height + 20
            self._draw_bet_slider(slider_y)
    
    def _draw_bet_slider(self, y_pos: int):
        """
        Draw the bet slider.
        
        Args:
            y_pos: Y position of the slider
        """
        # Get player stack
        player_info = self.game_state[self.human_player]["players"][self.human_player]
        player_stack = player_info["stack"]
        
        # Get min/max bet
        min_bet = self.env.table.big_blind
        max_bet = player_stack
        
        if self.selected_action == PokerAction.BET and PokerAction.BET in self.game_state[self.human_player]["legal_actions"]:
            legal_bets = self.game_state[self.human_player]["legal_actions"][PokerAction.BET]
            if legal_bets:
                min_bet = min(legal_bets)
                
        elif self.selected_action == PokerAction.RAISE and PokerAction.RAISE in self.game_state[self.human_player]["legal_actions"]:
            legal_raises = self.game_state[self.human_player]["legal_actions"][PokerAction.RAISE]
            if legal_raises:
                min_bet = min(legal_raises)
        
        # Slider dimensions
        slider_width = 400
        slider_height = 10
        slider_x = self.width // 2 - slider_width // 2
        
        # Draw slider background
        slider_rect = pygame.Rect(slider_x, y_pos, slider_width, slider_height)
        pygame.draw.rect(self.screen, self.GRAY, slider_rect, 0, 5)
        self.bet_slider_rect = slider_rect
        
        # Calculate button position based on selected amount
        if self.selected_bet_amount is None:
            self.selected_bet_amount = min_bet
            
        # Clamp bet amount to valid range
        self.selected_bet_amount = max(min_bet, min(max_bet, self.selected_bet_amount))
        
        # Calculate slider button position
        ratio = (self.selected_bet_amount - min_bet) / max(1, max_bet - min_bet)
        button_x = slider_x + int(ratio * slider_width)
        button_radius = 15
        
        # Draw slider button
        pygame.draw.circle(self.screen, self.BLUE, (button_x, y_pos + slider_height // 2), button_radius)
        pygame.draw.circle(self.screen, self.BLACK, (button_x, y_pos + slider_height // 2), button_radius, 2)
        self.bet_slider_button_rect = pygame.Rect(button_x - button_radius, y_pos + slider_height // 2 - button_radius, 
                                                button_radius * 2, button_radius * 2)
        
        # Draw min and max values
        min_text = self.small_font.render(f"Min: ${min_bet:.2f}", True, self.BLACK)
        max_text = self.small_font.render(f"Max: ${max_bet:.2f}", True, self.BLACK)
        
        self.screen.blit(min_text, (slider_x - min_text.get_width() - 10, y_pos - min_text.get_height() // 2))
        self.screen.blit(max_text, (slider_x + slider_width + 10, y_pos - max_text.get_height() // 2))
        
        # Draw current value
        value_text = self.small_font.render(f"${self.selected_bet_amount:.2f}", True, self.BLACK)
        self.screen.blit(value_text, (button_x - value_text.get_width() // 2, y_pos + button_radius + 5))

    def _get_seat_positions(self):
        """Get the positions for seat indicators like dealer button."""
        seat_positions = {}
        
        for player_id, player_info in self.game_state[0]["players"].items():
            player_id = int(player_id)
            if player_id < len(self.player_positions):
                pos_x, pos_y = self.player_positions[player_id]
                
                # Position the seat indicator between the player box and the table
                table_x, table_y = self.table_center
                vector_x = table_x - pos_x
                vector_y = table_y - pos_y
                
                # Normalize and scale
                length = math.sqrt(vector_x**2 + vector_y**2)
                if length > 0:
                    norm_x = vector_x / length
                    norm_y = vector_y / length
                    
                    seat_x = pos_x + norm_x * 80
                    seat_y = pos_y + norm_y * 80
                    
                    seat_positions[player_id] = (int(seat_x), int(seat_y))
        
        return seat_positions

    def _draw_game_info(self):
        """Draw game information panel."""
        if not self.game_state:
            return
            
        # Draw game info panel
        info_rect = pygame.Rect(10, 10, 200, 100)
        pygame.draw.rect(self.screen, self.LIGHT_GRAY, info_rect, 0, 5)
        pygame.draw.rect(self.screen, self.BLACK, info_rect, 2, 5)
        
        # Draw betting round
        round_name = self.env.table.betting_round.name
        round_text = self.medium_font.render(f"Round: {round_name}", True, self.BLACK)
        self.screen.blit(round_text, (20, 20))
        
        # Draw current player indicator
        if self.current_player is not None:
            player_text = self.medium_font.render(f"Current Player: {self.current_player}", True, self.BLACK)
            self.screen.blit(player_text, (20, 50))
        
        # Display hand result if the hand is complete
        if self.env.terminal and len(self.env.table.hand_history) > 0:
            self._draw_hand_result()

    def _draw_hand_result(self):
        """Draw the hand result when the hand is complete."""
        # Find the winner action in the hand history
        winners = []
        amount_per_winner = 0
        for action in reversed(self.env.table.hand_history):
            if action["type"] in ["WINNER", "SIDE_POT_WINNER"]:
                winners = action.get("winners", [])
                amount_per_winner = action.get("amount_per_winner", 0)
                break
                
        if not winners:
            return
            
        # Draw result panel
        panel_width = 400
        panel_height = 150
        panel_x = self.width // 2 - panel_width // 2
        panel_y = self.height // 2 - panel_height // 2
        
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(self.screen, self.LIGHT_GRAY, panel_rect, 0, 10)
        pygame.draw.rect(self.screen, self.BLACK, panel_rect, 2, 10)
        
        # Draw title
        title_text = self.large_font.render("Hand Complete", True, self.BLACK)
        self.screen.blit(title_text, (panel_x + panel_width // 2 - title_text.get_width() // 2, panel_y + 20))
        
        # Draw winners
        winners_str = ", ".join([f"Player {w}" for w in winners])
        winners_text = self.medium_font.render(f"Winner(s): {winners_str}", True, self.BLACK)
        self.screen.blit(winners_text, (panel_x + panel_width // 2 - winners_text.get_width() // 2, panel_y + 60))
        
        # Draw amount won
        amount_text = self.medium_font.render(f"Amount won: ${amount_per_winner:.2f}", True, self.BLACK)
        self.screen.blit(amount_text, (panel_x + panel_width // 2 - amount_text.get_width() // 2, panel_y + 90))