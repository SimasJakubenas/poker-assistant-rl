from rl_model.rl_environment import PokerEnv
from game_ui.table_ui import PokerTableUI


def run_ui(args):
    """Run the poker UI."""
    env = PokerEnv(n_players=args.n_players, 
                 small_blind=args.small_blind, 
                 big_blind=args.big_blind, 
                 initial_stack=args.initial_stack)
    
    ui = PokerTableUI(env, human_player=args.human_player)
    ui.start_game(args.n_players)