from rl_model.rl_environment import PokerEnv
from game_ui.table_ui import PokerTableUI


def run_ui(args):
    """Run the poker UI."""
    env = PokerEnv(n_players=args.n_players, 
                 small_blind=args.small_blind, 
                 big_blind=args.big_blind, 
                 initial_stack=args.initial_stack)
    
    ui = PokerTableUI(env, save_path=args.save_path, human_player=args.human_player, random_agents=args.random_agents)
    ui.start_game(args.n_players)