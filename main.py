import argparse
from running_modes.ui import run_ui
from running_modes.train import train
from running_modes.evaluate import evaluate

try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False
    print("PyTorch not found. RL training will not be available.")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Poker RL Environment")
    parser.add_argument("--mode", choices=["ui", "train", "evaluate"], default="ui",
                      help="Run mode: ui, train, or evaluate")
    parser.add_argument("--n_players", type=int, default=6,
                      help="Number of players (2-6)")
    parser.add_argument("--human_player", type=int, default=None,
                      help="ID of human player (0-5), default is None (all AI)")
    parser.add_argument("--small_blind", type=float, default=0.5,
                      help="Small blind amount")
    parser.add_argument("--big_blind", type=float, default=1.0,
                      help="Big blind amount")
    parser.add_argument("--initial_stack", type=float, default=100.0,
                      help="Initial stack size")
    
    # Training parameters
    if has_torch:
        parser.add_argument("--episodes", type=int, default=100000,
                          help="Number of episodes to train")
        parser.add_argument("--batch_size", type=int, default=32,
                          help="Batch size for training")
        parser.add_argument("--target_update", type=int, default=100,
                          help="Episodes between target network updates")
        parser.add_argument("--save_path", type=str, default="v01",
                          help="Path to save the trained model")
        parser.add_argument("--save_freq", type=int, default=1000,
                          help="Path to save the trained model")
        parser.add_argument("--select_agent", type=int, default=0,
                          help="Select agent for evaluation")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    if args.mode == "ui":
        run_ui(args)
    elif args.mode == "train":
        train(args)
    elif args.mode == "evaluate":
        evaluate(args)


if __name__ == "__main__":
    main()
