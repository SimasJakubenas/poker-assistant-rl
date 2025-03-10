import argparse
import os

# Import the poker environment and UI
from rl_model.rl_environment import PokerEnv
from rl_model.agent import DQNAgent
from rl_model.random_agent import RandomAgent
from game_ui.table_ui import PokerTableUI

# For reinforcement learning
try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False
    print("PyTorch not found. RL training will not be available.")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Poker RL Environment")
    parser.add_argument("--mode", choices=["ui", "train", "evaluate"], default="train",
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
        parser.add_argument("--episodes", type=int, default=1000,
                          help="Number of episodes to train")
        parser.add_argument("--batch_size", type=int, default=32,
                          help="Batch size for training")
        parser.add_argument("--target_update", type=int, default=10,
                          help="Episodes between target network updates")
        parser.add_argument("--save_path", type=str, default="models/dqn_poker.pt",
                          help="Path to save the trained model")
    
    return parser.parse_args()


def run_ui(args):
    """Run the poker UI."""
    env = PokerEnv(n_players=args.n_players, 
                 small_blind=args.small_blind, 
                 big_blind=args.big_blind, 
                 initial_stack=args.initial_stack)
    
    ui = PokerTableUI(env, human_player=args.human_player)
    ui.start_game(args.n_players)


def train(args):
    """Train a DQN agent."""
    if not has_torch:
        print("PyTorch is required for training. Please install it and try again.")
        return
        
    env = PokerEnv(n_players=args.n_players, 
                 small_blind=args.small_blind, 
                 big_blind=args.big_blind, 
                 initial_stack=args.initial_stack)
    
    # Initialize agents
    agents = []
    for i in range(args.n_players):
        if i == 0:  # DQN agent for first player
            agent = DQNAgent(state_size=13, action_size=10, player_id=i)
        else:  # Random agents for other players
            agent = RandomAgent(i)
        agents.append(agent)
        
    # Training loop
    for episode in range(args.episodes):
        obs = env.reset()
        done = False
        states = {}
        actions = {}
        
        while True:
            if env.current_player != None and not done:
                current_player = env.current_player
                
                if current_player == 0:  # DQN agent's turn
                    # Get state
                    player_obs = obs[current_player]
                    if "legal_actions" in player_obs:
                        state = agents[0]._process_state(player_obs)
                        
                        # Choose action
                        action = agents[0].act(obs)
                        
                        # Remember state and action
                        states[current_player] = state
                        actions[current_player] = action
                        
                        # Take step
                        next_obs, reward, done, _ = env.step(action)
                        
                        # Remember experience
                        next_state = agents[0]._process_state(next_obs[current_player])
                        agents[0].remember(state, action, reward, next_state, done)
                        
                        obs = next_obs
                else:  # Random agent's turn
                    action = agents[current_player].act(obs)
                    
                    if action is not None:
                        obs, _, done, _ = env.step(action)
            
            else:
                env.table._advance_game()
                if env.table.hand_complete == True:
                    env.calculate_final_rewards()
                    agents[0].remember(state, action, reward, state, done)
                    env.reset()
                    break
        
        # Train the model
        if episode % 10 == 0:
            agents[0].replay(args.batch_size)
            
        # Update target network
        if episode % args.target_update == 0:
            agents[0].update_target_model()
            
        # Print progress
        if episode % 100 == 0:
            print(f"Episode {episode}/{args.episodes}, Epsilon: {agents[0].epsilon:.4f}")
    
    # Save the trained model
    if not os.path.exists(os.path.dirname(args.save_path)):
        os.makedirs(os.path.dirname(args.save_path))
    torch.save(agents[0].model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")


def evaluate(args):
    """Evaluate a trained agent."""
    if not has_torch:
        print("PyTorch is required for evaluation. Please install it and try again.")
        return
        
    env = PokerEnv(n_players=args.n_players, 
                 small_blind=args.small_blind, 
                 big_blind=args.big_blind, 
                 initial_stack=args.initial_stack)
    
    # Initialize agents
    agents = []
    for i in range(args.n_players):
        if i == 0:  # Load trained DQN agent
            agent = DQNAgent(state_size=13, action_size=10, player_id=i)
            if os.path.exists(args.save_path):
                agent.model.load_state_dict(torch.load(args.save_path))
                agent.epsilon = 0.0  # No exploration during evaluation
            else:
                print(f"Model not found at {args.save_path}, using untrained agent")
        else:  # Random agents for other players
            agent = RandomAgent(i)
        agents.append(agent)
    
    # Evaluation loop
    total_reward = 0
    num_episodes = 100
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        
        while not done:
            current_player = env.current_player
            action = agents[current_player].act(obs)
            
            if action is not None:
                obs, reward, done, _ = env.step(action)
                
                if current_player == 0:
                    total_reward += reward
        
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes}")
    
    print(f"Average reward over {num_episodes} episodes: {total_reward / num_episodes:.2f}")


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
