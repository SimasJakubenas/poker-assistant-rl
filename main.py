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
    parser.add_argument("--mode", choices=["ui", "train", "evaluate"], default="evaluate",
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
        parser.add_argument("--target_update", type=int, default=10,
                          help="Episodes between target network updates")
        parser.add_argument("--save_path", type=str, default="outputs/models/rl",
                          help="Path to save the trained model")
        parser.add_argument("--save_freq", type=int, default=1000,
                          help="Path to save the trained model")
    
    return parser.parse_args()

def load_pretrained_agent_for_rl(state_size, action_size, player_id):
    """
    Load a pretrained agent and prepare it for reinforcement learning.
    
    Args:
        pretrained_model_path: Path to the saved pretrained model
        state_size: Size of the state vector
        action_size: Size of the action space
        player_id: ID of the player this agent controls
        
    Returns:
        Loaded DQNAgent ready for RL training
    """
    # Create a new agent
    agent = DQNAgent(state_size, action_size, player_id)
    if os.path.exists('outputs/models/sl/pretrained_hero_agent.pt'):
        try:
            # Load saved model weights and parameters
            checkpoint = torch.load('outputs/models/sl/pretrained_hero_agent.pt')
            agent.model.load_state_dict(checkpoint['model_state_dict'])
            agent.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            agent.epsilon = checkpoint['epsilon']
            agent.total = checkpoint['total']
            # Optionally load epsilon if saved
            if 'epsilon' in checkpoint:
                agent.epsilon = checkpoint['epsilon']
            
            print(f"Successfully loaded pretrained model from 'outputs/models/sl/pretrained_hero_agent.pt'")
            
            # Initialize target model to match main model
            agent.update_target_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Initializing agent with random weights")
    else:
        print(f"Pretrained model file 'outputs/models/sl/pretrained_hero_agent.pt' not found")
        print("Initializing agent with random weights")

    # Configure agent for RL training
    # You may want to reset some parameters for RL training
    agent.epsilon = 1.0  # Start with high exploration
    agent.epsilon_decay = 0.995  # Adjust decay rate as needed

    return agent

def run_ui(args):
    """Run the poker UI."""
    env = PokerEnv(n_players=args.n_players, 
                 small_blind=args.small_blind, 
                 big_blind=args.big_blind, 
                 initial_stack=args.initial_stack)
    
    ui = PokerTableUI(env, human_player=args.human_player)
    ui.start_game(args.n_players)


def train(args):
    """Train multiple DQN agents simultaneously."""
    if not has_torch:
        print("PyTorch is required for training. Please install it and try again.")
        return
        
    env = PokerEnv(n_players=args.n_players, 
                 small_blind=args.small_blind, 
                 big_blind=args.big_blind, 
                 initial_stack=args.initial_stack)
    
    # Create directory for saving models
    for player in range(args.n_players):
        partial_path = os.path.join(args.save_path, f"player_{player}")
        os.makedirs(partial_path, exist_ok=True)
        
    final_path = os.path.join(args.save_path, f"final")
    os.makedirs(final_path, exist_ok=True)
    
    # Initialize agents
    agents = [load_pretrained_agent_for_rl(state_size=16, action_size=10, player_id=i) for i in range(env.n_players)]
    
    # For keeping track of cumulative rewards
    all_rewards = [[] for _ in range(env.n_players)]
    episode_rewards = [0 for _ in range(env.n_players)]
    
    # Training loop
    for episode in range(args.episodes):
        obs = env.reset()
        done = False
        states = {}
        actions = {}
        
        # Reset episode rewards
        episode_rewards = [0 for _ in range(env.n_players)]
        
        while True:
            if env.current_player != None and not done:
                current_player = env.current_player
                
                # Get state
                player_obs = obs[current_player]
                if "legal_actions" in player_obs:
                    state = agents[current_player]._process_state(player_obs)
                    action = agents[current_player].act(obs)
                    
                    # Remember state and action
                    states[current_player] = state
                    actions[current_player] = action
                    
                    # Take step
                    next_obs, reward, done, prev_player = env.step(action)
                    
                    # if valid action
                    if reward <= 0:
                        
                        # Remember experience
                        if env.terminal != True and current_player in states:
                            next_state = agents[current_player]._process_state(next_obs[current_player])
                            agents[current_player].remember_action(
                                states[current_player],
                                actions[current_player],
                                reward,
                                next_state,
                                False
                            )
            
            else:
                env.table._advance_game()
                if env.table.hand_complete == True:
                    env.calculate_final_rewards(prev_player)
                    for player_id in range(env.n_players):
                        # Save rewards
                        episode_rewards[player_id] = env.rewards[player_id]
                        if player_id in states:
                            # Update with final rewards
                            agents[player_id].remember_action(
                                states[player_id],
                                actions[player_id],
                                env.rewards[player_id],  # Final rewards from the environment
                                states[player_id],  # Dummy next state
                                True
                            )

                    break
            
            obs = next_obs
        
        # After episode ends, train all agents
        for player_id in range(env.n_players):
            # Hand is complete - store final result
            agents[player_id].complete_hand(env.rewards[player_id])
            
            agents[player_id].replay(args.batch_size)
           
            # Update target networks periodically
            if episode % args.target_update == 0:
                agents[player_id].update_target_model()
        
        # Record rewards
        for player_id in range(env.n_players):
            all_rewards[player_id].append(episode_rewards[player_id])

        # Print progress
        if episode % 100 == 0 and episode != 0:
            print(f"Episode {episode}/{args.episodes}")
            avg_rewards = [sum(rews[-100:]) / min(len(rews), 100) for rews in all_rewards]
            print(f"Average rewards (last 100): {avg_rewards}")
        # Save models periodically
        if episode % args.save_freq == 0 and episode != 0:
            for player_id, agent in enumerate(agents):
                player_total = env.table.players[player_id].balance + env.table.players[player_id].stack - args.initial_stack
                save_path = os.path.join(args.save_path, f"player_{player_id}/dqn_player_{player_id}_ep{episode/args.save_freq}.pt")
                agent.save(save_path, player_total)
    
    # Save final models
    for player_id, agent in enumerate(agents):
        player_total = env.table.players[player_id].balance + env.table.players[player_id].balance - args.initial_stack
        save_path = os.path.join(args.save_path, f"final/dqn_player_{player_id}_final.pt")
        agent.save(save_path, player_total)
        
    env.reset()

def evaluate(args):
    """Evaluate the performance of trained agents."""
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
        agent = DQNAgent(state_size=16, action_size=10, player_id=i)
        if os.path.exists(args.save_path):
            load_path = os.path.join(args.save_path, f"final/dqn_player_{i}_final.pt")
            agent.load(load_path)
            agent.epsilon = 0.0  # No exploration during evaluation
        else:
            print(f"Model not found at {args.save_path}")

        agents.append(agent)
    
    # Evaluation loop
    total_rewards = [0.0 for _ in range(len(agents))]
    num_episodes = 100
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        
        while not done:
            current_player = env.current_player
            action = agents[current_player].act(obs)
            
            if action is not None:
                obs, reward, done, _ = env.step(action)
                
                # Save reward for the current player
                if not done:
                    total_rewards[current_player] += reward
        
        # Add final rewards at the end of the hand
        for player_id in range(len(agents)):
            total_rewards[player_id] += env.rewards[player_id]
        
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes}")
    
    # Calculate average rewards
    avg_rewards = [r / num_episodes for r in total_rewards]
    print(f"Average rewards after training: {avg_rewards}")


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
