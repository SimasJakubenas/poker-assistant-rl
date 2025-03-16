import os

# Import the poker environment and UI
from rl_model.rl_environment import PokerEnv
from rl_model.agent import DQNAgent

try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False
    print("PyTorch not found. RL training will not be available.")


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
        partial_path = os.path.join(f"outputs/models/rl/{args.save_path}", f"/player_{player}")
        os.makedirs(partial_path, exist_ok=True)
        
    final_path = os.path.join(f"outputs/models/rl/{args.save_path}", f"/final")
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
                save_path = os.path.join(f"outputs/models/rl/{args.save_path}", f"/player_{player_id}/dqn_player_{player_id}_ep{episode/args.save_freq}.pt")
                agent.save(save_path, player_total)
    
    # Save final models
    for player_id, agent in enumerate(agents):
        player_total = env.table.players[player_id].balance + env.table.players[player_id].balance - args.initial_stack
        save_path = os.path.join(f"outputs/models/rl/{args.save_path}", f"/final/dqn_player_{player_id}_final.pt")
        agent.save(save_path, player_total)
        
    env.reset()