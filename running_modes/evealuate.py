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