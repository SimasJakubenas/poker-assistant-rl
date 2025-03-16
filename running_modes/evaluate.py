import os

# Import the poker environment and UI
from rl_model.rl_environment import PokerEnv
from rl_model.agent import DQNAgent
from rl_model.random_agent import RandomAgent


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
        if i == 0:
            if os.path.exists(f"outputs/models/rl/{args.save_path}") and i == 0:
                load_path = os.path.join(f"outputs/models/rl/{args.save_path}", f"/final/dqn_player_{i}_final.pt")
                agent.load(load_path)
                agent.epsilon = 0.0  # No exploration during evaluation
                    
            else:
                print(f"RL model not found at outputs/models/rl/{args.save_path} loading sl model")
                if os.path.exists(f"outputs/models/sl"):
                    load_path = os.path.join(f"outputs/models/sl", f"/pretrained_hero_agent.pt")
                    agent.load(load_path)
                    agent.epsilon = 0.0  # No exploration during evaluation
                else:
                    print(f"SL model not found at outputs/models/sl loading random agent")
                    agent = RandomAgent(i)
                    
        else:
            if os.path.exists(f"outputs/models/sl"):
                load_path = os.path.join(f"outputs/models/sl", f"/pretrained_hero_agent.pt")
                agent.load(load_path)
                agent.epsilon = 0.0  # No exploration during evaluation
            else:
                print(f"SL model not found at outputs/models/sl loading random agent")
                agent = RandomAgent(i)

        agents.append(agent)
    
    # Evaluation loop
    total_rewards = [0.0 for _ in range(len(agents))]
    num_episodes = 1000
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        
        while True:
            if env.current_player != None and not done:
                current_player = env.current_player
                action = agents[current_player].act(obs)
                
                if action is not None:
                    obs, _, done, prev_player = env.step(action)
                    
            else:
                # Add final rewards at the end of the hand
                env.table._advance_game()
                if env.table.hand_complete == True:
                    env.calculate_final_rewards(prev_player)

                    for player_id in range(len(agents)):
                        total_rewards[player_id] += env.rewards[player_id]
                    
                    break
        
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes}")
    
    # Calculate average rewards
    avg_rewards = [r / num_episodes for r in total_rewards]
    print(f"Average rewards after training: {avg_rewards}")