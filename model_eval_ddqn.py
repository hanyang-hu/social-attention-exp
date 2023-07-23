import torch
import random
import numpy as np
from double_DQN import DQN

state_dim = 5
action_dim = 5
random.seed(53)
np.random.seed(53)
torch.manual_seed(53)

lr = 1e-3
num_episodes = 300 # 3000 episodes in total
hidden_dim = [256, 256]
gamma = 0.99
epsilon = 0.9
target_update = 30
buffer_size = 30000
minimal_size = 150
batch_size = 128
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)

folder_path = "./results/result10/"

agent.load_state_dict(torch.load(folder_path + "dqn_attn.pt"))
agent.eval()

import gymnasium as gym

env = gym.make('highway-fast-v0', render_mode = 'rgb_array')
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 40,
        "features": ["presence", "x", "y", "vx", "vy"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "absolute": False
    }
}
env.config["duration"] = 300
env.config["vehicles_density"] = 1.3
env.configure(config)

for _ in range(10):
    obs, info = env.reset()
    done = truncated = False
    input("Press Enter to continue...")
    while not (done or truncated):
        action = agent.take_action(obs)
        print(action)
        obs, reward, done, truncated, _ = env.step(action)
        env.render()
