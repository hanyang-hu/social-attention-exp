import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
import torch
import random
from SAC import SAC
import rl_utils
import os

# Initiate environment
env = gym.make('highway-fast-v0')

# Observation & action configuration
env.configure({
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["x", "y", "vx", "vy"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
        "grid_step": [5, 5],
        "absolute": True
    }, 
    "action": {
        "type": "ContinuousAction" # 2-D continuous action
    }
})

state_dim = 4
action_dim = 2
action_bound = 1.0
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

actor_lr = 5e-4
critic_lr = 5e-4
alpha_lr = 5e-4
num_episodes = 100
hidden_dim = [128, 64, 64, 32]
gamma = 0.8
tau = 0.005 # hyp for soft update
buffer_size = 15000
minimal_size = 200
batch_size = 64
target_entropy = -2.0 # 2-D action space maximal entropy is 2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

replay_buffer = rl_utils.ReplayBuffer(buffer_size)
agent = SAC(state_dim, hidden_dim, action_dim, action_bound, actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device)

return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)

folder_path = "./results/result01/"

os.makedirs(folder_path, exist_ok=True)

torch.save(agent.state_dict(), folder_path + "sac_attn.pt")

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('SAC on {}'.format('highway-fast-v0'))
plt.savefig(folder_path + "returns.png")

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('SAC on {}'.format('highway-fast-v0'))
plt.savefig(folder_path + "moving_avg.png")

print("Training completed!!!")
