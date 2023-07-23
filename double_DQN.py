import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from attention import Attention

'''Discrete Q network with social attention'''
class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, device, ego_dim = 5, oppo_dim = 5):
        super(QValueNet, self).__init__()
        self.attn = Attention(ego_dim, oppo_dim).to(device)
        layer1 = self.attn.embed_dim
        layer_dim = [layer1,] + hidden_dim
        self.fc = [torch.nn.Linear(layer_dim[i], layer_dim[i+1]).to(device) for i in range(len(hidden_dim))]
        self.fc_out = torch.nn.Linear(hidden_dim[-1], action_dim).to(device)

    def forward(self, x):
        if len(x.shape) > 2:
            x = self.attn(x[:,0,:], x[:,:,:])
        else:
            x = self.attn(x[0], x[:]) # the first line is always the ego vehicle

        for layer in self.fc:
            x = F.relu(layer(x))

        return self.fc_out(x)
    

'''double DQN to ameliorate overestimation of Q value'''
class DQN(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
        super(DQN, self).__init__()
        self.q_net = QValueNet(state_dim, hidden_dim, action_dim, device).to(device)
        self.target_q_net = QValueNet(state_dim, hidden_dim, action_dim, device).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.scheduler = lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=100)
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device

    def take_action(self, state):
        if self.training and np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(np.array([state]), dtype = torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
            if not self.training:
                print(self.q_net(state))
        return action 
    
    def max_q_value(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        return self.q_net(state).max().item()
    
    def eps_decay(self):
        self.epsilon = self.epsilon * 0.9

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        max_action = self.q_net(next_states).max(1)[1].view(-1, 1) # [1] return indices
        max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones) # TD error
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets)) 
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        self.count += 1

'''SAC over discrete action space, not working, just for my own reference :('''
'''
class SAC(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, alpha_lr, 
                 target_entropy, tau, gamma, device):
        super(SAC, self).__init__()
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, device).to(device)
        self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim, device).to(device)
        self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim, device).to(device)
        self.target_critic_1 = QValueNet(state_dim, hidden_dim, action_dim, device).to(device)
        self.target_critic_2 = QValueNet(state_dim, hidden_dim, action_dim, device).to(device)
        # Synchronize weights of original critic networks and corresponding target critic networks
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        # uuse log of alpha for more stabilized training
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # alpha can be optimized
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = Categorical(probs)
        action = action_dist.sample()
        print(action.item())
        return action.item()

    def calc_target(self, rewards, next_states, dones):  # compute target Q value
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value), dim=-1, keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)

        # Update the two Q networks
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = torch.mean(F.mse_loss(self.critic_1(states).gather(1, actions), td_target.detach()))
        critic_2_loss = torch.mean(F.mse_loss(self.critic_2(states).gather(1, actions), td_target.detach()))
        # exit()
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Update the policy networks
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value), dim=-1, keepdim=True)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the alpha value
        print("entropy: ", entropy[-1])
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
'''


if __name__ == '__main__':
    import gymnasium as gym
    from matplotlib import pyplot as plt
    import numpy as np
    import random
    import os
    import rl_utils

    # Initiate environment
    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
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
    env = gym.make('highway-fast-v0') # , render_mode='rgb_array'
    env.configure(config)
    env.config["vehicles_density"] = 1.3

    state_dim = 5
    action_dim = 5
    random.seed(53)
    np.random.seed(53)
    torch.manual_seed(53)

    lr = 1e-3
    num_episodes = 300 # 3000 episodes in total
    hidden_dim = [256, 256]
    gamma = 0.9
    epsilon = 0.9
    target_update = 60
    buffer_size = 30000
    minimal_size = 150
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
    return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)

    folder_path = "./results/result11/"

    os.makedirs(folder_path, exist_ok=True)

    torch.save(agent.state_dict(), folder_path + "dqn_attn.pt")

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Double DQN on {}'.format('highway-fast-v0'))
    plt.savefig(folder_path + "returns.png")

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Double DQN on {}'.format('highway-fast-v0'))
    plt.savefig(folder_path + "moving_avg.png")

    print("Training completed!!!")
