import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from attention import Attention

'''Actor with social attention'''
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, device, ego_dim = 4, oppo_dim = 4):
        super(PolicyNet, self).__init__()
        self.attn = Attention(ego_dim, oppo_dim).to(device)
        layer1 = self.attn.embed_dim
        layer_dim = [layer1,] + hidden_dim
        self.fc = [torch.nn.Linear(layer_dim[i], layer_dim[i+1]).to(device) for i in range(len(hidden_dim))]
        self.fc_mu = torch.nn.Linear(hidden_dim[-1], action_dim).to(device)
        self.fc_std = torch.nn.Linear(hidden_dim[-1], action_dim).to(device)
        self.action_bound = action_bound

    def forward(self, x):
        if len(x.shape) > 2:
            x = self.attn(x[:,0,:], x[:,1:,:])
        else:
            x = self.attn(x[0], x[1:]) # the first line is always the ego vehicle

        for layer in self.fc:
            x = F.relu(layer(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # reparameterization trick
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        # Compute the log probability density of the tanh_normal distribution
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        # sum the log_prob as we assume that action dimensions are independent
        log_prob = torch.sum(log_prob, axis = -1).unsqueeze(-1)
        action = action * self.action_bound
        return action, log_prob


'''Critic with social attention'''
class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, device, ego_dim = 4, oppo_dim = 4):
        super(QValueNet, self).__init__()
        self.attn = Attention(ego_dim, oppo_dim).to(device)
        layer1 = self.attn.embed_dim
        layer_dim = [layer1 + action_dim,] + hidden_dim
        self.fc = [torch.nn.Linear(layer_dim[i], layer_dim[i+1]).to(device) for i in range(len(hidden_dim))]
        self.fc_out = torch.nn.Linear(hidden_dim[-1], 1).to(device)

    def forward(self, x, a):
        if len(x.shape) > 2:
            x = self.attn(x[:,0,:], x[:,1:,:])
            x = torch.cat([x, a], dim = 1)
        else:
            x = self.attn(x[0], x[1:]) # the first line is always the ego vehicle
            x = torch.cat([x, a], dim = 0)

        for layer in self.fc:
            x = F.relu(layer(x))
        return self.fc_out(x)
    

class SAC(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device):
        super(SAC, self).__init__()
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound, device).to(device)
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
        action = self.actor(state)[0]
        return action.detach().cpu().numpy()

    def calc_target(self, rewards, next_states, dones):  # compute target Q value
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)

        # Update the two Q networks
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = torch.mean(F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        # exit()
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Update the policy networks
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the alpha value
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

