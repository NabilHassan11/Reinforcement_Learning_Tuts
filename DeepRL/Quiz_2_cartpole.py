import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def cartpole_dynamics(state, F, M=1.0, m=0.1, l=0.5, g=9.81):
    x, x_dot, theta, theta_dot = state
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    total_mass = M + m
    polemass_length = m * l

    temp = (F + polemass_length * theta_dot**2 * sin_theta) / total_mass
    theta_acc = (g * sin_theta - cos_theta * temp) / (l * ((4.0 / 3.0) - m * cos_theta**2 / total_mass))
    x_acc = temp - polemass_length * theta_acc * cos_theta / total_mass

    return np.array([x_dot, x_acc, theta_dot, theta_acc])

# Enhanced Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)
    
    def get_action(self, state):
        state = torch.FloatTensor(state)
        probs = self.forward(state)
        action = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs.squeeze(0)[action])
        return action, log_prob, probs

# Improved REINFORCE with baseline and entropy
def reinforce(env, policy, episodes=500, batch_size=10, max_steps=500, gamma=0.99, entropy_coef=0.01):
    reward_history = []
    
    for episode in range(episodes):
        states, log_probs, rewards, entropies = [], [], [], []
        episode_reward = 0
        
        # Collect trajectories
        state = env.reset()
        for _ in range(max_steps):
            action, log_prob, probs = policy.get_action(state)
            next_state, reward, done = env.step(action)
            
            states.append(state)
            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(-(probs * torch.log(probs)).sum())  # Entropy
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Compute returns and advantages
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Update policy
        policy.optimizer.zero_grad()
        loss = -torch.sum(torch.stack(log_probs) * returns) - entropy_coef * torch.stack(entropies).mean()
        loss.backward()
        policy.optimizer.step()
        
        reward_history.append(episode_reward)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {episode_reward}")
    
    return reward_history

# Modified Environment with random initialization
class CartPoleEnv:
    def __init__(self):
        self.dt = 0.05
        self.theta_threshold = 0.2095  # 12 degrees
        self.x_threshold = 2.4
        
    def reset(self):
        # Random initial angle (±0.05 rad)
        self.state = np.array([0.0, 0.0, np.random.uniform(-0.05, 0.05), 0.0])  
        return self.state.copy()
    
    def step(self, action):
        F = 10.0 if action == 1 else -10.0
        derivatives = cartpole_dynamics(self.state, F)
        self.state = self.state + derivatives * self.dt
        x, theta = self.state[0], self.state[2]
        
        done = (abs(x) > self.x_threshold) or (abs(theta) > self.theta_threshold)
        reward = 1.0 if not done else 0.0
        
        return self.state.copy(), reward, done

# Training
env = CartPoleEnv()
policy = PolicyNetwork()
rewards = reinforce(env, policy, episodes=500)

# Plot results
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Improved REINFORCE Training")
plt.show()