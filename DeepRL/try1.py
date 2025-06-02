import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import matplotlib.pyplot as plt
import time

# Policy Network: Simple feed-forward neural network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=2):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)  # Softmax to get probabilities

    def get_action(self, state):
        state = torch.FloatTensor(state)  # Convert state to tensor
        probs = self.forward(state)
        action = torch.multinomial(probs, 1).item()  # Sample action from the probability distribution
        log_prob = torch.log(probs[action])  # Log probability of chosen action
        return action, log_prob, probs

# REINFORCE Algorithm with Baseline and Entropy Regularization
def reinforce(env, policy, episodes=500, max_steps=500, gamma=0.99, entropy_coef=0.01):
    reward_history = []
    
    for episode in range(episodes):
        states, log_probs, rewards, entropies = [], [], [], []
        episode_reward = 0
        
        # Collect trajectories
        state, _ = env.reset()  # Now reset returns a tuple (state, info)
        state = np.array(state)  # Ensure state is a numpy array
        for _ in range(max_steps):
            action, log_prob, probs = policy.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
            states.append(state)
            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(-(probs * torch.log(probs)).sum())  # Entropy term
            
            state = np.array(next_state)  # Ensure state is a numpy array
            episode_reward += reward
            
            if done:
                break
            
            # Render the environment to show the GUI
            env.render()  # Display GUI during training
            time.sleep(0.01)  # Delay to allow GUI to render
        
        # Compute returns (discounted rewards)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # Normalize returns
        
        # Update policy
        policy.optimizer.zero_grad()
        loss = -torch.sum(torch.stack(log_probs) * returns) - entropy_coef * torch.stack(entropies).mean()
        loss.backward()
        policy.optimizer.step()
        
        reward_history.append(episode_reward)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {episode_reward}")
    
    return reward_history

# Main Training Function
def main():
    env = gym.make('CartPole-v1')
    policy = PolicyNetwork()
    rewards = reinforce(env, policy, episodes=500)

    # Plot Results
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("REINFORCE on CartPole-v1")
    plt.show()

if __name__ == "__main__":
    main()
