import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class StochasticWalkQLearning:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.1):
        # Environment parameters (same as SARSA)
        self.n_states = 1000
        self.terminal_states = {0: -100, 999: 100}
        self.bucket_size = 100
        self.num_buckets = 10
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.w_left = np.full(self.num_buckets, 0.1)  # Initial bias
        self.w_right = np.full(self.num_buckets, 0.1)  
        self.value_history = []

    # --- Common methods with SARSA ---
    def bucket_state(self, state): 
        return (state - 1) // self.bucket_size

    def get_q_value(self, state, action):
        bucket = self.bucket_state(state)
        return self.w_left[bucket] if action == 'left' else self.w_right[bucket]

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(['left', 'right'])
        left_q = self.get_q_value(state, 'left')
        right_q = self.get_q_value(state, 'right')
        return 'left' if left_q > right_q else 'right'

    def generate_episode(self, max_steps=1000):
        state = 500
        episode = []
        steps = 0

        while state not in self.terminal_states and steps < max_steps:
            action = self.choose_action(state)

            if action == 'left':
                lower = max(0, state - 100)
                upper = state
            else:
                lower = state
                upper = min(999, state + 100)

            next_state = np.random.randint(lower, upper + 1)
            reward = self.terminal_states.get(next_state, -0.1)

            episode.append((state, action, reward, next_state))
            state = next_state
            steps += 1

        return episode


    # --- Key Q-Learning Changes ---
    def update_weights(self, episode):
        for transition in episode:
            s, a, r, s_next = transition
            
            if s_next in self.terminal_states:
                q_next = 0
            else:
                q_next_left = self.get_q_value(s_next, 'left')
                q_next_right = self.get_q_value(s_next, 'right')
                q_next = max(q_next_left, q_next_right)
                
            q_current = self.get_q_value(s, a)
            td_error = r + self.gamma * q_next - q_current
            
            bucket = self.bucket_state(s)
            if a == 'left':
                self.w_left[bucket] += self.alpha * td_error
            else:
                self.w_right[bucket] += self.alpha * td_error  # Fixed line

    # --- Same training and visualization ---
    def train(self, num_episodes=1000):
        self.value_history = []
        for _ in tqdm(range(num_episodes)):
            episode = self.generate_episode()
            self.update_weights(episode)
            self.epsilon *= 0.995
            self.value_history.append(self.get_q_value(500, 'right'))
            
    def print_policy(self):
        print("Bucket | Preferred Action")
        print("-------|-----------------")
        for bucket in range(self.num_buckets):
            action = "left" if self.w_left[bucket] > self.w_right[bucket] else "right"
            print(f"{bucket:6} | {action}")

    def plot_results(self):
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.value_history)
        plt.title("Q-value at Center State (500)")
        plt.xlabel("Episode")
        plt.ylabel("Value")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        action_preference = self.w_right - self.w_left
        plt.bar(range(10), action_preference)
        plt.axhline(0, color='k', linestyle='--')
        plt.title("Action Preference (Right - Left)")
        plt.xlabel("State Bucket")
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    agent = StochasticWalkQLearning(alpha=0.1, gamma=0.99, epsilon=0.1)
    agent.train(10000)
    agent.plot_results()
    agent.print_policy()