import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class StochasticWalkQLearning:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.n_states = 1000
        self.terminal_states = {0: -100, 999: 100}
        self.bucket_size = 100
        self.num_buckets = 10
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.w_left = np.full(self.num_buckets, 0.1)
        self.w_right = np.full(self.num_buckets, 0.1)
        self.value_history = []

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

    def update_weights(self, episode):
        for s, a, r, s_next in episode:
            if s_next in self.terminal_states:
                q_next = 0
            else:
                q_next = max(self.get_q_value(s_next, 'left'), self.get_q_value(s_next, 'right'))
            q_current = self.get_q_value(s, a)
            td_error = r + self.gamma * q_next - q_current
            bucket = self.bucket_state(s)
            if a == 'left':
                self.w_left[bucket] += self.alpha * td_error
            else:
                self.w_right[bucket] += self.alpha * td_error

    def train(self, num_episodes=10000):
        self.value_history = []
        for _ in tqdm(range(num_episodes)):
            episode = self.generate_episode()
            self.update_weights(episode)
            self.epsilon *= 0.995
            self.value_history.append(self.get_q_value(500, 'right'))

        # Print Q-values of bucket 0 after training
        print(f"Final Q-values of bucket 0: Left={self.w_left[0]:.3f}, Right={self.w_right[0]:.3f}")

    def print_policy(self):
        print("\nBucket | Preferred Action")
        print("-------|-----------------")
        for bucket in range(self.num_buckets):
            action = "left" if self.w_left[bucket] > self.w_right[bucket] else "right"
            print(f"{bucket:6} | {action}")

    def get_policy_array(self):
        return np.array(['left' if self.get_q_value(s, 'left') > self.get_q_value(s, 'right') else 'right'
                         for s in range(self.n_states)])

    def plot_policy_heatmap(self):
        policy = self.get_policy_array()
        heatmap = np.array([0 if a == 'left' else 1 for a in policy])
        plt.figure(figsize=(12, 2))
        plt.imshow(heatmap.reshape(1, -1), cmap='coolwarm', aspect='auto')
        plt.yticks([])
        plt.xticks(np.arange(0, 1000, 100), labels=[f"S{s}" for s in range(0, 1000, 100)])
        plt.title("Final Policy Heatmap (Red = Left, Blue = Right)")
        plt.colorbar(label="Action Preference (0 = Left, 1 = Right)")
        plt.tight_layout()
        plt.show()

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
    agent = StochasticWalkQLearning(alpha=0.1, gamma=0.95, epsilon=0.3)
    agent.train(30000)
    agent.plot_results()
    agent.print_policy()
    agent.plot_policy_heatmap()
