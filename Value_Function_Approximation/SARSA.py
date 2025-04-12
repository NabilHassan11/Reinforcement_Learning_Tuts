import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt  # Fixed import

class StochasticWalkSARSA:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.n_states = 1000
        self.terminal_states = {0: -100, 999: 100}
        self.bucket_size = 100
        self.num_buckets = 10
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.w_left = np.zeros(self.num_buckets)
        self.w_right = np.zeros(self.num_buckets)
        self.history_left = []
        self.history_right = []

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

    def generate_episode(self):
        state = 500
        episode = []
        while state not in self.terminal_states:
            action = self.choose_action(state)
            if np.random.random() < 0.5:
                next_state = max(0, state - 100) if action == 'left' else min(999, state + 100)
            else:
                next_state = min(999, state + 100) if action == 'left' else max(0, state - 100)
            reward = self.terminal_states.get(next_state, -1)
            episode.append((state, action, reward, next_state))
            state = next_state
        return episode

    def update_weights(self, episode):
        for t in range(len(episode)-1):
            s, a, r, s_next = episode[t]
            a_next = episode[t+1][1] if t+1 < len(episode) else None
            q = self.get_q_value(s, a)
            q_next = self.get_q_value(s_next, a_next) if a_next else 0
            td_error = r + self.gamma * q_next - q
            bucket = self.bucket_state(s)
            if a == 'left':
                self.w_left[bucket] += self.alpha * td_error
            else:
                self.w_right[bucket] += self.alpha * td_error

    def train(self, num_episodes):
        for _ in tqdm(range(num_episodes)):
            episode = self.generate_episode()
            self.update_weights(episode)
            self.epsilon *= 0.995
            # Track weights for plotting
            self.history_left.append(self.w_left.copy())
            self.history_right.append(self.w_right.copy())

    def print_policy(self):
        print("Bucket | Preferred Action")
        print("-------|-----------------")
        for bucket in range(self.num_buckets):
            action = "left" if self.w_left[bucket] > self.w_right[bucket] else "right"
            print(f"{bucket:6} | {action}")

    def plot_weights(self):
        history_left = np.array(self.history_left)
        history_right = np.array(self.history_right)
        plt.figure(figsize=(12, 6))
        for bucket in range(self.num_buckets):
            plt.plot(history_left[:, bucket], label=f"Left Bucket {bucket}", linestyle='--')
            plt.plot(history_right[:, bucket], label=f"Right Bucket {bucket}")
        plt.title("Q-Value Weights Over Time")
        plt.xlabel("Episodes")
        plt.ylabel("Weight Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_final_policy(self):
        policy = [0 if self.w_right[bucket] > self.w_left[bucket] else 1 for bucket in range(self.num_buckets)]
        actions = ['right' if a == 1 else 'left' for a in policy]
        plt.figure(figsize=(10, 5))
        bars = plt.bar(range(self.num_buckets), policy, tick_label=[f"B{b}" for b in range(self.num_buckets)])
        plt.title("Final Policy: 0 = Left, 1 = Right")
        plt.xlabel("Bucket")
        plt.ylabel("Preferred Action")

        # Label each bar with the actual action
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, actions[i],
                    ha='center', va='bottom', fontsize=10)

        plt.ylim(0, 1.5)
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    agent = StochasticWalkSARSA(alpha=0.1, gamma=0.95, epsilon=0.2)
    agent.train(20000)
    agent.print_policy()
    agent.plot_weights()
    agent.plot_final_policy()
