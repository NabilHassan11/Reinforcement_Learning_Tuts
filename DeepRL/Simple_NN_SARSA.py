import numpy as np

# np.random.seed(42)  # For reproducibility (optional)

### One-hot encode states ###
def one_hot(state, n_states=3):
    return np.eye(n_states)[state].astype(np.float32)

# ======================
# Neural Network Class
# ======================
class QNetwork:
    def __init__(self, input_size=3, hidden_size=4, output_size=2):
        scale = 1
        self.W1 = np.random.randn(input_size, hidden_size) * scale * np.sqrt(2 / input_size)
        self.W2 = np.random.randn(hidden_size, output_size) * scale * np.sqrt(2 / hidden_size)
    
    def forward(self, x):
        self.z1 = np.dot(x, self.W1)
        self.a1 = np.maximum(0, self.z1)
        self.q = np.dot(self.a1, self.W2)
        
        if np.array_equal(x, one_hot(2)):
            self.q = np.zeros(2)
        return self.q
        
    def backward(self, x, action, td_error, lr=0.01):
        td_error_vec = np.zeros(2)
        td_error_vec[action] = td_error
        grad_W2 = np.outer(self.a1, td_error_vec)
        
        delta = np.dot(self.W2, td_error_vec) * (self.z1 > 0)
        grad_W1 = np.outer(x, delta)
        
        grad_W2 = np.clip(grad_W2, -1, 1)
        grad_W1 = np.clip(grad_W1, -1, 1)

        self.W2 += lr * grad_W2
        self.W1 += lr * grad_W1

# ======================
# Neural Network Training
# ======================
def train_nn(n_episodes=1000):
    q_net = QNetwork()
    gamma = 0.9
    alpha = 0.01
    epsilon = 0.1

    state = 0
    for _ in range(n_episodes):
        x = one_hot(state)
        q_values = q_net.forward(x)

        if np.random.rand() < epsilon:
            action = np.random.choice(2)
        else:
            action = np.argmax(q_values)
        
        next_state = state + 1 if action == 1 else state - 1
        next_state = np.clip(next_state, 0, 2)
        reward = 1 if next_state == 2 else -0.5

        if next_state == 2:
            target = reward
        else:
            x_next = one_hot(next_state)
            q_next = q_net.forward(x_next)

            if np.random.rand() < epsilon:
                next_action = np.random.choice(2)
            else:
                next_action = np.argmax(q_next)

            target = reward + gamma * q_next[next_action]
        
        td_error = target - q_values[action]
        q_net.backward(x, action, td_error, lr=alpha)

        state = next_state if next_state != 2 else 0

    return q_net

# ======================
# Tabular SARSA Training
# ======================
def train_tabular(n_episodes=1000):
    q_table = np.zeros((3, 2))
    gamma = 0.9
    alpha = 0.01
    epsilon = 0.1

    state = 0
    for _ in range(n_episodes):
        if np.random.rand() < epsilon:
            action = np.random.choice(2)
        else:
            action = np.argmax(q_table[state])
        
        next_state = state + 1 if action == 1 else state - 1
        next_state = np.clip(next_state, 0, 2)
        reward = 1 if next_state == 2 else -0.5

        if next_state == 2:
            target = reward
        else:
            if np.random.rand() < epsilon:
                next_action = np.random.choice(2)
            else:
                next_action = np.argmax(q_table[next_state])
            target = reward + gamma * q_table[next_state][next_action]
        
        q_table[state, action] += alpha * (target - q_table[state, action])
        state = next_state if next_state != 2 else 0

    return q_table

# ======================
# Run and Compare
# ======================
if __name__ == "__main__":
    nn_model = train_nn(n_episodes=5000)
    tabular_model = train_tabular(n_episodes=5000)

    print("\n=== Neural Network Q-values ===")
    for state in range(3):
        nn_q = nn_model.forward(one_hot(state))
        print(f"State {state} - Left: {nn_q[0]:.3f}, Right: {nn_q[1]:.3f}")

    print("\n=== Tabular Q-values ===")
    for state in range(3):
        tabular_q = tabular_model[state]
        print(f"State {state} - Left: {tabular_q[0]:.3f}, Right: {tabular_q[1]:.3f}")

    print("\nVerification:")
    print("Expected policy:")
    print("State 0: Should prefer Right")
    print("State 1: Should prefer Right")
    print("State 2: Terminal state (values should be close to 0)")
