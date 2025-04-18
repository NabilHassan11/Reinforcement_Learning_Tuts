import numpy as np
import matplotlib.pyplot as plt

# ======================
# OPTIMIZED ENVIRONMENT
# ======================
class FastGridworld:
    def __init__(self):
        self.size = 4
        self.goal = (3, 3)
        self.walls = {(1,1), (1,2), (2,1)}
        self.actions = ['up', 'down', 'left', 'right']
        self.max_steps = 50  # Prevent infinite episodes
        
    def reset(self):
        self.state = (0, 0)
        self.step_count = 0
        return self.state
    
    def step(self, action):
        self.step_count += 1
        row, col = self.state
        
        # Calculate next state
        if action == 'up': row = max(0, row-1)
        elif action == 'down': row = min(3, row+1)
        elif action == 'left': col = max(0, col-1)
        elif action == 'right': col = min(3, col+1)
        
        next_state = (row, col)
        
        # Check for walls
        if next_state in self.walls:
            return self.state, -1, False
        
        # Update state and get rewards
        self.state = next_state
        
        if next_state == self.goal:
            return next_state, 10, True
        if self.step_count >= self.max_steps:
            return next_state, -1, True
            
        return next_state, -0.1, False

# ======================
# EFFICIENT FEATURES
# ======================
def state_to_features(state):
    """Convert grid position to one-hot feature vector"""
    features = np.zeros(16)  # 4x4 grid = 16 states
    idx = state[0] * 4 + state[1]
    features[idx] = 1
    return features

# ======================
# SMART POLICY
# ======================
def smart_policy(state, env):
    """80% random, 20% goal-directed"""
    if np.random.random() < 0.8:
        return np.random.choice(env.actions)
    
    # Move toward goal
    dr = env.goal[0] - state[0]
    dc = env.goal[1] - state[1]
    
    if abs(dr) > abs(dc):
        return 'down' if dr > 0 else 'up'
    else:
        return 'right' if dc > 0 else 'left'

# =================================
# INCREMENTAL LEARNING MONTE CARLO
# =================================
def optimized_mc(env, num_episodes, alpha=0.1, gamma=0.9):
    w = np.zeros(16)  # Match feature dimension
    value_history = []

    for _ in range(num_episodes):
        state = env.reset()
        episode = []
        done = False

        # Generate episode
        while not done:
            action = smart_policy(state, env)
            next_state, reward, done = env.step(action)
            episode.append((state, reward))
            state = next_state

        # Calculate Monte Carlo returns
        G = 0
        for t in reversed(range(len(episode))):
            state_t, reward_t = episode[t]
            G = reward_t + gamma * G

            # Compute feature and prediction
            x_t = state_to_features(state_t)
            V_hat = x_t.dot(w)

            # Per-step weight update (professor's style)
            w += alpha * (G - V_hat) * x_t

        # Track starting state value
        value_history.append(state_to_features((0,0)).dot(w))

    return w, value_history


# ======================
# VISUALIZATION
# ======================
def plot_results(value_history, w):
    plt.figure(figsize=(12, 4))
    
    # Learning curve
    plt.subplot(1, 2, 1)
    plt.plot(value_history)
    plt.title("Learning Progress")
    plt.xlabel("Episode")
    plt.ylabel("Start State Value")
    
    # Value heatmap
    plt.subplot(1, 2, 2)
    grid = np.zeros((4,4))
    for row in range(4):
        for col in range(4):
            grid[row,col] = state_to_features((row,col)).dot(w)
            
    plt.imshow(grid, cmap='viridis', origin='upper')
    plt.colorbar()
    plt.title("Learned Value Function")
    
    plt.tight_layout()
    plt.show()

# ======================
# RUN OPTIMIZED VERSION
# ======================
env = FastGridworld()
w, history = optimized_mc(env, num_episodes=10000)
plot_results(history, w)

print("Final weights:", w)