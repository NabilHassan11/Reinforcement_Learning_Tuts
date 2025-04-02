import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ======================
# MAZE CONFIGURATION
# ======================
maze = np.array([
    [2, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 1],
    [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1, 0, 0, 0, 3]
])

# ======================
# SARSA PARAMETERS
# ======================
actions = ['up', 'down', 'left', 'right']
alpha = 0.1       # Learning rate
gamma = 0.9      # Discount factor
epsilon = 0.1     # Exploration rate
episodes = 1000    # Total training episodes
save_interval = 100  # Save Q-table every N episodes

# Initialize Q-table and tracking
q_table = np.zeros((maze.shape[0], maze.shape[1], len(actions)))
snapshots = []
steps_history = []

# Find start and goal positions
start_pos = tuple(np.argwhere(maze == 2)[0])
goal_pos = tuple(np.argwhere(maze == 3)[0])

# ======================
# HELPER FUNCTIONS
# ======================
def get_next_state(state, action):
    row, col = state
    if action == 'up' and row > 0:
        return (row-1, col)
    elif action == 'down' and row < 9:
        return (row+1, col)
    elif action == 'left' and col > 0:
        return (row, col-1)
    elif action == 'right' and col < 9:
        return (row, col+1)
    return state  # Invalid move

def get_reward(next_state):
    if maze[next_state] == 3:
        return 100  # Goal reward
    if maze[next_state] == 1:
        return -10  # Wall penalty
    return -1  # Step penalty

def epsilon_greedy(state, epsilon):
    if np.random.random() < epsilon:
        return np.random.choice(actions)
    return actions[np.argmax(q_table[state[0], state[1]])]

# ======================
# Q-Learning TRAINING LOOP
# ======================
for episode in range(episodes):
    state = start_pos
    action = epsilon_greedy(state, epsilon)
    steps = 0
    episode_reward = 0
    
    while True:
        # Execute action
        next_state = get_next_state(state, action)
        reward = get_reward(next_state)
        next_action = actions[np.argmax(q_table[next_state[0], next_state[1]])]
        
        # Q-Learning update
        current_q = q_table[state[0], state[1], actions.index(action)]
        next_q = q_table[next_state[0], next_state[1], actions.index(next_action)]
        q_table[state[0], state[1], actions.index(action)] += alpha * (
            reward + gamma * next_q - current_q
        )
        
        # Transition to next state
        state, action = next_state, next_action
        steps += 1
        episode_reward += reward
        
        # Termination conditions
        if state == goal_pos or steps > 100:
            steps_history.append(steps)
            break
    
    # Save Q-table snapshot
    if (episode + 1) % save_interval == 0:
        snapshots.append(q_table.copy())

# Add final snapshot
snapshots.append(q_table.copy())

# ======================
# VISUALIZATION
# ======================
def plot_paths(snapshots):
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('Path Progression During SARSA Training', y=1, fontsize=12)
    
    # Create colormap
    cmap = mcolors.ListedColormap(['white', 'black', 'green', 'red'])
    
    # Create 3x4 grid for 12 subplots
    for i, q_snapshot in enumerate(snapshots):
        ax = fig.add_subplot(3, 4, i+1)
        
        # Generate path
        path = [start_pos]
        state = start_pos
        for _ in range(50):  # Max path steps
            if state == goal_pos:
                break
            action = actions[np.argmax(q_snapshot[state[0], state[1]])]
            state = get_next_state(state, action)
            path.append(state)
        
        # Plot maze
        ax.imshow(maze, cmap=cmap)
        
        # Plot path
        path_x = [p[1] for p in path]
        path_y = [p[0] for p in path]
        ax.plot(path_x, path_y, marker='o', markersize=4, 
                color='blue', linewidth=1.2, alpha=0.7)
        
        ax.set_title(f'Episode {(i)*save_interval}', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()

# Generate and plot final path
def plot_final_path():
    path = [start_pos]
    state = start_pos
    while state != goal_pos:
        action = actions[np.argmax(q_table[state[0], state[1]])]
        state = get_next_state(state, action)
        path.append(state)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(maze, cmap=mcolors.ListedColormap(['white', 'black', 'green', 'red']))
    path_x = [p[1] for p in path]
    path_y = [p[0] for p in path]
    plt.plot(path_x, path_y, marker='o', color='blue', linewidth=2)
    plt.title('Final Optimal Path')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    print("Final Path Coordinates:")
    for p in path:
        print(f"({p[0]}, {p[1]})")

# Execute visualizations
plot_paths(snapshots)
plot_final_path()

# Learning progress plot
plt.figure(figsize=(10, 5))
plt.plot(steps_history)
plt.title('Learning Progress')
plt.xlabel('Episode')
plt.ylabel('Steps to Goal')
plt.grid(True)
plt.show()