import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Maze layout (0: empty, 1: wall, 2: start, 3: goal)
maze = np.array([
    [2, 0, 0, 1, 3],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0]
])

# Define parameters
actions = ["up", "down", "left", "right"]
num_actions = 4
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate
episodes = 500

# Initialize Q-table
q_table = np.zeros((maze.shape[0], maze.shape[1], num_actions))

# Find start and goal positions
start_pos = np.argwhere(maze == 2)[0]
goal_pos = np.argwhere(maze == 3)[0]

# Visualization setup
fig, ax = plt.subplots()
ax.set_xticks(np.arange(-0.5, 5, 1))
ax.set_yticks(np.arange(-0.5, 5, 1))
ax.grid(which="both", color="black", linewidth=2)
ax.set_xticklabels([])
ax.set_yticklabels([])

# Color mapping: Start=green, Goal=red, Agent=blue, Walls=black
cmap = plt.cm.colors.ListedColormap(['white', 'black', 'green', 'red'])
img = ax.imshow(maze, cmap=cmap, interpolation='nearest')

agent_patch = plt.Circle((start_pos[1], start_pos[0]), 0.3, fc='blue')
ax.add_patch(agent_patch)

def get_reward(state):
    if (state == goal_pos).all():
        return 10
    if maze[tuple(state)] == 1:
        return -10  # Wall penalty
    return -1  # Step penalty

def move(state, action):
    new_state = np.copy(state)
    if action == 0 and state[0] > 0:  # Up
        new_state[0] -= 1
    elif action == 1 and state[0] < 4:  # Down
        new_state[0] += 1
    elif action == 2 and state[1] > 0:  # Left
        new_state[1] -= 1
    elif action == 3 and state[1] < 4:  # Right
        new_state[1] += 1
    return new_state

# Training loop
history = []
for episode in range(episodes):
    state = np.copy(start_pos)
    done = False
    
    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(q_table[state[0], state[1]])
            
        next_state = move(state, action)
        reward = get_reward(next_state)
        
        # Q-learning update
        old_value = q_table[state[0], state[1], action]
        next_max = np.max(q_table[next_state[0], next_state[1]])
        new_value = old_value + alpha * (reward + gamma * next_max - old_value)
        q_table[state[0], state[1], action] = new_value
        
        # Update visualization every 50 episodes
        if episode % 50 == 0:
            history.append((state[0], state[1]))
            
        if (next_state == goal_pos).all() or maze[tuple(next_state)] == 1:
            done = True
            
        state = next_state

# Animation function with early stopping
def update(frame):
    y, x = history[frame]
    agent_patch.center = (x, y)
    
    # Stop animation if goal is reached
    if np.array_equal([x, y], goal_pos):
        ani.event_source.stop()
    
    return agent_patch,

# Create animation
ani = FuncAnimation(fig, update, frames=len(history), interval=50, blit=True)
plt.show()