import numpy as np
import matplotlib.pyplot as plt

# Initialize parameters
gamma = 1.0  # Discount factor
theta = 1e-4  # Threshold for convergence
reward = -1  # Reward for each step
policy = 0.25  # Probability for each action (uniform random policy)
grid_size = 4
num_states = grid_size * grid_size

# Initialize value function
V = np.zeros(num_states)
terminal_state = [0, 15]

def visualize_values(V, grid_size):
    plt.imshow(V.reshape((grid_size, grid_size)), cmap='coolwarm', interpolation='none')
    plt.colorbar(label='State Value')
    plt.title('Grid World State Values')
    plt.show()

def policy_evaluation():
    while True:
        delta = 0
        for s in range(num_states):
            if s in terminal_state:
                continue

            v = V[s]
            new_v = 0
            for action in [-1, 1, -grid_size, grid_size]:
                ns = s + action
                if ns < 0 or ns >= num_states or (s % grid_size == 0 and action == -1) or ((s + 1) % grid_size == 0 and action == 1):
                    ns = s
                new_v += policy * (reward + gamma * V[ns])

            V[s] = new_v
            delta = max(delta, abs(v - V[s]))
        
        
        print(delta)
        if delta < theta:
            break
    
    

policy_evaluation()
print(V)
visualize_values(V, grid_size)
