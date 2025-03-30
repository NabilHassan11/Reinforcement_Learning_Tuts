import numpy as np
import random

# Define gridworld size
GRID_SIZE = 5
TERMINAL_STATES = {(4, 4): 20, (4, 0): -20}  # Goal and bad state
ACTIONS = ['N', 'S', 'E', 'W']
ACTION_MAP = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1)}
gamma = 1.0  # Discount factor
alpha = 0.1  # Learning rate

# Initialize Q-values and policy
Q = {(r, c): {a: 0.0 for a in ACTIONS} for r in range(GRID_SIZE) for c in range(GRID_SIZE)}
pi = {(r, c): random.choice(ACTIONS) for r in range(GRID_SIZE) for c in range(GRID_SIZE)}
C = np.zeros((4, 5, 5))
# print(C[1,0,0])

def get_C(action):
    if(action == 'N'):
        return 0
    elif(action == 'S'):
        return 1
    elif(action == 'E'):
        return 2
    elif(action == 'W'):
        return 3

def get_next_state(state, action):
    """Returns the next state given an action while staying within the grid."""
    if state in TERMINAL_STATES:
        return state  # Terminal state stays the same
    
    r, c = state
    dr, dc = ACTION_MAP[action]
    new_r, new_c = r + dr, c + dc
    
    # Stay within bounds
    new_r = max(0, min(GRID_SIZE - 1, new_r))
    new_c = max(0, min(GRID_SIZE - 1, new_c))
    
    return (new_r, new_c)

def generate_episode():
    """Generates an episode using the behavior policy (random actions)."""
    episode = []
    state = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))  # Random start
    
    while state not in TERMINAL_STATES:
        action = random.choice(ACTIONS)  # Behavior policy is random
        next_state = get_next_state(state, action)
        reward = TERMINAL_STATES.get(next_state, -1)  # Default step cost is -1
        episode.append((state, action, reward))
        state = next_state
    
    return episode

# Off-policy Monte Carlo Control
for episode_num in range(5000):  # Run multiple episodes
    episode = generate_episode()
    G = 0  # Total return
    W = 1  # Importance sampling weight
    
    # Iterate backwards through the episode
    for t in reversed(range(len(episode))):
        state, action, reward = episode[t]
        G = gamma * G + reward  # Compute return
        
        
        #######################################################################################
        
        C[get_C(action), state[0], state[1]] += W
        
        # Update Q-value using C and importance weight W
        Q[state][action] += float(W/C[get_C(action), state[0], state[1]]) * (G - Q[state][action])
        
        ### This Method yielded better results than the α- learning rate one ###
        
        ########################################################################################
        
        ########################################################################################
        
        ### uncomment this to use and comment the above ###
        
        # # Update Q-value using α and importance weight W
        # Q[state][action] += alpha * W * (G - Q[state][action])
        
        ########################################################################################
        
        # Update target policy π to be greedy
        pi[state] = max(Q[state], key=Q[state].get)  # Choose best action
        
        # If the action wasn't selected by π, stop updating
        if action != pi[state]:
            break
        
        # Update importance sampling weight (since b is uniform, P(b) = 1/4)
        W *= 1.0 / 0.25

# Print learned policy
print("Learned Policy (Best Actions at Each State):")
for r in range(GRID_SIZE):
    print([pi[(r, c)] if (r, c) not in TERMINAL_STATES else "T" for c in range(GRID_SIZE)])

# Print Q-values
print("\nQ-values:")
for state in Q:
    print(f"State {state}: {Q[state]}")
