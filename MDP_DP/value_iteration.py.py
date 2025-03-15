import numpy as np

# ================== Fixed State Representation ==================
states = ['O', 'NM', 'M', 'B']
state_to_idx = {s: i for i, s in enumerate(states)}  # Map states to indices
actions = ['PM', 'CO', 'SM', 'R']

gamma = 0.9
theta = 1e-6

# ================== Fixed Reward Function (Next-state based) ==================
def get_reward(next_state):
    """Reward based on RESULTING state (per assignment spec)"""
    if next_state == 'O':
        return 20
    elif next_state == 'NM':
        return -5
    elif next_state == 'B':
        return -50
    return 0  # Maintenance state (M)

# ================== Fixed Transition Function ==================
def get_transitions(state, action):
    """
    Returns list of (probability, next_state) tuples
    Based on assignment state diagram
    """
    transitions = []
    
    if state == 'O':
        if action == 'CO':
            transitions = [('O', 0.8), ('NM', 0.2)]  # 80% stay, 20% degrade
        elif action == 'PM':
            transitions = [('M', 1.0)]  # Force maintenance
            
    elif state == 'NM':
        if action == 'CO':
            transitions = [('B', 0.5), ('NM', 0.5)]  # 50% break
        elif action == 'PM':
            transitions = [('M', 1.0)]  # Start maintenance
        elif action == 'SM':
            transitions = [('M', 1.0)]  # Schedule maintenance
            
    elif state == 'M':
        if action == 'SM':
            transitions = [('O', 1.0)]  # Complete maintenance
        else:
            transitions = [('M', 1.0)]  # Stay in maintenance
            
    elif state == 'B':
        if action == 'R':
            transitions = [('O', 1.0)]  # Repair to operational
        else:
            transitions = [('B', 1.0)]  # Stay broken
            
    # Convert state names to indices and add probabilities
    return [(prob, state_to_idx[next_s]) for next_s, prob in transitions]

# ================== Fixed Value Iteration ==================
V = np.zeros(len(states))  # Now using proper indices

def value_iteration():
    global V
    while True:
        delta = 0
        new_V = np.copy(V)
        
        for state in states:
            state_idx = state_to_idx[state]
            
            # Skip terminal state (broken stays broken without repair)
            if state == 'B' and action != 'R':
                continue
                
            max_value = -np.inf
            for action in actions:
                expected_value = 0
                # Calculate expected value over all transitions
                for prob, next_idx in get_transitions(state, action):
                    reward = get_reward(states[next_idx])  # Get reward for next state
                    expected_value += prob * (reward + gamma * V[next_idx])
                
                if expected_value > max_value:
                    max_value = expected_value
                    
            delta = max(delta, abs(new_V[state_idx] - max_value))
            new_V[state_idx] = max_value
            
        V = np.copy(new_V)
        if delta < theta:
            break

# ================== Fixed Policy Extraction ==================
def extract_policy():
    policy = {}
    for state in states:
        state_idx = state_to_idx[state]
        best_action = None
        max_value = -np.inf
        
        for action in actions:
            expected_value = 0
            for prob, next_idx in get_transitions(state, action):
                reward = get_reward(states[next_idx])
                expected_value += prob * (reward + gamma * V[next_idx])
                
            if expected_value > max_value:
                max_value = expected_value
                best_action = action
                
        policy[state] = best_action
        
    return policy

# ================== Execution ==================
if __name__ == "__main__":
    value_iteration()
    optimal_policy = extract_policy()
    
    print("Optimal State Values:")
    for state, idx in state_to_idx.items():
        print(f"{state}: {V[idx]:.2f}")
    
    print("\nOptimal Policy:")
    for state, action in optimal_policy.items():
        print(f"{state}: {action}")