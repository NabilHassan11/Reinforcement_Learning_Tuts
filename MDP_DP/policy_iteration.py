import numpy as np

# ================== Environment Setup ==================
states = ['O', 'NM', 'M', 'B']
state_to_idx = {s: i for i, s in enumerate(states)}
actions = ['PM', 'CO', 'SM', 'R']
gamma = 0.9
theta = 1e-6

# Initial policy and values
policy = {'O': 'CO', 'NM': 'SM', 'M': 'CO', 'B': 'PM'}
V = np.zeros(len(states))

# ================== Reward Function ==================
def get_reward(next_state):
    """Returns reward for entering a state"""
    if next_state == 'O': return 20
    elif next_state == 'NM': return -5
    elif next_state == 'B': return -50
    return 0  # Maintenance state

# ================== Transition Function ==================
def get_transitions(state, action):
    """Returns list of (probability, state_index) tuples"""
    transitions = []
    
    if state == 'O':
        if action == 'CO':
            transitions = [('O', 0.8), ('NM', 0.2)]
        elif action == 'PM':
            transitions = [('M', 1.0)]
            
    elif state == 'NM':
        if action == 'CO':
            transitions = [('B', 0.5), ('NM', 0.5)]
        elif action == 'PM' or action == 'SM':
            transitions = [('M', 1.0)]
            
    elif state == 'M':
        if action == 'SM':
            transitions = [('O', 1.0)]
        else:
            transitions = [('M', 1.0)]
            
    elif state == 'B':
        if action == 'R':
            transitions = [('O', 1.0)]
        else:
            transitions = [('B', 1.0)]
    
    return [(prob, state_to_idx[s]) for s, prob in transitions]

# ================== Policy Evaluation ==================
def policy_evaluation():
    global V
    while True:
        delta = 0
        new_V = np.copy(V)
        
        for state in states:
            state_idx = state_to_idx[state]
            current_action = policy[state]
            
            # Calculate expected value for policy action
            expected_value = 0
            for prob, next_idx in get_transitions(state, current_action):
                reward = get_reward(states[next_idx])
                expected_value += prob * (reward + gamma * V[next_idx])
            
            # Update value and delta
            new_V[state_idx] = expected_value
            delta = max(delta, abs(new_V[state_idx] - V[state_idx]))
        
        V = np.copy(new_V)
        if delta < theta:
            break

# ================== Policy Improvement ==================
def policy_improvement():
    global policy
    policy_stable = True
    
    for state in states:
        state_idx = state_to_idx[state]
        old_action = policy[state]
        best_action = old_action
        max_value = -np.inf
        
        # Find best action
        for action in actions:
            action_value = 0
            for prob, next_idx in get_transitions(state, action):
                reward = get_reward(states[next_idx])
                action_value += prob * (reward + gamma * V[next_idx])
            
            if action_value > max_value:
                max_value = action_value
                best_action = action
        
        # Update policy if needed
        if best_action != old_action:
            policy[state] = best_action
            policy_stable = False
            
    return policy_stable

# ================== Policy Iteration ==================
def policy_iteration():
    global V, policy
    iteration = 0
    
    while True:
        # Policy Evaluation
        policy_evaluation()
        
        # Policy Improvement
        stable = policy_improvement()
        
        if stable:
            break

# ================== Execution ==================
if __name__ == "__main__":
    policy_iteration()
    
    print("Final State Values:")
    for state, idx in state_to_idx.items():
        print(f"{state}: {V[idx]:.2f}")
    
    print("\nOptimal Policy:")
    for state, action in policy.items():
        print(f"{state}: {action}")