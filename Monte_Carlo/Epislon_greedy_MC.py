import numpy as np
import matplotlib.pyplot as plt

# =============================================
# Title: Epsilon-Greedy Monte Carlo Control for Simple MDP
# =============================================

# ---------------------------
# Environment Configuration
# ---------------------------
states = ["S1", "S2", "Terminal"]
actions = {
    "S1": ["A", "B"],
    "S2": ["C"],
    "Terminal": []
}

# Deterministic transitions: (current_state, action) -> next_state
transitions = {
    ("S1", "A"): "S2",
    ("S1", "B"): "Terminal",
    ("S2", "C"): "Terminal"
}

# Rewards: (current_state, action) -> reward
rewards = {
    ("S1", "A"): 5,
    ("S1", "B"): 2,
    ("S2", "C"): 10
}

# ---------------------------
# Algorithm Parameters
# ---------------------------
gamma = 1.0       # Discount factor
epsilon = 0.2     # Exploration rate
episodes = 100     # Number of training episodes

# ---------------------------
# Initialize Data Structures
# ---------------------------
# Q-values (state-action pairs)
Q = {
    ("S1", "A"): 0.0,
    ("S1", "B"): 0.0,
    ("S2", "C"): 0.0
}

# Policy: probability distribution for each state
policy = {
    "S1": {"A": 0.5, "B": 0.5},  # Initial ε-soft policy (50/50)
    "S2": {"C": 1.0}             # Only one action
}

# Track policy probabilities over episodes
policy_history = {
    "S1_A": [],
    "S1_B": []
}

# ---------------------------
# Monte Carlo Control Loop
# ---------------------------
for episode in range(episodes):
    # Generate an episode using current policy
    trajectory = []
    current_state = "S1"
    
    while current_state != "Terminal":
        # Choose action probabilistically
        available_actions = list(policy[current_state].keys())
        probs = list(policy[current_state].values())
        chosen_action = np.random.choice(available_actions, p=probs)
        
        # Record (state, action, reward)
        reward = rewards[(current_state, chosen_action)]
        trajectory.append((current_state, chosen_action, reward)) # State Action Reward
        
        # Transition to next state
        current_state = transitions[(current_state, chosen_action)]
    
    # Process episode backward (first-visit MC)
    G = 0
    visited = set()
    
    for t in reversed(range(len(trajectory))):
        state, action, reward = trajectory[t]
        sa_pair = (state, action)
        
        # First-visit check
        if sa_pair not in visited:
            visited.add(sa_pair)
            G = gamma * G + reward
            
            # Update Q-value (simple average for demonstration)
            Q[sa_pair] = (Q[sa_pair] * (episode) + G) / (episode + 1)
    
    # Update policy (ε-greedy)
    for state in states:
        if state == "Terminal":
            continue
        
        # Find best action(s)
        q_values = {a: Q[(state, a)] for a in actions[state]}
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        
        # Update probabilities
        num_actions = len(actions[state])
        for a in actions[state]:
            if a in best_actions:
                # Split (1-ε) among best actions + ε/num_actions
                prob = (1 - epsilon)/len(best_actions) + epsilon/num_actions
            else:
                prob = epsilon/num_actions
            policy[state][a] = prob
    
    # Track policy probabilities for S1
    policy_history["S1_A"].append(policy["S1"]["A"])
    policy_history["S1_B"].append(policy["S1"]["B"])

# ---------------------------
# Visualization
# ---------------------------
plt.figure(figsize=(10, 4))
plt.plot(policy_history["S1_A"], label="Action A (Go to S2)", color="blue")
plt.plot(policy_history["S1_B"], label="Action B (Terminate)", color="red")
plt.xlabel("Episode")
plt.ylabel("Probability")
plt.title("Epsilon-Greedy Policy Evolution for State S1")
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------
# Final Results
# ---------------------------
print("\nFinal Q-values:")
print(f"Q(S1, A) = {Q[('S1', 'A')]:.2f}")
print(f"Q(S1, B) = {Q[('S1', 'B')]:.2f}")
print(f"Q(S2, C) = {Q[('S2', 'C')]:.2f}")

print("\nFinal Policy for S1:")
print(f"P(A|S1) = {policy['S1']['A']*100:.1f}%")
print(f"P(B|S1) = {policy['S1']['B']*100:.1f}%")