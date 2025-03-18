# Define the episodes (states, rewards, and terminal state)
episodes = [
    {
        "states": ["A", "A", "B", "A", "B"],
        "rewards": [3, 2, -4, 4, -3],
    },
    {
        "states": ["B", "A", "B"],
        "rewards": [-2, 3, -3],
    },
]

# Precompute first-visit flags for each episode
for episode in episodes:
    states = episode["states"]
    visited = set()
    first_visit = []
    for state in states:
        if state not in visited:
            first_visit.append(True)
            visited.add(state)
        else:
            first_visit.append(False)
    episode["first_visit"] = first_visit

# [{'states': ['A', 'A', 'B', 'A', 'B'], 'rewards': [3, 2, -4, 4, -3], 'first_visit': [True, False, True, False, False]},
#  {'states': ['B', 'A', 'B'], 'rewards': [-2, 3, -3], 'first_visit': [True, True, False]}]



def mc_prediction(episodes, method="first_visit", gamma=1.0):
    """
    Compute state-value functions using Monte Carlo methods.
    
    Args:
        episodes (list): List of episodes with states, rewards, and first_visit flags.
        method (str): "first_visit" or "every_visit".
        gamma (float): Discount factor (default=1.0).
    
    Returns:
        dict: Estimated value for each state.
    """
    returns = {}  # Stores cumulative returns for each state
    
    for episode in episodes:
        states = episode["states"]
        rewards = episode["rewards"]
        first_visit = episode["first_visit"]
        G = 0  # Initialize return
        
        # Process the episode backward
        for t in reversed(range(len(rewards))):
            state = states[t]
            # print(state)
            reward = rewards[t]
            G = gamma * G + reward  # Update discounted return
            
            # Update returns based on the method
            if method == "first_visit":
                if first_visit[t]:
                    if state not in returns:
                        returns[state] = []
                    returns[state].append(G)
            elif method == "every_visit":
                if state not in returns:
                    returns[state] = []
                returns[state].append(G)
            # print(returns)
    
    # Calculate average returns for each state
    V = {state: sum(returns_list) / len(returns_list) for state, returns_list in returns.items()}
    return V

# Calculate values using First-Visit and Every-Visit MC
# V_first_visit = mc_prediction(episodes, method="first_visit", gamma=1.0)
V_every_visit = mc_prediction(episodes, method="every_visit", gamma=1.0)

print("First-Visit MC Results:")
print(f"V(A) = {V_first_visit['A']}, V(B) = {V_first_visit['B']}\n")

print("Every-Visit MC Results:")
print(f"V(A) = {V_every_visit['A']}, V(B) = {V_every_visit['B']}")