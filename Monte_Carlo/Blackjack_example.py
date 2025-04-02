import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
ACTIONS = ['HIT', 'STICK']
STATES_PLAYER_SUM = list(range(12, 22))  # 12-21
STATES_DEALER_CARD = list(range(1, 11))  # 1-10 (Ace=1)
USABLE_ACE = [True, False]

# Initialize Q-table and policy
Q = np.zeros((22, 11, 2, len(ACTIONS)))  # Q[player_sum][dealer_card][usable_ace][action]
policy = np.full((22, 11, 2), 'HIT', dtype=object)  # Initialize to HIT

# Set initial policy: Stick only on 20 or 21 (from the book example)
for p_sum in range(12, 22):
    for d_card in range(1, 11):
        for ace in [0, 1]:
            if p_sum >= 20:
                policy[p_sum, d_card, ace] = 'STICK'

def draw_card():
    """Draw a card with value between 1-10."""
    card = np.random.randint(1, 14)
    return min(card, 10)

def dealer_play(dealer_sum):
    """Dealer plays until sum >= 17"""
    while dealer_sum < 17:
        dealer_sum += draw_card()
    return dealer_sum

def generate_episode():
    """Generate an episode using exploring starts."""
    # Random initial state (exploring starts)
    player_sum = np.random.choice(STATES_PLAYER_SUM)
    dealer_showing = np.random.choice(STATES_DEALER_CARD)
    usable_ace = np.random.choice([True, False])
    
    # Initial random action (exploring starts)
    action = np.random.choice(ACTIONS)
    
    episode = []
    # player_trajectory = []
    current_sum = player_sum
    has_usable_ace = usable_ace
    
    while True:
        # Store current state before taking action
        state = (current_sum, dealer_showing, has_usable_ace)
        
        if action == 'HIT':
            # Player hits
            current_sum += draw_card()
            
            # Check if bust but has usable ace
            if current_sum > 21 and has_usable_ace:
                current_sum -= 10  # Convert ace from 11 to 1
                has_usable_ace = False
            
            # Record experience
            episode.append((state, action))
            
            if current_sum > 21:  # Player busts
                reward = -1
                break
            
        else:  # STICK
            # Dealer's turn
            dealer_hidden = draw_card()
            dealer_total = dealer_showing + dealer_hidden
            dealer_total = dealer_play(dealer_total)
            
            # Determine reward
            if dealer_total > 21:
                reward = 1
            elif current_sum > dealer_total:
                reward = 1
            elif current_sum == dealer_total:
                reward = 0
            else:
                reward = -1
            
            episode.append((state, action))
            break
        
        # Next action follows current policy
        action = policy[current_sum, dealer_showing, int(has_usable_ace)]
    
    return episode, reward

# Monte Carlo ES Training
num_episodes = 500000
returns_sum = np.zeros_like(Q)
returns_count = np.zeros_like(Q)

for i in range(num_episodes):
    if (i+1) % 100000 == 0:
        print(f"Processing episode {i+1}/{num_episodes}")
    
    episode, reward = generate_episode()
    
    # Update Q-values
    for (state, action) in episode:
        player_sum, dealer_card, usable_ace = state
        ace_idx = int(usable_ace)
        action_idx = ACTIONS.index(action)
        
        returns_sum[player_sum, dealer_card, ace_idx, action_idx] += reward
        returns_count[player_sum, dealer_card, ace_idx, action_idx] += 1
        if returns_count[player_sum, dealer_card, ace_idx, action_idx] > 0:
            Q[player_sum, dealer_card, ace_idx, action_idx] = (
                returns_sum[player_sum, dealer_card, ace_idx, action_idx] /
                returns_count[player_sum, dealer_card, ace_idx, action_idx]
            )
    
    # Update policy greedily
    for (state, _) in episode:
        player_sum, dealer_card, usable_ace = state
        ace_idx = int(usable_ace)
        
        if returns_count[player_sum, dealer_card, ace_idx].sum() > 0:
            best_action = np.argmax(Q[player_sum, dealer_card, ace_idx])
            policy[player_sum, dealer_card, ace_idx] = ACTIONS[best_action]

# Visualization
def plot_policy(policy_matrix, usable_ace):
    plt.figure(figsize=(10, 6))
    plt.title(f"Optimal Policy ({'With Usable Ace' if usable_ace else 'No Usable Ace'})")
    plt.xlabel("Dealer Showing Card")
    plt.ylabel("Player Sum")
    plt.xticks(range(1, 11))
    plt.yticks(range(12, 22))
    
    data = np.zeros((10, 10))  # 10 dealer cards x 10 player sums (12-21)
    
    for p_sum in range(12, 22):
        for d_card in range(1, 11):
            action = policy[p_sum, d_card, int(usable_ace)]
            data[p_sum-12, d_card-1] = 0 if action == 'HIT' else 1
    
    plt.imshow(data, cmap='RdYlGn', origin='lower', 
              extent=[0.5, 10.5, 11.5, 21.5], aspect='auto')
    plt.colorbar(ticks=[0, 1], label='Action (0=HIT, 1=STICK)')
    plt.show()

plot_policy(policy, usable_ace=True)
plot_policy(policy, usable_ace=False)