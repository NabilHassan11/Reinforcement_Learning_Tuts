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
counts = np.zeros((22, 11, 2, len(ACTIONS)), dtype=int)  # Visit counts

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

# Training with incremental updates
num_episodes = 1000000

for i in range(num_episodes):
    if (i+1) % 100000 == 0:
        print(f"Episode {i+1}/{num_episodes}")
    
    episode, reward = generate_episode()
    
    # Update Q-values incrementally
    for (state, action) in episode:
        p_sum, d_card, usable_ace = state
        ace_idx = int(usable_ace)
        action_idx = ACTIONS.index(action)
        
        counts[p_sum, d_card, ace_idx, action_idx] += 1
        alpha = 1 / counts[p_sum, d_card, ace_idx, action_idx]
        Q[p_sum, d_card, ace_idx, action_idx] += alpha * (
            reward - Q[p_sum, d_card, ace_idx, action_idx]
        )
    
    # Update policy greedily
    for (state, _) in episode:
        p_sum, d_card, usable_ace = state
        ace_idx = int(usable_ace)
        
        if np.sum(counts[p_sum, d_card, ace_idx]) > 0:
            best_action = np.argmax(Q[p_sum, d_card, ace_idx])
            policy[p_sum, d_card, ace_idx] = ACTIONS[best_action]

# Visualization
def plot_combined_policies(policy_matrix):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('Optimal Blackjack Policies', y=0.98, fontsize=14)
    
    titles = ['With Usable Ace', 'Without Usable Ace']
    cmap = plt.get_cmap('RdYlGn')
    
    for i, (usable_ace, title) in enumerate(zip([True, False], titles)):
        # Create policy matrix
        data = np.zeros((10, 10))
        for p_sum in range(12, 22):
            for d_card in range(1, 11):
                action = policy[p_sum, d_card, int(usable_ace)]
                data[p_sum-12, d_card-1] = 0 if action == 'HIT' else 1
        
        # Plot to subplot
        im = axes[i].imshow(data, cmap=cmap, origin='lower', 
                          extent=[0.5, 10.5, 11.5, 21.5], aspect='auto')
        axes[i].set_title(title, pad=12)
        axes[i].set_xlabel('Dealer Showing Card')
        axes[i].set_ylabel('Player Sum')
        axes[i].set_xticks(range(1, 11))
        axes[i].set_yticks(range(12, 22))
        
        # Add colorbar for each subplot
        cbar = fig.colorbar(im, ax=axes[i], shrink=0.8)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['HIT (0)', 'STICK (1)'])
        cbar.set_label('Optimal Action', rotation=270, labelpad=10)

    plt.tight_layout()
    plt.show()

# Call the combined plotting function
plot_combined_policies(policy)