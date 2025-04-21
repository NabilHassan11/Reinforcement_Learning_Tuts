import numpy as np

### One-hot encode states (input to NN)###

### This function takes a state (integer) and returns a one-hot encoded vector###
### example: one_hot(1, 3) ---> [0.0, 1.0, 0.0] ###
def one_hot(state, n_states=3):
    return np.eye(n_states)[state].astype(np.float32)

# ======================
# Neural Network Class
# ======================
class QNetwork:
    def __init__(self, input_size=3, hidden_size=4, output_size=2):
        # He initialization for ReLU
        ### Initialize weights for the hidden layer and output layer ###
        ### W1: input to hidden layer, W2: hidden to output layer ###
        ### Creates a matrix of shape (input_size × hidden_size) ###
        ### Scales the random weights by sqrt(2/n) where n is the number of input units ###
        ### This helps in faster convergence of the network ###
        scale = 1
        self.W1 = np.random.randn(input_size, hidden_size) * scale * np.sqrt(2 / input_size)
        self.W2 = np.random.randn(hidden_size, output_size) * scale *  np.sqrt(2 / hidden_size)
    
    def forward(self, x):
        ### Forward pass through the network ###
        ### x: input state (one-hot encoded) ###
        ### z1: pre-activation of hidden layer ###
        ### a1: activation of hidden layer (ReLU) ###
        ### q: Q-values for all actions (output layer) ###
        ### np.dot: matrix multiplication ###
        
        self.z1 = np.dot(x, self.W1)          # Hidden layer pre-activation
        self.a1 = np.maximum(0, self.z1)       # ReLU activation
        self.q = np.dot(self.a1, self.W2)      # Q-values for all actions
        
         # Force terminal state (state 2) to have zero Q-values
        if np.array_equal(x, one_hot(2)):
            self.q = np.zeros(2)
        return self.q
        
    def backward(self, x, action, td_error, lr=0.01):
        # Gradient for W2: dL/dW2 = a1 * td_error
        ### Compute the gradient of the loss with respect to W2 ###
        ### td_error: temporal difference error ###
        ### a1: activation of hidden layer (output of forward pass) ###
        
            # Create a vector of td_error for all actions
        td_error_vec = np.zeros(2)  # 2 actions
        td_error_vec[action] = td_error
        grad_W2 = np.outer(self.a1, td_error_vec)
        
        # Gradient for W1: dL/dW1 = x * (ReLU'(z1) * (W2 * td_error))
        ### Compute the gradient of the loss with respect to W1 ###
        ### ReLU'(z1): derivative of ReLU activation function ###
        ### delta: backpropagated error from output to hidden layer ###
        
        delta = np.dot(self.W2, td_error_vec) * (self.z1 > 0)  # ReLU derivative = 1 if z1 > 0
        grad_W1 = np.outer(x, delta)
        
        # Update weights
        ### Update the weights using the gradients computed above ###
        ### lr: learning rate ###
        # Clip gradients to prevent large updates
        grad_W2 = np.clip(grad_W2, -1, 1)
        grad_W1 = np.clip(grad_W1, -1, 1)
    
        # Update weights
        self.W2 += lr * grad_W2
        self.W1 += lr * grad_W1
        
# ======================
# Neural Network Training
# ======================
def train_nn(n_episodes=1000):
    ### Initialize the QNetwork ###
    ### q_net: instance of QNetwork class ###
    ### state: initial state (0) ###
    ### gamma: discount factor (0.9) ###
    ### alpha: learning rate (0.01) ###
    ### epsilon: exploration rate (0.1) ###
    
    q_net = QNetwork()
    state = 0  # Start state
    gamma = 0.9
    alpha = 0.01
    epsilon = 0.1

    for _ in range(n_episodes):
        x = one_hot(state) # One-hot encode the state
        q_values = q_net.forward(x) # Forward pass to get Q-values
        
        # Epsilon-greedy action selection
        ### Choose action based on epsilon-greedy policy ###
        ### If random number < epsilon, choose random action ###
        ### Else, choose action with highest Q-value ###
        if np.random.rand() < epsilon:
            action = np.random.choice(2)
        else:
            action = np.argmax(q_values)
        
        # Environment dynamics
        next_state = state + 1 if action == 1 else state - 1 # Move right or left
        next_state = np.clip(next_state, 0, 2) # Ensure state is within bounds
        reward = 1 if next_state == 2 else -0.5 # Reward for reaching terminal state
        
        # Calculate TD target
        if next_state == 2:  # Terminal state
            target = reward 
        else:
            x_next = one_hot(next_state) # One-hot encode next state
            q_next = q_net.forward(x_next) # Q-values for next state
            target = reward + gamma * np.max(q_next) # TD target Caluclation
        
        # Calculate TD error and update
        td_error = target - q_values[action] # Temporal difference error
        q_net.backward(x,action, td_error, lr=alpha) # Backpropagation to update weights
        
        # Update state
        state = next_state if next_state != 2 else 0 # Reset to start state if terminal state reached
    return q_net
# ======================
# Tabular Q-Learning
# ======================
def train_tabular(n_episodes=1000):
    ### Initialize Q-table ###
    ### q_table: 3 states, 2 actions (left/right) ###
    ### state: initial state (0) ###
    ### gamma: discount factor (0.9) ###
    ### alpha: learning rate (0.01) ###
    ### epsilon: exploration rate (0.1) ###
    ### q_table: table to store Q-values for each state-action pair ###
    q_table = np.zeros((3, 2))
    state = 0
    gamma = 0.9
    alpha = 0.01
    epsilon = 0.1

    for _ in range(n_episodes):
        # Epsilon-greedy action
        if np.random.rand() < epsilon:  # Random action
            action = np.random.choice(2) # left or right
        else:
            action = np.argmax(q_table[state]) # Choose action with highest Q-value
        
        # Environment dynamics
        next_state = state + 1 if action == 1 else state - 1 # Move right or left
        next_state = np.clip(next_state, 0, 2) # Ensure state is within bounds
        reward = 1 if next_state == 2 else -0.5 # Reward for reaching terminal state
        
        # TD update
        if next_state == 2:
            target = reward # Terminal state
        else:
            target = reward + gamma * np.max(q_table[next_state]) # TD target calculation
        
        q_table[state, action] += alpha * (target - q_table[state, action]) # Update Q-value
        
        # Update state
        state = next_state if next_state != 2 else 0 # Reset to start state if terminal state reached

    return q_table

# ======================
# Run and Compare
# ======================
if __name__ == "__main__":
    # Train both models
    nn_model = train_nn(n_episodes = 5000)
    tabular_model = train_tabular(n_episodes = 5000)

        # Compare Q-values for all states
    print("\n=== Neural Network Q-values ===")
    for state in range(3):
        nn_q = nn_model.forward(one_hot(state))
        print(f"State {state} - Left: {nn_q[0]:.3f}, Right: {nn_q[1]:.3f}")

    print("\n=== Tabular Q-values ===")
    for state in range(3):
        tabular_q = tabular_model[state]
        print(f"State {state} - Left: {tabular_q[0]:.3f}, Right: {tabular_q[1]:.3f}")

    print("\nVerification:")
    print("Expected policy:")
    print("State 0: Should prefer Right")
    print("State 1: Should prefer Right")
    print("State 2: Terminal state (values should be close to 0)")
    
    ### Note that the NN model may not converge to the same values as the tabular model due to the stochastic nature of training and the limited number of episodes. ###
    ### The NN model should learn to approximate the Q-values, but may not match the tabular model exactly. ###
    ### The tabular model is deterministic and will converge to the optimal Q-values given enough episodes. ###
    ### The NN model may require more training episodes or a different architecture to match the tabular model. ###
    ### The NN model can choose left action when it should choose right action, but it should learn to prefer the right action in state 0 and 1. ###
    ### The tabular model will always prefer the right action in state 0 and 1. ###