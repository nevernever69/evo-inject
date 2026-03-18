"""
Brain - Q-Network for action selection

Small neural network that maps state → action values.
State encodes app features, response context, organism history.
Actions select which target app to attack next.
Uses numpy only. Experience replay for stable learning.
"""

import numpy as np
from collections import deque
import random
from config import (
    STATE_SIZE, ACTION_SIZE, HIDDEN_LAYERS,
    LEARNING_RATE, DISCOUNT_FACTOR,
    EXPLORE_START, EXPLORE_MIN, EXPLORE_DECAY,
    REPLAY_BUFFER_SIZE, BATCH_SIZE,
)


class QNetwork:
    """
    Simple feedforward neural network: state → Q-values for each action.
    Architecture: STATE_SIZE → HIDDEN_LAYERS → ACTION_SIZE
    All numpy, no frameworks.
    """

    def __init__(self, weights=None):
        self.layer_sizes = [STATE_SIZE] + HIDDEN_LAYERS + [ACTION_SIZE]
        # Xavier initialization
        if weights is None:
            self.weights = []
            self.biases = []
            for i in range(len(self.layer_sizes) - 1):
                fan_in = self.layer_sizes[i]
                fan_out = self.layer_sizes[i + 1]
                scale = np.sqrt(2.0 / (fan_in + fan_out))
                self.weights.append(np.random.randn(fan_in, fan_out) * scale)
                self.biases.append(np.zeros(fan_out))
        else:
            self.weights, self.biases = weights

    def forward(self, state):
        """
        Forward pass. ReLU hidden layers, linear output.
        Returns Q-values for all actions.
        """
        x = np.array(state, dtype=np.float64)
        self._activations = [x]  # Cache for backprop

        for i in range(len(self.weights) - 1):
            x = x @ self.weights[i] + self.biases[i]
            x = np.maximum(0, x)  # ReLU
            self._activations.append(x)

        # Output layer (linear - no activation)
        x = x @ self.weights[-1] + self.biases[-1]
        self._activations.append(x)

        return x

    def backward(self, state, action, target_q):
        """
        Backpropagation for a single (state, action, target_q) tuple.
        Only updates the Q-value for the taken action.
        Returns loss.
        """
        q_values = self.forward(state)
        loss = (q_values[action] - target_q) ** 2

        # Gradient of loss w.r.t. output
        d_output = np.zeros(ACTION_SIZE)
        d_output[action] = 2.0 * (q_values[action] - target_q)

        # Backprop through layers (reverse order)
        d = d_output
        for i in range(len(self.weights) - 1, -1, -1):
            a = self._activations[i]

            # Gradient for weights and biases
            d_w = np.outer(a, d)
            d_b = d

            # Gradient for previous layer
            d = d @ self.weights[i].T

            # ReLU derivative (skip for input layer)
            if i > 0:
                d = d * (self._activations[i] > 0)

            # Update
            self.weights[i] -= LEARNING_RATE * np.clip(d_w, -1, 1)
            self.biases[i] -= LEARNING_RATE * np.clip(d_b, -1, 1)

        return loss

    def get_flat_weights(self):
        """Flatten all weights + biases into a single 1D array (for evolution)."""
        parts = []
        for w, b in zip(self.weights, self.biases):
            parts.append(w.flatten())
            parts.append(b.flatten())
        return np.concatenate(parts)

    def set_flat_weights(self, flat):
        """Load weights from a flat 1D array (from evolution)."""
        idx = 0
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]

            w_size = fan_in * fan_out
            self.weights[i] = flat[idx:idx + w_size].reshape(fan_in, fan_out)
            idx += w_size

            self.biases[i] = flat[idx:idx + fan_out].copy()
            idx += fan_out

    def num_params(self):
        """Total number of trainable parameters."""
        return sum(w.size + b.size for w, b in zip(self.weights, self.biases))

    def copy(self):
        """Deep copy of the network."""
        new_weights = ([w.copy() for w in self.weights],
                       [b.copy() for b in self.biases])
        return QNetwork(weights=new_weights)


class ReplayBuffer:
    """
    Experience replay buffer.
    Stores (state, action, reward, next_state, done) tuples.
    Samples random mini-batches for training.
    """

    def __init__(self, capacity=REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state, dtype=np.float64),
            action,
            reward,
            np.array(next_state, dtype=np.float64),
            done,
        ))

    def sample(self, batch_size=BATCH_SIZE):
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(list(self.buffer), batch_size)

    def __len__(self):
        return len(self.buffer)


class Brain:
    """
    The organism's brain. Wraps Q-Network + replay buffer + exploration.

    Decides what action to take given a server state.
    Learns from experience using Q-learning with experience replay.
    """

    def __init__(self, network=None):
        self.network = network or QNetwork()
        self.replay = ReplayBuffer()
        self.epsilon = EXPLORE_START
        self.total_steps = 0
        self.training_losses = []

    def choose_action(self, state):
        """
        Epsilon-greedy action selection.
        Returns action index.
        """
        self.total_steps += 1

        if random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, ACTION_SIZE - 1)
        else:
            # Exploit: best Q-value
            q_values = self.network.forward(state)
            return int(np.argmax(q_values))

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.replay.push(state, action, reward, next_state, done)

    def learn(self):
        """
        Sample a mini-batch from replay buffer and do Q-learning update.
        Returns average loss.
        """
        if len(self.replay) < BATCH_SIZE:
            return 0.0

        batch = self.replay.sample()
        total_loss = 0.0

        for state, action, reward, next_state, done in batch:
            # Compute target Q-value
            if done:
                target = reward
            else:
                next_q = self.network.forward(next_state)
                target = reward + DISCOUNT_FACTOR * np.max(next_q)

            # Update network
            loss = self.network.backward(state, action, target)
            total_loss += loss

        avg_loss = total_loss / len(batch)
        self.training_losses.append(avg_loss)

        # Decay exploration
        self.epsilon = max(EXPLORE_MIN, self.epsilon * EXPLORE_DECAY)

        return avg_loss

    def get_q_values(self, state):
        """Get Q-values for a state without exploration."""
        return self.network.forward(state)

    def copy(self):
        """Deep copy of brain (for offspring)."""
        new_brain = Brain(network=self.network.copy())
        new_brain.epsilon = self.epsilon
        return new_brain
