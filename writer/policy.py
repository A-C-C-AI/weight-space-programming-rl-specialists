"""Policy interface. A batch of six input channels maps to three action channels."""
import numpy as np

def act(weights, obs):
    h1 = np.tanh(obs @ weights["W1"] + weights["b1"])
    h2 = np.tanh(h1 @ weights["W2"] + weights["b2"])
    return np.tanh(h2 @ weights["W3"] + weights["b3"])
