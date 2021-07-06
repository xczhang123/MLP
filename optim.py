import numpy as np


# Ref 1: cs231 assignment 2 template for the excellent comments
# https://cs231n.github.io/assignments2017/assignment2/

# Ref 2: cs231, https://www.youtube.com/watch?v=_JB0AO7QxSA

"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.
"""

def sgd(w, dw, config=None): 
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config

def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    next_w = None
    
    v = config["momentum"] * v + config["learning_rate"] * dw
    next_w = w - v
    config["v"] = v

    return next_w, config

def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    next_w = None

    learning_rate, decay_rate = config["learning_rate"], config["decay_rate"]
    epsilon, v = config["epsilon"], config["cache"]

    v = decay_rate * v + (1 - decay_rate) * (dw * dw)
    next_w = w - learning_rate * dw / (np.sqrt(v) + epsilon)
    config["cache"] = v

    return next_w, config

def adam(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    next_w = None

    learning_rate, epsilon = config["learning_rate"], config["epsilon"]
    beta1, beta2 = config["beta1"], config["beta2"]
    m, v, t = config["m"], config["v"], config["t"]

    t = t + 1
    m = beta1 * m + (1 - beta1) * dw          # SGD momentum
    mbias = m / (1 - beta1 ** t)            # Bias correction
    v = beta2 * v + (1 - beta2) * (dw * dw)   # RMSprop
    vbias = v / (1 - beta2 ** t)             # Bias correction

    next_w = w - learning_rate * mbias / (np.sqrt(vbias) + epsilon)  

    config["m"], config["v"], config["t"] = m, v, t

    return next_w, config