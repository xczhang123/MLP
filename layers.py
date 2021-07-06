import numpy as np

# Ref: cs231 assignment 2 template for the excellent comments
# https://cs231n.github.io/assignments2017/assignment2/
def affine_forward(x, w, b):
    """
    Compute the forward pass for a fully connected layer

    Inputs:
    - x: A numpy array containing input data, of shape (N, D)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M, )

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = x.dot(w) + b
    cache = (x, w, b)

    return out, cache

def affine_backward(dout, cache):
    """
    Compute the backward pass for a fully connected layer
    
    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of
        - x: Input data, of shape (N, D)
        - w: Weights, of shape (D, M)
        - b: Biases, of shape (M, )
    Returns a tuple of:
    - dx: Gradients with respect to x, of shape (N, D)
    - dw: Gradients with respect to w, of shape (D, M)
    - db: Gradients with respect to b, of shape (M, )
    """

    x, w, b = cache
    dx = dout.dot(w.T)
    dw = x.T.dot(dout)
    db = dout.sum(axis=0)

    return (dx, dw, db)

def sigmoid_forward(x):
    """
    Compute the forward pass for a layer of sigmoid activation

    Inputs:
    - x: Inputs, of any shape
    
    Returns a tuple of:
    - out: output, of the same shape as x
    - cache: x
    """
    out = 1.0 / (1.0 + np.exp(-x))
    cache = x
    
    return (out, cache)

def sigmoid_backward(dout, cache):
    """
    Compute the backward pass for a layer of sigmoid activation
    
    Inputs:
    - dout: Upstream derivative, of any shape
    - cache: Input x, of same shape as dout

    Returns a tuple of:
    - dx: Gradients with respect to x
    """
    x = cache
    a = 1.0 / (1.0 + np.exp(-x))
    dx = dout * (a * (1 - a))

    return dx

def tanh_forward(x):
    """
    Compute the forward pass for a layer of tanh activation

    Inputs:
    - x: Inputs, of any shape
    
    Returns a tuple of:
    - out: output, of the same shape as x
    - cache: x
    """
    out = np.tanh(x)
    cache = x
    
    return (out, cache)

def tanh_backward(dout, cache):
    """
    Compute the backward pass for a layer of tanh activation
    
    Inputs:
    - dout: Upstream derivative, of any shape
    - cache: Input x, of same shape as dout

    Returns a tuple of:
    - dx: Gradients with respect to x
    """
    x = cache
    dx = dout * (1.0 - np.tanh(x)**2)

    return dx

def lrelu_forward(x, a):
    """
    Compute the forward pass for a layer of leaky rectified layer units

    Inputs:
    - x: Inputs, of any shape
    - a: parameter alpha determining the graidient in the negtive regime (default to 0.1)
    Returns a tuple of:
    - out: output, of the same shape as x
    - cache: A tuple of x and a
    """
    out = np.maximum(x, a * x)
    cache = (x, a)

    return (out, cache)

def lrelu_backward(dout, cache):
    """
    Compute the backward pass for a layer of leaky rectified layer units
    
    Inputs:
    - dout: Upstream derivative, of any shape
    - cache: Input x, of same shape as dout

    Returns a tuple of:
    - dx: Gradients with respect to x
    - da: Gradients with respect to a
    """
    x, a = cache
    dx = dout * (x >= 0) + dout * a * (x < 0)
    da = np.sum(dout * np.minimum(x, 0))

    return (dx, da)

def relu_forward(x):
    """
    Compute the forward pass for a layer of rectified layer units

    Inputs:
    - x: Inputs, of any shape
    
    Returns a tuple of:
    - out: output, of the same shape as x
    - cache: x
    """
    out = np.maximum(x, 0)
    cache = x
    
    return (out, cache)

def relu_backward(dout, cache):
    """
    Compute the backward pass for a layer of rectified layer units
    
    Inputs:
    - dout: Upstream derivative, of any shape
    - cache: Input x, of same shape as dout

    Returns a tuple of:
    - dx: Gradients with respect to x
    """
    x = cache
    dx = dout * (x > 0)

    return dx

def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Compute the forward pass for batch normalization

    During training, the sample mean and sample variance are calculated from the minibatch statistics
    and used to normalize the incoming data.
    During training, we also keep an exponentially decaying running mean for mean and variance 
    of each feature, and these averages are used to normalize data at test-time

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean 
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Input:
    - x: Input of shape (N, D)
    - gamma: Scale parameter of shape (D, )
    - beta: Shift parameter of shape (D, )
    - bn_param: Dictionary with the following keys:
        - mode: 'train' or 'test', required
        - eps: Constant for numerical stability
        - momentum: Constant for running mean / variance
        - running_mean: Array of shape (D, ) giving the running mean of features
        - running_var: Array of shape (D, ) giving the running variance of features
    
    Returns a tuple of:
    - out: Output of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """

    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        # Step 1
        mu = 1./N * np.sum(x, axis=0)
        
        # Step 2
        xmu = x - mu

        # Step 3
        sq = xmu ** 2

        # Step 4
        var = 1./N * np.sum(sq, axis=0)
        
        # Step 5
        std = np.sqrt(var + eps)
        
        # Step 6
        ivar = 1./std

        # Step 7
        xhat = xmu * ivar

        # Step 8
        gammax = gamma * xhat
        
        # Step 9
        out = gammax + beta
        
        running_mean = momentum * running_mean + (1 - momentum) * mu 
        running_var = momentum * running_var + (1 - momentum) * (std**2)

        cache = (xhat, gamma, xmu, ivar, std, var, eps)
    elif mode == "test":
        out = gamma * ((x - running_mean) / np.sqrt(running_var + eps)) + beta
    
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var
    
    return out, cache

def batchnorm_backward(dout, cache):
    """
    Compute the backward pass for batch normalization

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: A tuple of values needed in the backward pass

    Returns a tuple of:
    - dx: Gradients with respect to inputs x, of shape (N, D)
    - dgamma: Gradients with respect to scale paramater gamma, of shape (D,)
    - dbeta: Gradients with respect to shift paramater gamma, of shape (D,)
    """

    # Ref: https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html#:~:text=Batch%20Normalization%20is%20a%20technique,makes%20this%20algorithm%20really%20powerful.


    N, D = dout.shape

    xhat, gamma, xmu, ivar, std, var, eps = cache

    # Step 9
    dbeta = np.sum(dout, axis=0)           # [D,]
    dgammax = dout                         # [NxD]

    # Step 8
    dgamma = np.sum(dgammax*xhat, axis=0)
    dxhat = dgammax * gamma

    # Step 7
    dxmu1 = dxhat * ivar                   # [NxD]
    divar = np.sum(dxhat * xmu, axis=0)    # [D,]

    # Step 6
    dstd = divar / -(std**2)               # [D,]

    # Step 5
    dvar = dstd * (0.5 / np.sqrt(var+eps)) # [D,]

    # Step 4
    dsq = dvar * 1. / N * np.ones((N, D))  # [NxD]

    # Step 3
    dxmu2 = dsq * 2 * xmu                  # [NxD]

    # Step 2
    dx1 = dxmu1 + dxmu2                    # [NxD]
    dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0) # [D,]

    # Step 1
    dx2 = dmu * 1. / N * np.ones((N,D))

    dx = dx1 + dx2

    return (dx, dgamma, dbeta)

def dropout_forward(x, dropout_param):
    """
    Calculate the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """

    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None
    if mode == 'train':
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
    elif mode == 'test':
        out = x
    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)
    
    return out, cache

def dropout_backward(dout, cache):
    """
    Calculate the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """

    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        dx = dout * mask
    elif mode == 'test':
        dx = dout
    
    return dx

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    # Calculate softmax score
    exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True)) # avoid numerical instability
    softmax_matrix = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # Calculate softmax loss
    loss = np.sum(-np.log(softmax_matrix[np.arange(N), y]))

    # Calcuate delta of the output layer
    softmax_matrix[np.arange(N), y] -= 1

    dx = softmax_matrix

    loss /= N
    dx /= N

    return loss, dx
    








