import numpy as np
from layers import *
from layer_utils import *
from optim import *

# Ref: cs231 assignment 2 template for the excellent comments and structure
# https://cs231n.github.io/assignments2017/assignment2/
class MLP(object):
    """
    A MLP with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=128,
        num_classes=10,
        activation='relu',
        dropout=1,
        normalization=None,
        reg=0.0,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new MLP.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - activation: activation function used after each fully connected layer. 
          Valid values are "sigmoid" or "tanh" or "leaky_relu" or "relu" (the default)
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm" or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.activation = activation
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        layers_dims = np.hstack([input_dim, hidden_dims, num_classes])

        for i in range(self.num_layers):
            self.params['W'+str(i+1)] = np.random.uniform( 
                                        low=-np.sqrt(6. / (layers_dims[i] + layers_dims[i+1])), 
                                        high=np.sqrt(6. / (layers_dims[i] + layers_dims[i+1])),
                                        size=(layers_dims[i], layers_dims[i+1])
            )
            if activation == 'sigmoid':
                self.params['W'+str(i+1)] *= 4
            self.params['b'+str(i+1)] = np.zeros(layers_dims[i+1])

        if self.normalization != None:
            # batch norm parameters
            for i in range(self.num_layers-1):
                self.params['gamma'+str(i+1)] = np.ones(layers_dims[i+1])
                self.params['beta' +str(i+1)] = np.zeros(layers_dims[i+1])
    
        if self.activation == "leaky_relu":
            # leaky relu parameters (default to 0.25)
            for i in range(self.num_layers-1):
                self.params['a'+str(i+1)] = np.array([0.25])

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test).
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
    
    def fit(self, X, y, X_val, y_val,
            learning_rate=5e-4, learning_rate_decay=1,
            num_epochs=10, batch_size=250, sample_batches=True,
            update_rule="sgd", verbose=True):

        """
        Run optimization to train the model.
        
        Inputs:
        - X: Array, shape (N_train, D) of training images
        - X_val: Array, shape (N_val, D) of validation images
        - y: Array, shape (N_train,) of labels for training images
        - y_val: Array, shape (N_val,) of labels for validation images
        - learning_rate: Parameters defining the speed of learning
        - learing_rate_decay: Parameters defining the learning rate decay after each epoch
        - num_epochs: Number of times the dataset is presented to the network for learning
        - batch_size: Size of minibatches used to compute loss and gradient
          during training.
        - sample_batches: Whether we want to sample minibatches
        - update_rule: A string giving the name of an update rule in optim.py.
          Default is 'sgd'.
        - verbose: Boolean; if set to false then no output will be printed
          during training.
        """

        """
        A dictionary containing hyperparameters that will be
        passed to the chosen update rule. Each update rule requires different
        hyperparameters (see optim.py) but all update rules require a
        'learning_rate' parameter so that should always be present.
        """
        optim_config = {}
        for p, w in self.params.items():
            optim_config[p] = {"learning_rate" : learning_rate}

        num_train = X.shape[0]
        if sample_batches:
            iterations_per_epoch = max(num_train // batch_size, 1)
        else:
            iterations_per_epoch = num_train
        num_iters = num_epochs * iterations_per_epoch

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            if sample_batches:
                batch_indices = np.random.choice(num_train, batch_size)
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
            else:
                X_batch = X
                y_batch = y
            
            loss, grads = self.loss(X_batch, y_batch)

            for p, w in self.params.items():
                dw = grads[p]
                config = optim_config[p]
                if update_rule == "sgd":
                    next_w, next_config = sgd(w, dw, config)
                elif update_rule == "sgd_momentum":
                    next_w, next_config = sgd_momentum(w, dw, config)
                elif update_rule == "rmsprop":
                    next_w, next_config = rmsprop(w, dw, config)
                elif update_rule == "adam":
                    next_w, next_config = adam(w, dw, config)
                self.params[p] = next_w
                optim_config[p] = next_config

            loss_history.append(loss)

            # Every epoch, check train and val accuracy and decay learning rate
            if (it + 1) % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                if (verbose):
                    print('Finished epoch %d / %d: loss %f, train %f, val %f' % 
                        ((it + 1) // iterations_per_epoch, num_epochs, loss, train_acc, val_acc))

                # Decay learning rate
                for p, w in self.params.items():
                    optim_config[p]["learning_rate"] *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history
        }

    def loss(self, X, y=None):
        """
        Compute the loss and gradients for MLP backbone

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.
        - loss_only: Return only loss or a tuple of (loss, grads)

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None

        # Forward pass
        x = X
        caches = []
        a, gamma, beta, bn_params = None, None, None, None

        # Forward pass through the first L - 1 hidden layers
        for i in range(self.num_layers-1):
            w = self.params['W'+str(i+1)]
            b = self.params['b'+str(i+1)]
            if self.normalization != None:
               gamma = self.params['gamma'+str(i+1)]
               beta  = self.params['beta'+str(i+1)]
               bn_params = self.bn_params[i]
            if self.activation == "leaky_relu":
                a = self.params['a'+str(i+1)]
            if self.activation == 'relu':
                x, cache = affine_norm_relu_forward(x,w,b, gamma, beta, bn_params, self.normalization,
                                                self.use_dropout, self.dropout_param)
            elif self.activation == 'sigmoid':
                x, cache = affine_norm_sigmoid_forward(x,w,b, gamma, beta, bn_params, self.normalization,
                                                self.use_dropout, self.dropout_param)
            elif self.activation == 'tanh':
                x, cache = affine_norm_tanh_forward(x,w,b, gamma, beta, bn_params, self.normalization,
                                                self.use_dropout, self.dropout_param)
            elif self.activation == 'leaky_relu':
                x, cache = affine_norm_lrelu_forward(x,w,b,a, gamma, beta, bn_params, self.normalization,
                                                self.use_dropout, self.dropout_param)
            caches.append(cache)
        
        # Forward pass through the last fc layer
        w = self.params['W'+str(self.num_layers)]
        b = self.params['b'+str(self.num_layers)]
        scores, cache = affine_forward(x,w,b)
        caches.append(cache)

        # If test mode return early
        if mode == "test":
            return scores

        # Calculate loss using softmax (with L2 regularization)
        loss, grads = 0.0, {}

        loss, softmax_grad = softmax_loss(scores, y)
        for i in range(self.num_layers):
            w = self.params['W'+str(i+1)]
            loss += 0.5 * self.reg * np.sum(w * w) 

        # Backward pass through the softmax layer
        dout = softmax_grad

        # Backward pass through the last affine layer
        dout, dw, db = affine_backward(dout, caches[self.num_layers - 1])
        grads['W' + str(self.num_layers)] = dw + self.reg * self.params['W' + str(self.num_layers)]
        grads['b' + str(self.num_layers)] = db

        # Backward pass through the rest of hidden layers
        for i in range(self.num_layers - 2, -1, -1):
            if self.activation == 'relu':
                dx, dw, db, dgamma, dbeta = affine_norm_relu_backward(dout, caches[i], self.normalization,
                                                                  self.use_dropout)
            elif self.activation == 'sigmoid':
                dx, dw, db, dgamma, dbeta = affine_norm_sigmoid_backward(dout, caches[i], self.normalization,
                                                                  self.use_dropout)
            elif self.activation == 'tanh':
                dx, dw, db, dgamma, dbeta = affine_norm_tanh_backward(dout, caches[i], self.normalization,
                                                                  self.use_dropout)
            elif self.activation == 'leaky_relu':
                dx, dw, db, da, dgamma, dbeta = affine_norm_lrelu_backward(dout, caches[i], self.normalization,
                                                                  self.use_dropout)
            if self.normalization != None:
                grads['gamma'+str(i+1)] = dgamma
                grads['beta' +str(i+1)] = dbeta
            if self.activation == 'leaky_relu':
                grads['a'+str(i+1)] = da
            grads['W' + str(i + 1)] = dw + self.reg * self.params['W' + str(i + 1)]
            grads['b' + str(i + 1)] = db
            dout = dx

        return loss, grads

    def predict(self, X):
        y_pred = None
        y_pred = np.argmax(self.loss(X), axis=1)

        return y_pred