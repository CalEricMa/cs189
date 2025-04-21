"""
Author: Sophia Sanborn, Sagnik Bhattacharya
Institution: UC Berkeley
Date: Spring 2020
Course: CS189/289A
Website: github.com/sophiaas, github.com/sagnibak
Modified by: Bill Zheng, Tejas Prabhune, Spring 2025
Website: github.com/WJ2003B, github.com/tejasprabhune
"""

import numpy as np
from abc import ABC, abstractmethod

from neural_networks.activations import initialize_activation
from neural_networks.weights import initialize_weights
from collections import OrderedDict

from typing import Callable, List, Literal, Tuple, Union


class Layer(ABC):
    """Abstract class defining the `Layer` interface."""

    def __init__(self):
        self.activation = None

        self.n_in = None
        self.n_out = None

        self.parameters = {}
        self.cache = {}
        self.gradients = {}

        super().__init__()

    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        pass

    def clear_gradients(self) -> None:
        self.cache = OrderedDict({a: [] for a, b in self.cache.items()})
        self.gradients = OrderedDict(
            {a: np.zeros_like(b) for a, b in self.gradients.items()}
        )

    def forward_with_param(
        self, param_name: str, X: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Call the `forward` method but with `param_name` as the variable with
        value `param_val`, and keep `X` fixed.
        """

        def inner_forward(param_val: np.ndarray) -> np.ndarray:
            self.parameters[param_name] = param_val
            return self.forward(X)

        return inner_forward

    def _get_parameters(self) -> List[np.ndarray]:
        return [b for a, b in self.parameters.items()]

    def _get_cache(self) -> List[np.ndarray]:
        return [b for a, b in self.cache.items()]

    def _get_gradients(self) -> List[np.ndarray]:
        return [b for a, b in self.gradients.items()]


def initialize_layer(
    name: str,
    activation: str = None,
    weight_init: str = None,
    n_out: int = None,
    kernel_shape: Tuple[int, int] = None,
    stride: int = None,
    pad: int = None,
    mode: str = None,
    eps: float = 1e-8,
    momentum: float = 0.95,
    keep_dim: str = "first",
) -> Layer:
    """Factory function for layers."""
    if name == "fully_connected":
        return FullyConnected(
            n_out=n_out, activation=activation, weight_init=weight_init,
        )

    elif name == "conv2d":
        return Conv2D(
            n_out=n_out,
            activation=activation,
            kernel_shape=kernel_shape,
            stride=stride,
            pad=pad,
            weight_init=weight_init,
        )
    
    elif name == "batchnorm1d":
        return BatchNorm1D(eps=eps, momentum=momentum,)

    elif name == "pool2d":
        return Pool2D(kernel_shape=kernel_shape, mode=mode, stride=stride, pad=pad)

    elif name == "flatten":
        return Flatten(keep_dim=keep_dim)

    else:
        raise NotImplementedError("Layer type {} is not implemented".format(name))


class FullyConnected(Layer):
    """A fully-connected layer multiplies its input by a weight matrix, adds
    a bias, and then applies an activation function.
    """

    def __init__(
        self, n_out: int, activation: str, weight_init="xavier_uniform"
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.activation = initialize_activation(activation)

        # instantiate the weight initializer
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int, int]) -> None:
        """Initialize all layer parameters (weights, biases)."""
        self.n_in = X_shape[1]

        ### BEGIN YOUR CODE ###

        W = self.init_weights((self.n_in, self.n_out))
        b = np.zeros((1, self.n_out))


        self.parameters = OrderedDict({"W": W, "b": b}) 

        # self.cache: OrderedDict = ...  # cache for backprop
        # self.gradients: OrderedDict = ...  # parameter gradients initialized to zero
        #                                    # MUST HAVE THE SAME KEYS AS `self.parameters`

        self.cache = OrderedDict()
        self.gradients = OrderedDict([
            ("W", np.zeros_like(W)),
            ("b", np.zeros_like(b))
        ])
        ### END YOUR CODE ###

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: multiply by a weight matrix, add a bias, apply activation.
        Also, store all necessary intermediate results in the `cache` dictionary
        to be able to compute the backward pass.

        Parameters
        ----------
        X  input matrix of shape (batch_size, input_dim)

        Returns
        -------
        a matrix of shape (batch_size, output_dim)
        """
        # initialize layer parameters if they have not been initialized
        if self.n_in is None:
            self._init_parameters(X.shape)

        ### BEGIN YOUR CODE ###


        W = self.parameters["W"]     
        b = self.parameters["b"]     
        Z = X.dot(W) + b            
        self.cache["X"] = X
        self.cache["Z"] = Z

        out = self.activation.forward(Z)
        

        # store information necessary for backprop in `self.cache`

        ### END YOUR CODE ###

        return out

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for fully connected layer.
        Compute the gradients of the loss with respect to:
            1. the weights of this layer (mutate the `gradients` dictionary)
            2. the bias of this layer (mutate the `gradients` dictionary)
            3. the input of this layer (return this)

        Parameters
        ----------
        dLdY  gradient of the loss with respect to the output of this layer
              shape (batch_size, output_dim)

        Returns
        -------
        gradient of the loss with respect to the input of this layer
        shape (batch_size, input_dim)
        """
        ### BEGIN YOUR CODE ###

        W = self.parameters["W"]     
        X = self.cache["X"]          
        Z = self.cache["Z"]          

        dZ = self.activation.backward(Z, dLdY) 

        dW = X.T.dot(dZ)         
        db = np.sum(dZ, axis=0, keepdims=True)  
        dX = dZ.dot(W.T)            

        self.gradients["W"] = dW
        self.gradients["b"] = db

        ### END YOUR CODE ###

        return dX


class BatchNorm1D(Layer):
    def __init__(
        self, 
        weight_init: str = "xavier_uniform",
        eps: float = 1e-8,
        momentum: float = 0.9,
        n_in: int = None, 
    ) -> None:
        super().__init__()
        
        # instantiate the weight initializer
        self.init_weights = initialize_weights(weight_init,)

        self.eps = eps
        self.momentum = momentum
        self.n_in = None

        self.running_mu  = None
        self.running_var = None
        # cache placeholder
        self.cache = OrderedDict()

        # If test passed in n_in, immediately init parameters
        if n_in is not None:
            # we only need the second dimension
            self._init_parameters((0, n_in))


    def _init_parameters(self, X_shape: Tuple[int, int]) -> None:
        """Initialize all layer parameters (weights, biases)."""
        self.n_in = X_shape[1]

        ### BEGIN YOUR CODE ###

        gamma = self.init_weights((1, X_shape[1]))
        beta  = np.zeros((1, X_shape[1]))

        # self.parameters = OrderedDict({"gamma": gamma, "beta": beta}) # DO NOT CHANGE THE KEYS
        # self.cache = OrderedDict({"X": ..., "X_hat": ..., 
        #                           "mu": ..., "var": ..., 
        #                           "running_mu": ..., "running_var": ...})  
        # # cache for backprop
        # self.gradients: OrderedDict = ...  # parameter gradients initialized to zero
        #                                    # MUST HAVE THE SAME KEYS AS `self.parameters`

        self.parameters = OrderedDict([
            ("gamma", gamma),
            ("beta",  beta),
        ])

        self.gradients = OrderedDict([
            ("gamma", np.zeros_like(gamma)),
            ("beta",  np.zeros_like(beta)),
        ])

        self.running_mu  = np.zeros((1, X_shape[1]))
        self.running_var = np.zeros((1, X_shape[1]))
        # placeholder for forward‐cache
        self.cache = OrderedDict([
            ("X",      None),
            ("X_hat",  None),
            ("mu",     None),
            ("var",    None),
        ])
        ### END YOUR CODE ###

    def forward(self, X: np.ndarray, mode: str = "train") -> np.ndarray:
        """ Forward pass for 1D batch normalization layer.
        Allows taking in an array of shape (B, C) and performs batch normalization over it. 

        We use Exponential Moving Average to update the running mean and variance. with alpha value being equal to self.gamma

        You should set the running mean and running variance to the mean and variance of the first batch after initializing it.
        You should also make separate cases for training mode and testing mode.
        """
        ### BEGIN YOUR CODE ###

        # implement a batch norm forward pass

        # cache any values required for backprop


        if self.n_in is None:
            self._init_parameters(X.shape)

        gamma = self.parameters["gamma"]  # (1, C)
        beta  = self.parameters["beta"]   # (1, C)

        if mode == "train":
            # 1) batch statistics
            mu  = X.mean(axis=0, keepdims=True)         # (1, C)
            var = X.var(axis=0, keepdims=True)          # (1, C)

            # 2) normalize
            X_centered = X - mu                         # (B, C)
            inv_std    = 1.0 / np.sqrt(var + self.eps)  # (1, C)
            X_hat      = X_centered * inv_std           # (B, C)

            # 3) scale & shift
            out = gamma * X_hat + beta                  # (B, C)

            # 4) update running stats
            self.running_mu  = self.momentum * self.running_mu \
                                 + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var \
                                 + (1 - self.momentum) * var

            # 5) cache for backward
            self.cache["X"]     = X
            self.cache["X_hat"] = X_hat
            self.cache["mu"]    = mu
            self.cache["var"]   = var

            self.cache["running_mu"]  = self.running_mu
            self.cache["running_var"] = self.running_var

        else:
            # test mode: use running averages
            X_centered = X - self.running_mu
            inv_std    = 1.0 / np.sqrt(self.running_var + self.eps)
            X_hat      = X_centered * inv_std
            out        = gamma * X_hat + beta


        ### END YOUR CODE ###
        return out

    def backward(self, dY: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward method for batch normalization layer. You don't need to implement this to get full credit, although it is
        fun to do so if you have the time.
        """

        ### BEGIN YOUR CODE ###

        # implement backward pass for batchnorm.


        X     = self.cache["X"]
        X_hat = self.cache["X_hat"]
        mu    = self.cache["mu"]
        var   = self.cache["var"]
        gamma = self.parameters["gamma"]
        B, C  = X.shape

        # Gradients w.r.t. gamma & beta

        dgamma = np.sum(dY * X_hat, axis=0)  # (C,)
        dbeta  = np.sum(dY,       axis=0)    # (C,)


        # Gradient w.r.t. normalized X
        dX_hat = dY * gamma                                 # (B, C)

        # Backprop through normalization
        inv_std = 1.0 / np.sqrt(var + self.eps)             # (1, C)
        dvar    = np.sum(dX_hat * (X - mu) * -0.5 * inv_std**3,
                         axis=0, keepdims=True)            # (1, C)
        dmu     = np.sum(dX_hat * -inv_std, axis=0, keepdims=True)
        dmu    += dvar * np.mean(-2 * (X - mu), axis=0, keepdims=True)

        # Gradient w.r.t. original input X
        dX = (dX_hat * inv_std) \
           + (dvar * 2 * (X - mu) / B) \
           + (dmu / B)

        # Store parameter gradients
        self.gradients["gamma"] = dgamma
        self.gradients["beta"]  = dbeta



        ### END YOUR CODE ###
        

        return dX

class Conv2D(Layer):
    """Convolutional layer for inputs with 2 spatial dimensions."""

    def __init__(
        self,
        n_out: int,
        kernel_shape: Tuple[int, int],
        activation: str,
        stride: int = 1,
        pad: str = "same",
        weight_init: str = "xavier_uniform",
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.pad = pad

        self.activation = initialize_activation(activation)
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int, int, int, int]) -> None:
        """Initialize all layer parameters and determine padding."""
        self.n_in = X_shape[3]

        W_shape = self.kernel_shape + (self.n_in,) + (self.n_out,)
        W = self.init_weights(W_shape)
        b = np.zeros((1, self.n_out))

        self.parameters = OrderedDict({"W": W, "b": b}) # DO NOT CHANGE THE KEYS
        self.cache = OrderedDict({"Z": [], "X": []}) # cache for backprop
        self.gradients = OrderedDict({"W": np.zeros_like(W), "b": np.zeros_like(b)}) # parameter gradients initialized to zero
                                                                                     # MUST HAVE THE SAME KEYS AS `self.parameters`

        if self.pad == "same":
            self.pad = ((W_shape[0] - 1) // 2, (W_shape[1] - 1) // 2)
        elif self.pad == "valid":
            self.pad = (0, 0)
        elif isinstance(self.pad, int):
            self.pad = (self.pad, self.pad)
        else:
            raise ValueError("Invalid Pad mode found in self.pad.")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass for convolutional layer. This layer convolves the input
        `X` with a filter of weights, adds a bias term, and applies an activation
        function to compute the output. This layer also supports padding and
        integer strides. Intermediates necessary for the backward pass are stored
        in the cache.

        Parameters
        ----------
        X  input with shape (batch_size, in_rows, in_cols, in_channels)

        Returns
        -------
        output feature maps with shape (batch_size, out_rows, out_cols, out_channels)
        """
        if self.n_in is None:
            self._init_parameters(X.shape)

        W = self.parameters["W"]
        b = self.parameters["b"]

        kernel_height, kernel_width, in_channels, out_channels = W.shape
        n_examples, in_rows, in_cols, in_channels = X.shape
        kernel_shape = (kernel_height, kernel_width)

        ### BEGIN YOUR CODE ###

        # implement a convolutional forward pass

        # cache any values required for backprop

        ### END YOUR CODE ###

        return out

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for conv layer. Computes the gradients of the output
        with respect to the input feature maps as well as the filter weights and
        biases.

        Parameters
        ----------
        dLdY  gradient of loss with respect to output of this layer
              shape (batch_size, out_rows, out_cols, out_channels)

        Returns
        -------
        gradient of the loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, in_channels)
        """
        ### BEGIN YOUR CODE ###

        # perform a backward pass

        ### END YOUR CODE ###

        return dX

class Pool2D(Layer):
    """Pooling layer, implements max and average pooling."""

    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        mode: str = "max",
        stride: int = 1,
        pad: Union[int, Literal["same"], Literal["valid"]] = 0,
    ) -> None:

        if type(kernel_shape) == int:
            kernel_shape = (kernel_shape, kernel_shape)

        self.kernel_shape = kernel_shape
        self.stride = stride

        if pad == "same":
            self.pad = ((kernel_shape[0] - 1) // 2, (kernel_shape[1] - 1) // 2)
        elif pad == "valid":
            self.pad = (0, 0)
        elif isinstance(pad, int):
            self.pad = (pad, pad)
        else:
            raise ValueError("Invalid Pad mode found in self.pad.")

        self.mode = mode

        if mode == "max":
            self.pool_fn = np.max
            self.arg_pool_fn = np.argmax
        elif mode == "average":
            self.pool_fn = np.mean

        self.cache = {
            "out_rows": [],
            "out_cols": [],
            "X_pad": [],
            "p": [],
            "pool_shape": [],
        }
        self.parameters = {}
        self.gradients = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: use the pooling function to aggregate local information
        in the input. This layer typically reduces the spatial dimensionality of
        the input while keeping the number of feature maps the same.

        As with all other layers, please make sure to cache the appropriate
        information for the backward pass.

        Parameters
        ----------
        X  input array of shape (batch_size, in_rows, in_cols, channels)

        Returns
        -------
        pooled array of shape (batch_size, out_rows, out_cols, channels)
        """
        ### BEGIN YOUR CODE ###

        # implement the forward pass

        # cache any values required for backprop

        ### END YOUR CODE ###

        return X_pool

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for pooling layer.

        Parameters
        ----------
        dLdY  gradient of loss with respect to the output of this layer
              shape (batch_size, out_rows, out_cols, channels)

        Returns
        -------
        gradient of loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, channels)
        """
        ### BEGIN YOUR CODE ###

        # perform a backward pass

        ### END YOUR CODE ###

        return gradX

class Flatten(Layer):
    """Flatten the input array."""

    def __init__(self, keep_dim: str = "first") -> None:
        super().__init__()

        self.keep_dim = keep_dim
        self._init_params()

    def _init_params(self):
        self.X = []
        self.gradients = {}
        self.parameters = {}
        self.cache = {"in_dims": []}

    def forward(self, X: np.ndarray, retain_derived: bool = True) -> np.ndarray:
        self.cache["in_dims"] = X.shape

        if self.keep_dim == -1:
            return X.flatten().reshape(1, -1)

        rs = (X.shape[0], -1) if self.keep_dim == "first" else (-1, X.shape[-1])
        return X.reshape(*rs)

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        in_dims = self.cache["in_dims"]
        gradX = dLdY.reshape(in_dims)
        return gradX
