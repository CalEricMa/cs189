### Activation Function Implementations:

Implementation of `activations.Linear`:

```python
class Linear(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for f(z) = z.
        
        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        return Z

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for f(z) = z.
        
        Parameters
        ----------
        Z   input to `forward` method
        dY  gradient of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        gradient of loss w.r.t. input of this layer
        """
        return dY

```

Implementation of `activations.Sigmoid`:

```python
class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for sigmoid function:
        f(z) = 1 / (1 + exp(-z))
        
        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        ### YOUR CODE HERE ###
        self.output = 1 / (1 + np.exp(-Z))
        return self.output

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for sigmoid.
        
        Parameters
        ----------
        Z   input to `forward` method
        dY  gradient of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        gradient of loss w.r.t. input of this layer
        """
        ### YOUR CODE HERE ###
        sigmoid_Z = 1 / (1 + np.exp(-Z))
        return dY * sigmoid_Z * (1 - sigmoid_Z)

```

Implementation of `activations.ReLU`:

```python
class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for relu activation:
        f(z) = z if z >= 0
               0 otherwise
        
        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        ### YOUR CODE HERE ###
        Y = np.maximum(0, Z)
        return Y

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for relu activation.
        
        Parameters
        ----------
        Z   input to `forward` method
        dY  gradient of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        gradient of loss w.r.t. input of this layer
        """
        ### YOUR CODE HERE ###
        indicator = (Z > 0).astype(Z.dtype)
        dLdZ = dY * indicator
        return dLdZ

```

Implementation of `activations.SoftMax`:

```python
class SoftMax(Activation):
    def __init__(self):
        super().__init__()
        self.cache = {}

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """Forward pass for softmax activation.
        Hint: The naive implementation might not be numerically stable.
        
        Parameters
        ----------
        Z  input pre-activations (any shape)

        Returns
        -------
        f(z) as described above applied elementwise to `Z`
        """
        ### YOUR CODE HERE ###

        shift = Z - np.max(Z, axis=1, keepdims=True)   # shape (B,K)
        exp_shift = np.exp(shift)                      # shape (B,K)
        sums = np.sum(exp_shift, axis=1, keepdims=True)  # shape (B,1)
        Y = exp_shift / sums                           # shape (B,K)

        self.cache["Y"] = Y
        return Y

    def backward(self, Z: np.ndarray, dY: np.ndarray) -> np.ndarray:
        """Backward pass for softmax activation.
        
        Parameters
        ----------
        Z   input to `forward` method
        dY  gradient of loss w.r.t. the output of this layer
            same shape as `Z`

        Returns
        -------
        gradient of loss w.r.t. input of this layer
        """
        ### YOUR CODE HERE ###

        
        Y = self.cache["Y"]                            # shape (B,K)

        dot = np.sum(dY * Y, axis=1, keepdims=True)    # shape (B,1)
        dZ = Y * (dY - dot)                            # shape (B,K)
        return dZ

```


### Layer Implementations:

Implementation of `layers.FullyConnected`:

```python
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

```

Implementation of `layers.Pool2D`:

```python
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


        B, H, W, C = X.shape
        kh, kw   = self.kernel_shape
        ph, pw   = self.pad
        s        = self.stride

        X_pad = np.pad(
            X,
            ((0,0),(ph,ph),(pw,pw),(0,0)),
            mode="constant",
            constant_values=0
        )

        H_out = (H + 2*ph - kh)//s + 1
        W_out = (W + 2*pw - kw)//s + 1

        patches = []
        for i in range(kh):
            for j in range(kw):
                patch = X_pad[
                    :,
                    i : i + H_out*s : s,
                    j : j + W_out*s : s,
                    :
                ]
                patches.append(patch)
        patches = np.stack(patches, axis=0)
        P = patches.shape[0]

        if self.mode == "max":
            out = patches.max(axis=0)        # (B,H_out,W_out,C)
            argmax = patches.argmax(axis=0)  

            self.cache["argmax"] = argmax
        else:
            out = patches.mean(axis=0)       # (B,H_out,W_out,C)

        self.cache["X_shape"]   = (B, H, W, C)
        self.cache["X_pad"]     = X_pad
        self.cache["patches"]   = patches.shape  
        self.cache["pool_shape"]= (kh, kw)
        self.cache["stride"]    = s
        self.cache["pad"]       = (ph, pw)

        return out

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
        B, H, W, C = self.cache["X_shape"]
        ph, pw     = self.cache["pad"]
        s          = self.cache["stride"]
        kh, kw     = self.cache["pool_shape"]
        patches_sh = self.cache["patches"]    
        P, _, H_out, W_out, _ = patches_sh

        X_pad = self.cache["X_pad"]
        gradX = np.zeros_like(X_pad)

        if self.mode == "max":
            argmax = self.cache["argmax"]       # (B,H_out,W_out,C)
            p = 0
            for i in range(kh):
                for j in range(kw):
                    mask = (argmax == p)        # (B,H_out,W_out,C)
                    dpatch = dLdY * mask        # (B,H_out,W_out,C)
                    gradX[
                        :,
                        i : i + H_out*s : s,
                        j : j + W_out*s : s,
                        :
                    ] += dpatch
                    p += 1

        else:  # average pooling
            dpatch = dLdY / float(P)            # (B,H_out,W_out,C)
            for i in range(kh):
                for j in range(kw):
                    gradX[
                        :,
                        i : i + H_out*s : s,
                        j : j + W_out*s : s,
                        :
                    ] += dpatch

        dX = gradX[:, ph:ph+H, pw:pw+W, :]

        ### END YOUR CODE ###

        return dX

```

Implementation of `layers.Conv2D.__init__`:

```python
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

```

Implementation of `layers.Conv2D._init_parameters`:

```python
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

```

Implementation of `layers.Conv2D.forward`:

```python
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

        ph, pw = self.pad            
        s = self.stride

        B, H_in, W_in, Cin = X.shape
        kh, kw, _, Cout   = W.shape
        X_pad = np.pad(
            X,
            ((0,0),               # batch dim
             (ph,ph),             # height
             (pw,pw),             # width
             (0,0)),              # channels
            mode="constant",
            constant_values=0
        )

        H_out = (H_in + 2*ph - kh)//s + 1
        W_out = (W_in + 2*pw - kw)//s + 1

        Z = np.zeros((B, H_out, W_out, Cout), dtype=X.dtype)

        for i in range(kh):
            for j in range(kw):
                patch = X_pad[
                    :,
                    i : i + H_out*s : s,
                    j : j + W_out*s : s,
                    :
                ]
                Z += np.einsum("bhwc,cf->bhwf", patch, W[i, j, :, :])
        Z += b                       
        Y = self.activation.forward(Z)
        self.cache["X_pad"] = X_pad
        self.cache["Z"]     = Z


        ### END YOUR CODE ###

        return Y

```

Implementation of `layers.Conv2D.backward`:

```python
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

```


### Loss Function Implementations:

Implementation of `losses.CrossEntropy`:

```python
class CrossEntropy(Loss):
    """Cross entropy loss function."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, Y: np.ndarray, Y_hat: np.ndarray) -> float:
        return self.forward(Y, Y_hat)

    def forward(self, Y: np.ndarray, Y_hat: np.ndarray) -> float:
        """Computes the loss for predictions `Y_hat` given one-hot encoded labels
        `Y`.

        Parameters
        ----------
        Y      one-hot encoded labels of shape (batch_size, num_classes)
        Y_hat  model predictions in range (0, 1) of shape (batch_size, num_classes)

        Returns
        -------
        a single float representing the loss
        """
        ### YOUR CODE HERE ###


        B = Y.shape[0]
        eps = 1e-12
        Y_hat = np.clip(Y_hat, eps, 1.0 - eps)
        loss = - np.sum(Y * np.log(Y_hat)) / B
        return loss

    def backward(self, Y: np.ndarray, Y_hat: np.ndarray) -> np.ndarray:
        """Backward pass of cross-entropy loss.
        NOTE: This is correct ONLY when the loss function is SoftMax.

        Parameters
        ----------
        Y      one-hot encoded labels of shape (batch_size, num_classes)
        Y_hat  model predictions in range (0, 1) of shape (batch_size, num_classes)

        Returns
        -------
        the gradient of the cross-entropy loss with respect to the vector of
        predictions, `Y_hat`
        """
        ### YOUR CODE HERE ###

        
        B = Y.shape[0]
        eps = 1e-12
        Y_hat = np.clip(Y_hat, eps, 1.0 - eps)
        grad = - (Y / Y_hat) / B
        return grad

```


### Model Implementations:

Implementation of `models.NeuralNetwork.forward`:

```python
    def forward(self, X: np.ndarray) -> np.ndarray:
        """One forward pass through all the layers of the neural network.

        Parameters
        ----------
        X  design matrix whose must match the input shape required by the
           first layer

        Returns
        -------
        forward pass output, matches the shape of the output of the last layer
        """
        ### YOUR CODE HERE ###
        # Iterate through the network's layers.


        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

```

Implementation of `models.NeuralNetwork.backward`:

```python
    def backward(self, target: np.ndarray, out: np.ndarray) -> float:
        """One backward pass through all the layers of the neural network.
        During this phase we calculate the gradients of the loss with respect to
        each of the parameters of the entire neural network. Most of the heavy
        lifting is done by the `backward` methods of the layers, so this method
        should be relatively simple. Also make sure to compute the loss in this
        method and NOT in `self.forward`.

        Note: Both input arrays have the same shape.

        Parameters
        ----------
        target  the targets we are trying to fit to (e.g., training labels)
        out     the predictions of the model on training data

        Returns
        -------
        the loss of the model given the training inputs and targets
        """
        ### YOUR CODE HERE ###
        # Compute the loss.
        # Backpropagate through the network's layers.

        loss = self.loss.forward(target, out)
        dLdY = self.loss.backward(target, out)
        grad = dLdY
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return loss

```

Implementation of `models.NeuralNetwork.predict`:

```python
    def predict(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, float]:
        """Make a forward and backward pass to calculate the predictions and
        loss of the neural network on the given data.

        Parameters
        ----------
        X  input features
        Y  targets (same length as `X`)

        Returns
        -------
        a tuple of the prediction and loss
        """
        ### YOUR CODE HERE ###
        # Do a forward pass. Maybe use a function you already wrote?
        # Get the loss. Remember that the `backward` function returns the loss.

        Y_hat = self.forward(X)
        loss  = self.loss.forward(Y, Y_hat)
        return Y_hat, loss

```

