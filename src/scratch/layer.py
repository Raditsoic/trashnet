import numpy as np

class Layer:
    def __init__(self):
        self.last_input = None
        
    def forward(self, X):
        raise NotImplementedError
        
    def backward(self, dY):
        raise NotImplementedError
        
    def get_params(self):
        return {}
        
    def set_params(self, params):
        pass


class Dense(Layer):
    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = num_neurons
        self.weights = np.random.randn(num_neurons) / num_neurons
        self.biases = np.zeros(num_neurons)
        self.d_weights = None
        self.d_biases = None
        
    def forward(self, X):
        return np.dot(X, self.weights) + self.biases
        
    def backward(self, dY):
        self.d_weights = np.dot(self.last_input.T, dY)
        self.d_biases = np.sum(dY, axis=0)
        return np.dot(dY, self.weights.T)
        
    def get_params(self):
        return {
            'weights': self.weights,
            'biases': self.biases
        }
        
    def set_params(self, params):
        self.weights = params['weights']
        self.biases = params['biases']

class Flatten(Layer):
    def forward(self, X):
        self.last_input_shape = X.shape
        return X.reshape(X.shape[0], -1)
        
    def backward(self, dY):
        return dY.reshape(self.last_input_shape)

class Activation(Layer):
    def __init__(self, activation_function):
        super().__init__()
        self.activation_function = activation_function
        
    def forward(self, X):
        if self.activation_function == 'relu':
            return np.maximum(0, X)
        
    def backward(self, dY):
        if self.activation_function == 'relu':
            return dY * (self.last_input > 0)
   

class Dropout(Layer):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate
        self.mask = None
        
    def forward(self, X, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.rate, X.shape)
            return X * self.mask / (1 - self.rate)
        return X
        
    def backward(self, dY):
        return dY * self.mask / (1 - self.rate)


# Convolutions and pooling
class Convolutional(Layer):
    def __init__(self, num_filters, filter_size, stride=1, padding=0):
        super().__init__()
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / (filter_size * filter_size)
        self.biases = np.zeros((num_filters, 1))
        self.d_filters = None
        self.d_biases = None
        
    def _pad_input(self, X):
        if self.padding == 0:
            return X
        return np.pad(X, ((0,0), (self.padding,self.padding), (self.padding,self.padding)), mode='constant')
        
    def forward(self, X):
        self.last_input = X
        batch_size, height, width = X.shape
        
        # Apply padding
        X_padded = self._pad_input(X)
        padded_height, padded_width = X_padded.shape[1:]
        
        # Calculate output dimensions
        out_height = (padded_height - self.filter_size) // self.stride + 1
        out_width = (padded_width - self.filter_size) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, self.num_filters, out_height, out_width))
        
        # Perform convolution
        for i in range(0, out_height):
            for j in range(0, out_width):
                h_start = i * self.stride
                h_end = h_start + self.filter_size
                w_start = j * self.stride
                w_end = w_start + self.filter_size
                
                receptive_field = X_padded[:, h_start:h_end, w_start:w_end]
                
                for k in range(self.num_filters):
                    output[:, k, i, j] = np.sum(
                        receptive_field * self.filters[k], axis=(1,2)
                    ) + self.biases[k]
        
        return output
        
    def backward(self, dY):
        batch_size = self.last_input.shape[0]
        X_padded = self._pad_input(self.last_input)
        
        # Initialize gradients
        self.d_filters = np.zeros_like(self.filters)
        self.d_biases = np.zeros_like(self.biases)
        dX_padded = np.zeros_like(X_padded)
        
        # Calculate gradients
        for i in range(dY.shape[2]):
            for j in range(dY.shape[3]):
                h_start = i * self.stride
                h_end = h_start + self.filter_size
                w_start = j * self.stride
                w_end = w_start + self.filter_size
                
                receptive_field = X_padded[:, h_start:h_end, w_start:w_end]
                
                for k in range(self.num_filters):
                    self.d_filters[k] += np.sum(
                        receptive_field * dY[:, k, i, j][:, np.newaxis, np.newaxis],
                        axis=0
                    )
                    self.d_biases[k] += np.sum(dY[:, k, i, j])
                    dX_padded[:, h_start:h_end, w_start:w_end] += (
                        self.filters[k] * dY[:, k, i, j][:, np.newaxis, np.newaxis]
                    )
        
        # Remove padding from dX if necessary
        if self.padding == 0:
            return dX_padded
        return dX_padded[:, self.padding:-self.padding, self.padding:-self.padding]

class MaxPooling(Layer):
    def __init__(self, pool_size, stride):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None
        
    def forward(self, X):
        self.last_input = X
        batch_size, height, width = X.shape
        
        # Calculate output dimensions
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        
        # Initialize output and cache for max positions
        output = np.zeros((batch_size, out_height, out_width))
        self.cache = np.zeros_like(X)
        
        # Perform max pooling
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                pool_region = X[:, h_start:h_end, w_start:w_end]
                output[:, i, j] = np.max(pool_region, axis=(1,2))
                
                # Store positions of max values for backprop
                max_mask = (pool_region == output[:, i, j][:, np.newaxis, np.newaxis])
                self.cache[:, h_start:h_end, w_start:w_end] += max_mask
        
        return output
        
    def backward(self, dY):
        dX = np.zeros_like(self.last_input)
        out_height = dY.shape[1]
        out_width = dY.shape[2]
        
        # Distribute gradients to max positions
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                dX[:, h_start:h_end, w_start:w_end] += (
                    self.cache[:, h_start:h_end, w_start:w_end] * 
                    dY[:, i, j][:, np.newaxis, np.newaxis]
                )
        
        return dX

class BatchNormalization(Layer):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = np.ones((1,))
        self.beta = np.zeros((1,))
        self.running_mean = None
        self.running_var = None
        self.cache = None
        
    def forward(self, X, training=True):
        if self.running_mean is None:
            self.running_mean = np.zeros(X.shape[1:])
            self.running_var = np.zeros(X.shape[1:])
            
        if training:
            mean = np.mean(X, axis=0)
            var = np.var(X, axis=0)
            
            # Update running statistics
            self.running_mean = 0.9 * self.running_mean + 0.1 * mean
            self.running_var = 0.9 * self.running_var + 0.1 * var
        else:
            mean = self.running_mean
            var = self.running_var
            
        # Normalize
        X_norm = (X - mean) / np.sqrt(var + self.epsilon)
        out = self.gamma * X_norm + self.beta
        
        # Cache variables for backward pass
        self.cache = (X, X_norm, mean, var)
        
        return out
        
    def backward(self, dY):
        X, X_norm, mean, var = self.cache
        N = X.shape[0]
        
        # Gradient with respect to beta
        self.d_beta = np.sum(dY, axis=0)
        
        # Gradient with respect to gamma
        self.d_gamma = np.sum(dY * X_norm, axis=0)
        
        # Gradient with respect to X
        dX_norm = dY * self.gamma
        dvar = np.sum(dX_norm * (X - mean) * -0.5 * (var + self.epsilon)**(-1.5), axis=0)
        dmean = np.sum(dX_norm * -1/np.sqrt(var + self.epsilon), axis=0)
        dX = dX_norm / np.sqrt(var + self.epsilon) + dvar * 2 * (X - mean) / N + dmean / N
        
        return dX

