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
        
    def forward(self, X):
        # Implementation of convolution operation
        pass
        
    def backward(self, dY):
        # Implementation of backward pass
        pass
        
    def get_params(self):
        return {
            'filters': self.filters,
            'biases': self.biases
        }
        
    def set_params(self, params):
        self.filters = params['filters']
        self.biases = params['biases']

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

class MaxPooling(Layer):
    def __init__(self, pool_size, stride):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None
        
    def forward(self, X):
        # Implementation of max pooling
        pass
        
    def backward(self, dY):
        # Implementation of backward pass
        pass

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
        # Add other activation functions as needed
        
    def backward(self, dY):
        if self.activation_function == 'relu':
            return dY * (self.last_input > 0)
        # Add other activation functions as needed

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
        # Implementation of batch normalization
        pass
        
    def backward(self, dY):
        # Implementation of backward pass
        pass

class Model:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, X, training=True):
        for layer in self.layers:
            if isinstance(layer, (Dropout, BatchNormalization)):
                X = layer.forward(X, training)
            else:
                X = layer.forward(X)
        return X

    def backward(self, dY):
        for layer in reversed(self.layers):
            dY = layer.backward(dY)
            
    def get_params(self):
        params = {}
        for i, layer in enumerate(self.layers):
            layer_params = layer.get_params()
            if layer_params:
                params[f'layer_{i}'] = layer_params
        return params
        
    def set_params(self, params):
        for i, layer in enumerate(self.layers):
            if f'layer_{i}' in params:
                layer.set_params(params[f'layer_{i}'])

    def train(self, X, Y, epochs, batch_size):
        for epoch in range(epochs):
            total_loss = 0
            total_acc = 0
            num_batches = 0
            
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size]
                Y_batch = Y[i:i+batch_size]
                
                # Forward pass
                Y_pred = self.forward(X_batch, training=True)
                
                # Calculate loss and accuracy
                loss = self.loss.forward(Y_pred, Y_batch)
                acc = np.mean(np.argmax(Y_pred, axis=1) == np.argmax(Y_batch, axis=1))
                
                # Backward pass
                dY = self.loss.backward(Y_pred, Y_batch)
                self.backward(dY)
                
                # Update parameters using optimizer
                self.optimizer.update(self.get_params())
                
                total_loss += loss
                total_acc += acc
                num_batches += 1
            
            # Print epoch statistics
            avg_loss = total_loss / num_batches
            avg_acc = total_acc / num_batches
            print(f'Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f} Accuracy: {avg_acc:.4f}')