import numpy as np

class Optimizer:
    def __init__(self):
        self.iterations = 0
    
    def update(self, params):
        raise NotImplementedError

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
        
    def initialize_moments(self, params):
        """Initialize moment estimates for each parameter if not already initialized"""
        for layer_name, layer_params in params.items():
            if layer_name not in self.m:
                self.m[layer_name] = {}
                self.v[layer_name] = {}
                for param_name, param in layer_params.items():
                    self.m[layer_name][param_name] = np.zeros_like(param)
                    self.v[layer_name][param_name] = np.zeros_like(param)

    def update(self, params):
        """Update parameters using Adam optimization"""
        if not self.m:  # Initialize moments if this is the first update
            self.initialize_moments(params)
        
        self.iterations += 1
        
        # Bias correction terms
        m_correction = 1 / (1 - self.beta1 ** self.iterations)
        v_correction = 1 / (1 - self.beta2 ** self.iterations)
        
        # Update each layer's parameters
        for layer_name, layer_params in params.items():
            layer_updates = {}
            
            for param_name, param in layer_params.items():
                grad = param.d  # Assuming each parameter has a .d attribute for its gradient
                
                # Update moment estimates
                self.m[layer_name][param_name] = (self.beta1 * self.m[layer_name][param_name] + 
                                                (1 - self.beta1) * grad)
                self.v[layer_name][param_name] = (self.beta2 * self.v[layer_name][param_name] + 
                                                (1 - self.beta2) * np.square(grad))
                
                # Compute bias-corrected moment estimates
                m_hat = self.m[layer_name][param_name] * m_correction
                v_hat = self.v[layer_name][param_name] * v_correction
                
                # Update parameters
                update = (self.learning_rate * m_hat / 
                         (np.sqrt(v_hat) + self.epsilon))
                param -= update
                
                layer_updates[param_name] = param
                
            params[layer_name] = layer_updates
            
        return params