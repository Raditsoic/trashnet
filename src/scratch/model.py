from . import Dropout, BatchNormalization

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