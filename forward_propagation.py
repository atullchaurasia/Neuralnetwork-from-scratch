import numpy as np

np.random.seed(0)

X = [[4.5, -6.7, 7.6, 2.1],
     [2.0, 4.5, -6.7, 4.2],
     [-4.5, 2.1, 9.7, 8.7]]

class DenseLayer():
    def __init__(self, n_inputs, n_neurons ):
        self.weights = 0.10* np.random.randn(n_inputs, n_neurons) ##random weights 
        self.bias = np.zeros((1, n_neurons))
    
    def forward_propagation(self, inputs):
        self.output = np.dot(inputs, self.weights ) + self.bias
      
      
layer1 = DenseLayer(4, 5) 
layer2 = DenseLayer(5, 2)

layer1.forward_propagation(X)
print(layer1.output)

layer2.forward_propagation(layer1.output)   
print(layer2.output)   
