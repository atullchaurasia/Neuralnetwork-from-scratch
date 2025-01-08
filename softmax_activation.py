import numpy as np
import nnfs
from nnfs.datasets import spiral_data  

nnfs.init()
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
        
        
class ActivationFunction_ReLU():
    def forward(self, inputs):
        self.output = np.maximum(0, inputs) 
      
layer1 = DenseLayer(2, 5)
activation1 = ActivationFunction_ReLU() 

class Softmax_Activation():
    def forwad(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

X,y = spiral_data(samples = 100, classes = 3)

dense1 = DenseLayer(2,3)
activation1 = ActivationFunction_ReLU()

dense2 = DenseLayer(3, 3)
activation2 = Softmax_Activation()

dense1.forward_propagation(X)
activation1.forward(dense1.output)

dense2.forward_propagation(activation1.output)
activation2.forwad(dense2.output)

print(activation2.output[:5])
