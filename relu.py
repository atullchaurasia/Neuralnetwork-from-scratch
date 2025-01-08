import numpy as np
import nnfs
from nnfs.datasets import spiral_data  

nnfs.init()
np.random.seed(0)

X = [[4.5, -6.7, 7.6, 2.1],
     [2.0, 4.5, -6.7, 4.2],
     [-4.5, 2.1, 9.7, 8.7]]

X,y = spiral_data(100, 3)

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


layer1.forward_propagation(X)
#print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)