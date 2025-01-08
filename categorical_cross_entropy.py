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
        
class Softmax_Activation():
    def forwad(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_losses= self.forward(output, y)     
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_actual):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        if len(y_actual.shape) == 1:
            correct_confidence = y_pred_clipped[range(samples), y_actual]
        
        elif len(y_actual.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped*y_actual, axis=1)
         
        negative_log_likelihood = -np.log(correct_confidence)
        return negative_log_likelihood               
      
layer1 = DenseLayer(2, 5)
activation1 = ActivationFunction_ReLU()         

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


loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output, y)

print('Loss: ',loss)