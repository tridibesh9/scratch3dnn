import numpy as np
from .module import Module

class Layer(Module):
  def __init__(self, input_size, output_size):
    self.w = np.random.randn(input_size, output_size)*0.001
    self.b = np.zeros((1,output_size))
    self.w_grad = np.zeros_like(self.w)
    self.b_grad = np.zeros_like(self.b)
  def forward(self, input_data):
    input_data = np.array(input_data)
    self.input_data = input_data
    output = self.input_data@self.w + self.b
    return output
  def backward(self, input_gradient):
    output_gradient = input_gradient@self.w.T
    self.w_grad += self.input_data.T@input_gradient
    self.b_grad += np.sum(input_gradient, axis=0, keepdims=True)
    return output_gradient
  def get_params(self):
    return [(self.w, self.w_grad), (self.b, self.b_grad)]