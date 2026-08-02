import numpy as np
from .module import Module

class Relu(Module):
  def forward(self, input_data):
    self.input_data = input_data
    output = np.maximum(0.0,input_data)
    self.output = output
    return output
  def backward(self, input_gradient):
    curr_grad = np.where(self.output > 0, 1, 0)
    output_gradient = input_gradient*curr_grad
    return output_gradient

class Sigmoid(Module):
  def forward(self, input_data):
    self.input_data = input_data
    output = 1/(1+np.exp(-input_data))
    self.output = output
    return output
  def backward(self, input_gradient):
    curr_grad = self.output*(1-self.output)
    output_gradient = input_gradient*curr_grad
    return output_gradient