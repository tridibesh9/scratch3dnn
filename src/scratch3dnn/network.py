import numpy as np
from .module import Module

class NeuralNet(Module):
  def __init__(self, *layers):
    self.layers = layers

  def forward(self, input_data):
    current_data = input_data
    for layer in self.layers:
        current_data = layer.forward(current_data)
    return current_data

  def backward(self, final_gradient):
    current_gradient = final_gradient
    for layer in reversed(self.layers):
        current_gradient = layer.backward(current_gradient)
    return current_gradient
  def get_params(self):
    params = []
    for i in self.layers:
      params.extend(i.get_params())
    return params