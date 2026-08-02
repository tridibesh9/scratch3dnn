import numpy as np
from .module import Module

class MSELoss(Module):
  def forward(self, prediction, target):
    self.prediction = prediction
    self.target = target
    error = 1/2*np.mean(((prediction - target))**2)
    return error
  def backward(self):
    n = self.prediction.shape[0]
    initial_gradient = (1/n)*(self.prediction- self.target)
    return initial_gradient