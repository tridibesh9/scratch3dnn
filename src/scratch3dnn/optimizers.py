import numpy as np
from .module import Module

class SGDOptimizer:
  def __init__(self, parameters, learning_rate = 0.001):
    self.lr = learning_rate
    self.params = parameters
  def step(self):
    for param,grad in self.params:
      param-=self.lr*grad
  def zero_grad(self):
    for param, grad in self.params:
          grad.fill(0.0)