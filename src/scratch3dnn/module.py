import numpy as np 
class Module:
  def forward(self,input_data):
    raise NotImplementedError
  def backward(self, gradient):
    raise NotImplementedError
  def get_params(self):
    return []