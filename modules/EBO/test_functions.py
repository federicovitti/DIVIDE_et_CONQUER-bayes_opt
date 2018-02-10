import numpy as np
from representation import DenseL1Kernel
from gp import DenseKernelGP
import os 
import functions as fn


class sampled_rosenbrock_func(object):
  def __init__(self, x_range, dx):
    self.dx = dx
    self.x_range = x_range
    self.get_max()
  def get_max(self):
    self.fmax = 0.

  def __call__(self, x):
    return fn.meno_rosenbrock(x)



class sampled_branin_func(object):
  def __init__(self, x_range, dx):
    self.dx = dx
    self.x_range = x_range
    self.get_max()
  def get_max(self):
    self.fmax = 0.397887

  def __call__(self, x):
    return fn.meno_branin(x)



class sampled_hartmann3_func(object):
  def __init__(self, x_range, dx):
    self.dx = dx
    self.x_range = x_range
    self.get_max()
  def get_max(self):
    self.fmax = -3.86278

  def __call__(self, x):
    return fn.meno_hartmann3(x)



class sampled_hartmann6_func(object):
  def __init__(self, x_range, dx):
    self.dx = dx
    self.x_range = x_range
    self.get_max()
  def get_max(self):
    self.fmax = -3.32237

  def __call__(self, x):
    return fn.meno_hartmann6(x)



class sampled_ackley_func(object):
  def __init__(self, x_range, dx):
    self.dx = dx
    self.x_range = x_range
    self.get_max()
  def get_max(self):
    self.fmax = 0.

  def __call__(self, x):
    return fn.meno_ackley(x)
