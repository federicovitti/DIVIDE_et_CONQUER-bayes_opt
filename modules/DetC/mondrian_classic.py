import numpy as np

def sample_categorical(prob):
  prob = prob / np.sum(prob)
  return np.random.choice(len(prob), p=prob)

class Node(object):
  def __init__(self, X, y, x_range):
    self.X = X
    self.y = y
    self.totlen = (x_range[1]-x_range[0]).sum()
    self.x_range = x_range
    self.get_vol()
    assert self.totlen > 0, 'Node is empty. Totoal length of the search space is 0.'
    self.datasize = y.shape[0]
    self.left = None
    self.right = None
    self.epsilon = 0. # overlap of the leaves
    #mindata = 100

  def get_vol(self):
    self.volumn = np.exp(np.log((self.x_range[1]-self.x_range[0])).sum())
    
  def partition(self):
    prob = self.x_range[1]-self.x_range[0]
    assert prob.dtype == float, 'Forgot to set x_range to be float?'
    d = np.random.choice(range(len(prob)))
    cut = np.random.uniform(self.x_range[0,d], self.x_range[1,d])
    leftinds = np.where(self.X[:,d]<=cut + self.epsilon)
    rightinds = np.where(self.X[:,d]>=cut - self.epsilon)
    left_range, right_range = self.x_range.copy(), self.x_range.copy()
    left_range[1, d] = cut
    right_range[0, d] = cut
    self.left = Node(self.X[leftinds], self.y[leftinds], left_range)
    self.right = Node(self.X[rightinds], self.y[rightinds], right_range)
    return self.left, self.right

  def delete_data(self):
    self.X = None
    self.y = None

class Decomposition(object):
  def __init__(self, X, y, x_range, poolsize=10):
    self.X = X
    self.y = y
    self.x_range = x_range
    self.root = Node(X, y, x_range)
    self.poolsize = poolsize
    self.leaves = None

  def building(self, min_leaf_size=5):
    leaves = [self.root]
    flag = True
    while flag:
      if len(leaves) >= self.poolsize:
        break
      prob = np.array([[node.totlen, node.datasize] \
                      for node in leaves])
      mask = np.maximum(prob[:,1]-min_leaf_size, 0)
      if mask.sum() == 0:
        break
      prob = mask*prob[:,0]
      nodeidx = sample_categorical(prob)      
      chosen_leaf = leaves[nodeidx]
      left, right = chosen_leaf.partition()
      leaves[nodeidx] = left
      leaves += [right]
      chosen_leaf.delete_data()
    self.leaves = leaves
    return leaves
  