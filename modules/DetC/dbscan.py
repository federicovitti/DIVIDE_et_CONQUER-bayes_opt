import numpy as np
from sklearn.cluster import DBSCAN

class Node(object):
  def __init__(self, X, y, x_range):
    self.X = X
    self.y = y
    self.x_range = x_range
    self.get_vol()
    self.left = None
    self.right = None

  def get_vol(self):
    self.volumn =  np.exp(np.log((self.x_range[1]-self.x_range[0]).sum()))

  def delete_data(self):
    self.X = None
    self.y = None

class Decomposition(object):
  def __init__(self, X, y, x_range):
    self.X = X
    self.y = y
    self.x_range = x_range
    self.root = Node(X, y, x_range)
    self.leaves = None

  def building(self, min_leaf_size=5):
    leaves = [self.root]
    dbscan = DBSCAN(algorithm = 'kd_tree', eps=1, leaf_size=min_leaf_size)
    cutree = dbscan.fit(self.X)
    for i in np.unique(cutree.labels_):
      index = np.asarray(np.where(cutree.labels_==i))[0]
      X = self.X[index]
      y = self.y[index]
      x_range0 = np.array([X[:,0].min()])
      x_range1 = np.array([X[:,0].max()])
      for j in range(1, X.shape[1]):
        x_range0 = np.append(x_range0, X[:,j].min())
        x_range1 = np.append(x_range1, X[:,j].max())
      x_range = np.vstack((x_range0, x_range1))
      if (x_range[1]-x_range[0]).sum() > 0:
        partition = Node(X, y, x_range)
        leaves += [partition]
    return leaves