import numpy as np

def load_mnist(path='mnist.npz'):

  from keras.datasets import mnist # mnist digits 
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  return (x_train, y_train), (x_test, y_test) 


def load_fmnist_helper(path, kind='train'):
  import os
  import pickle
  from tensorflow.python.lib.io import file_io

  """Load fashion MNIST data from `path`"""
  labels_path = os.path.join(path, '%s-labels-idx1-ubyte.pkl' % kind)
  images_path = os.path.join(path, '%s-images-idx3-ubyte.pkl' % kind)
  
  file_stream = file_io.FileIO(labels_path, mode='r')
  labels  = pickle.load(file_stream) # syntax for python 2
  file_stream = file_io.FileIO(images_path, mode='r')
  images  = pickle.load(file_stream) # syntax for python 2
  
  '''
  with gzip.open(labels_path, 'rb') as lbpath:
      labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

  with gzip.open(images_path, 'rb') as imgpath:
      images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
  '''

  return images, labels

def load_fmnist(path):
  """Loads the fashion MNIST dataset.
  # Arguments
      path: path where to cache the dataset locally
  # Returns
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
  """
  x_train, y_train = load_fmnist_helper(path, kind='train')
  x_test, y_test = load_fmnist_helper(path, kind='t10k')  
  return (x_train, y_train), (x_test, y_test)

def run():
  local_path = '../../data/fashion-mnist/'
  (x_train, y_train), (x_test, y_test) = load_fmnist(path=local_path)
  print(type(x_train), np.shape(x_train))
  print(type(y_train), np.shape(y_train))
  visualize_data(x_test, y_test)

def visualize_data(x, y, n_imgs = 100):
  import matplotlib.pyplot as plt
  n_rows, n_cols = 10, 10
  f, axes = plt.subplots(n_rows, n_cols, sharey=True, figsize=(12,12))    
  for j in range(n_rows):
    for i in range(n_cols):
      axes[j, i].imshow( np.reshape(x[j+i*n_rows], (28,28)), cmap='Greys' ) 
      axes[j, i].set_xticklabels([])
      axes[j, i].set_yticklabels([])
      axes[j, i].set_title(y[j+i*n_rows])
  fig_name = 'data.pdf'
  f.savefig(fig_name) 
  

if __name__ == '__main__':
  run()
