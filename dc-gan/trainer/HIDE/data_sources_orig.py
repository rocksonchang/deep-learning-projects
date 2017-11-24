import numpy as np
import pickle
import os

# Fashion mnist data globals
FMNIST_REPO = 'https://github.com/zalandoresearch/fashion-mnist/blob/master/data/fashion/'
FMNIST_TRAIN_DATA = 'train-images-idx3-ubyte'
FMNIST_TRAIN_LBLS = 'train-labels-idx1-ubyte'
FMNIST_TEST_DATA = 't10k-images-idx3-ubyte'
FMNIST_TEST_LBLS = 't10k-labels-idx1-ubyte', 
FMNIST_FILES = [FMNIST_TRAIN_DATA, FMNIST_TRAIN_LBLS, FMNIST_TEST_DATA, FMNIST_TEST_LBLS]

def load_mnist(path='mnist.npz'):

  from keras.datasets import mnist # mnist digits 
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  return (x_train, y_train), (x_test, y_test) 


'''
def load_fmnist_helper(path, kind='train'):
  from tensorflow.python.lib.io import file_io

  """Load fashion MNIST data from `path`"""
  labels_path = os.path.join(path, '%s-labels-idx1-ubyte.pkl' % kind)
  images_path = os.path.join(path, '%s-images-idx3-ubyte.pkl' % kind)
  
  file_stream = file_io.FileIO(labels_path, mode='r')
  labels  = pickle.load(file_stream) # syntax for python 2
  file_stream = file_io.FileIO(images_path, mode='r')
  images  = pickle.load(file_stream) # syntax for python 2

  return images, labels
'''

def load_fmnist(path):
  """Loads the fashion MNIST dataset.
  # Arguments
      path: path where to cache the dataset locally
  # Returns
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
  """
  
  download_fmnist()
  pickle_fmnist()
  data = unpack_fmnist()
  
  x_train, y_train = data[FMNIST_TRAIN_DATA], data[FMNIST_TRAIN_LBLS]
  x_test, y_test = data[FMNIST_TEST_DATA], data[FMNIST_TEST_LBLS]

  #x_train, y_train = load_fmnist_helper(path, kind='train')
  #x_test, y_test = load_fmnist_helper(path, kind='t10k')  
  return (x_train, y_train), (x_test, y_test)

def unpack_fmnist(path):
  from tensorflow.python.lib.io import file_io
  for file in FMNIST_FILES:
    source_path = os.path.join(path, file + '.pkl')
    file_stream = file_io.FileIO(source_path, mode='r')
    data[file] = pickle.load(file_stream) # syntax for python 2
  return data


def download_fmnist():  
  import urllib    
  
  print('Downloading zip files from {}'.format(FMNIST_REPO))
  for file in FMNIST_FILES:    
    opener = urllib.URLopener()    
    target_path = os.path.join(path, file + '.gz')
    if not os.path.exists(target_path):
      print('{}: downloading'.format(file))
      opener.retrieve(FMNIST_REPO + '/' + file, target_path)
    else print('{}: already downloaded'.format(file))

def pickle_fmnist():
  import gzip

  for file in FMNIST_FILES:
    source_path = os.path.join(path, file + '.pkl')
    target_path = os.path.join(path, file + '.gz')
    if not os.path.exists(target_path):
      print('{}: pickling file'.format(file))
      with gzip.open(source_path, 'rb') as gz:
        if file[-4:]='LBLS':
          data = np.frombuffer(gz.read(), dtype=np.uint8, offset=8)
        elif file[-4:]='DATA':
          data = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        pickle.dump(data, open(target_path, 'wb'), -1)
    else:
      print('{}: already pickled'.format(file))

    

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

def run():
  local_path = 'fashion-mnist/'
  (x_train, y_train), (x_test, y_test) = load_fmnist(path=local_path)
  print(type(x_train), np.shape(x_train))
  print(type(y_train), np.shape(y_train))
  visualize_data(x_test, y_test)

if __name__ == '__main__':
  run()
