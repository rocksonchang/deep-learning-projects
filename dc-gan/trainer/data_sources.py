'''
Helper functions to load datasets

Available datasets: 
  
  MNIST digits dataset - 28x28x1 (length, width, depth) - 60k train, 10k test
  source: http://yann.lecun.com/exdb/mnist/
  
  MNIST fashion dataset - 28x28x1 (length, width, depth) - 60k train, 10k test
  source: https://github.com/zalandoresearch/fashion-mnist
'''

import math
import numpy as np
import matplotlib.pyplot as plt
import gzip 
import pickle
import os
import requests
import argparse
from tensorflow.python.lib.io import file_io  
from keras.datasets import mnist 

# Fashion mnist data globals
MNISTF_REPO = 'https://raw.github.com/zalandoresearch/fashion-mnist/master/data/fashion/'
MNISTF_TRAIN_DATA = 'train-images-idx3-ubyte'
MNISTF_TRAIN_LBLS = 'train-labels-idx1-ubyte'
MNISTF_TEST_DATA = 't10k-images-idx3-ubyte'
MNISTF_TEST_LBLS = 't10k-labels-idx1-ubyte'
MNISTF_FILES = [MNISTF_TRAIN_DATA, MNISTF_TRAIN_LBLS, MNISTF_TEST_DATA, MNISTF_TEST_LBLS]


def load_mnist():
  """Load MNIST digits dataset.
  Returns
    Tuple of Numpy arrays: (x_train, y_train), (x_test, y_test).
  """  
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  return (x_train, y_train), (x_test, y_test) 


def load_mnistf(path = 'fashion_mnist'):
  """Loads the fashion MNIST dataset.
  Arguments
      path: path where to cache the dataset locally
  Returns
    Tuple of Numpy arrays: (x_train, y_train), (x_test, y_test).
  """      
  download_mnistf(path = path)
  pickle_mnistf(path = path)
  data = unpack_mnistf(path = path)
  x_train, y_train = data[MNISTF_TRAIN_DATA], data[MNISTF_TRAIN_LBLS]
  x_test, y_test = data[MNISTF_TEST_DATA], data[MNISTF_TEST_LBLS]
  return (x_train, y_train), (x_test, y_test)


def download_mnistf(path):    
  """Download fashion MNIST data in gzip format
  Arguments
      path: path where to cache the dataset locally
  """
  ## TODO: os.path does not work on GCS storage files. 
  if not os.path.exists(path):
    os.makedirs(path)
  for file in MNISTF_FILES:            
    source_path = os.path.join(MNISTF_REPO, file + '.gz')
    target_path = os.path.join(path, file + '.gz')
    if not os.path.exists(target_path):
      print('{}: downloading'.format(file))
      r = requests.get(source_path)
      open(target_path, 'wb').write(r.content)    


def pickle_mnistf(path):
  """Pickle fashion MNIST data
  Opens gzip file, pickles dumps file into path directory.
  Arguments
      path: path where to cache the dataset locally
  """
  for file in MNISTF_FILES:
    source_path = os.path.join(path, file + '.gz')
    target_path = os.path.join(path, file + '.pkl')    
    if not os.path.exists(target_path):
      print('{}: pickling file'.format(file))
      with gzip.open(source_path, 'rb') as gz:
        if 'labels' in file:
          data = np.frombuffer(gz.read(), dtype=np.uint8, offset=8)
        elif 'images' in file:          
          data = np.frombuffer(gz.read(), dtype=np.uint8, offset=16)
          data = data.reshape(np.shape(data)[0]/784, 784)
        pickle.dump(data, open(target_path, 'wb'), -1)


def unpack_mnistf(path):
  """Load and unpack fashion mnist pickle files.
  Arguments
      path: path where to cache the dataset locally
  Returns
    dictionary of data: {file name: data}
  """  
  data={}
  for file in MNISTF_FILES:
    source_path = os.path.join(path, file + '.pkl')
    file_stream = file_io.FileIO(source_path, mode='r')
    data[file] = pickle.load(file_stream) # syntax for python 2
  return data


def visualize_data(x, y, fig_name, n_imgs = 100):
  """Visualize samples from dataset
  Argumetns
    x: image data
    y: image labels
    fig_name: figure name (without extension)
    n_imgs: number of samples to show
  """
  n_cols = int(math.sqrt(n_imgs))              
  n_rows = int(math.ceil(float(n_imgs)/n_cols))  
  f, axes = plt.subplots(n_rows, n_cols, sharey=True, figsize=(12,12))    
  for i in range(n_imgs):    
    axes[i / n_cols, i % n_cols].imshow( np.reshape(x[i], (28,28)), cmap='Greys_r' ) 
    axes[i / n_cols, i % n_cols].set_xticklabels([])
    axes[i / n_cols, i % n_cols].set_yticklabels([])
    axes[i / n_cols, i % n_cols].set_title(y[i])
    axes[i / n_cols, i % n_cols].axis('off')
  f.savefig(fig_name + '.pdf') 


def run(dataset, n_imgs):  
  if dataset == 'mnistf':
    print('Loading data set: Fashion MNIST')
    (x_train, y_train), (x_test, y_test) = load_mnistf(path = 'fashion_mnist')
  elif dataset == 'mnist':
    print('Loading data set: MNIST digits')
    (x_train, y_train), (x_test, y_test) = load_mnist()
  else:
    print('Loading default data set: MNIST digits')
    dataset = 'mnist'
    (x_train, y_train), (x_test, y_test) = load_mnist()    
  visualize_data(x_test, y_test, fig_name = dataset, n_imgs = n_imgs)


def get_args():
    parser = argparse.ArgumentParser()    
    parser.add_argument("--dataset", type=str, default='mnist')
    parser.add_argument("--n_imgs", type=int, default=100)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
  args = get_args()
  run(dataset = args.dataset, n_imgs = args.n_imgs)
