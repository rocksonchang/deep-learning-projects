import get_bucket_data as get

import argparse
import numpy as np
import copy

from tensorflow.python.lib.io import file_io
from keras.models import model_from_json
from keras.datasets import mnist

import matplotlib
matplotlib.use('PDF')
from matplotlib import pyplot as plt

def download_data(BUCKET_ID):
  get.run_cmd(BUCKET_FOLDER=BUCKET_ID)

def load_model(BUCKET_ID):
  # load json and create model
  arch_filename = BUCKET_ID + "/" + BUCKET_ID + "_arch.json"
  with file_io.FileIO(arch_filename, mode='r') as f:    
    loaded_model = model_from_json(f.read())    
  # load weight into model
  weights_filename = BUCKET_ID + "/" + BUCKET_ID + ".hdf5"
  loaded_model.load_weights(weights_filename)
  print("Loaded model from disk")
  
  return loaded_model

def load_data(n_imgs):
  # data from MNIST digits
  # input image dimensions
  img_rows, img_cols, img_chns = 28, 28, 1
  original_img_size = (img_rows, img_cols, img_chns)  
  (_, _), (x_test, y_test) = mnist.load_data()
  # reshape data to (data_size, n_pix, n_pix, n_channels)
  x_test = x_test.astype('float32') / 255.
  x_test = x_test.reshape((x_test.shape[0],) + original_img_size)  
  
  # Random sample of images
  assert len(x_test)==len(y_test)
  p = np.random.permutation(len(x_test))    
  data = x_test[p][0:n_imgs]
  labels = y_test[p][0:n_imgs]

  return data, labels

def aug_noise(data, noise_lvl):
  raw_data=copy.deepcopy(data)
  img_size = np.prod(raw_data.shape)
  for i in range(noise_lvl):    
    raw_data += np.random.normal(loc=0.0, scale =0.25, size=img_size).reshape(raw_data.shape)
  np.clip(raw_data, 0, 1, raw_data)
  return raw_data

'''
def evaluate(model, data, label):
  n_imgs = label.shape
  reconstructed = model.predict(data, batch_size=n_imgs)
    
  # plot reconstructed images and compare
  fig_name = 'evaluate.pdf'
  n_rows = 3 # split orignal images into 2 rows 
  n_cols = n_imgs//n_rows
  f, axes = plt.subplots(n_rows*2, n_cols, sharey=True, figsize=(10,10))  
  for i in range(n_imgs):
      axes[i//n_cols * 2, i % n_cols].imshow(data[i,:,:,0])
      axes[i//n_cols * 2, 0].set_ylabel('Original')
      axes[i//n_cols * 2 +1, i % n_cols].imshow(reconstructed[i,:,:,0])
      axes[i//n_cols * 2 +1, 0].set_ylabel('Reconstructed')
  f.savefig(fig_name)  
'''

def evaluate_noisy(model, data, noisy_data, labels):
  n_imgs = labels.shape[0]

  reconstructed = model.predict(data, batch_size=n_imgs)
  reconstructed_noisy = model.predict(noisy_data, batch_size=n_imgs)
    
  # plot reconstructed images and compare
  fig_name = 'evaluate_noisy_60.pdf'
  n_cols = 4 
  n_rows = n_imgs
  f, axes = plt.subplots(n_rows, n_cols, sharey=True, figsize=(10,10))  
  for i in range(n_imgs):
    axes[i,0].imshow(data[i,:,:,0])
    axes[i,0].set_ylabel('Original')
    axes[i,1].imshow(reconstructed[i,:,:,0])
    axes[i,1].set_ylabel('Reconstructed')
    axes[i,2].imshow(noisy_data[i,:,:,0])
    axes[i,2].set_ylabel('Noisy original')
    axes[i,3].imshow(reconstructed_noisy[i,:,:,0])
    axes[i,3].set_ylabel('Noisy reconstructed')      
  f.savefig(fig_name)  

def run(BUCKET_ID, DOWNLOAD, n_imgs=5, noise_lvl=1):
  ## download data from cloud to local
  if DOWNLOAD:
    download_data(BUCKET_ID)
  ## load model from local files
  model = load_model(BUCKET_ID)
  ## load MNIST test data
  data, labels = load_data(n_imgs)  
  noisy_data = aug_noise(data, noise_lvl)

  #evaluate(model, data, labels)
  evaluate_noisy(model, data, noisy_data, labels)


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--BUCKET_ID', help='Specify bucket id.', default="mnist_AE_20170829_184249")
  parser.add_argument('--DOWNLOAD', help='Flag to specify if data is to be downloaded from cloud.', action='store_true', default=False)
  parser.add_argument('--n_imgs', help='Number of images to evaluate.', default=5)
  parser.add_argument('--noise_lvl', help='Number of noise iterations to add.', default=1, type=int)
  args = parser.parse_args()
  arguments = args.__dict__

  run(**arguments)