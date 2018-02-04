import get_bucket_data as get

import argparse
import numpy as np
import copy

from tensorflow.python.lib.io import file_io
from keras.models import model_from_json, Model
from keras.datasets import mnist
from keras.layers import Input
from keras import backend as K


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
  print("Autoencoder loaded")
  
  return loaded_model

def load_encoder(autoencoder):
  #encoder = load_model(BUCKET_ID)
  #encoder.layers = encoder.layers[:22]
  img_rows, img_cols, img_chns = 28, 28, 1
  original_img_size = (img_rows, img_cols, img_chns)  
  x = Input(shape=original_img_size)
  n_encoder_layers = 22
  enco = autoencoder.layers[1](x)
  for i in range(2,n_encoder_layers):
      enco = autoencoder.layers[i](enco)        
  encoder = Model(x, enco)
  print("Encoder loaded")
  return encoder
  

def load_decoder(autoencoder):
  # https://stackoverflow.com/questions/44472693/how-to-decode-encoded-data-from-deep-autoencoder-in-keras-unclarity-in-tutorial
  encoded_size = (4, 4, 32)
  x_encoded = Input(shape=encoded_size)
  n_decoder_layers = 23
  deco = autoencoder.layers[-n_decoder_layers](x_encoded)
  for i in range(n_decoder_layers-1,0,-1):
      deco = autoencoder.layers[-i](deco)        
  decoder = Model(x_encoded, deco)
  print("Decoder loaded")
  return decoder

def get_encoded(encoder, data):
  # https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer
  inp = encoder.input                                           # input placeholder  
  '''
  outputs = [layer.output for layer in encoder.layers]          # all layer outputs
  functor = K.function([inp]+ [K.learning_phase()], outputs )   # evaluation function
  x_encoded = [func([data, 1.]) for func in functors]
  '''
  outputs = encoder.layers[-1].get_output_at(0)
  functor = K.function([inp]+ [K.learning_phase()], [outputs] )   # evaluation function
  x_encoded = functor([data, 0.])
  return x_encoded

def generate_latent_viz(x_encoded, decoder):
  result = []
  rank = []
  for i in range(32):
    r,c = np.argmax(x_encoded[0,:,:,i]) / 4, np.argmax(x_encoded[0,:,:,i]) % 4 
    mask = np.zeros(np.shape(x_encoded))
    #mask[0,r,c,i] = 16 # let mask all but max activation of layer i
    mask[0,:,:,i] = 4 # let mask pass layer i
    result.append( decoder.predict(x_encoded*mask) )
    rank.append( np.max(x_encoded[0,:,:,i]) ) 

  rank_sorted, result_sorted, filter_index = ( list(x) for x in zip(*sorted( zip(rank, result, range(32)), reverse=True ) ) )
  return rank_sorted, result_sorted, filter_index

def interpolate_latent(x1_encoded, x2_encoded, decoder, n_steps=10):
  
  shape = [1, 4, 4, 32]
  x1_encoded = np.reshape(x1_encoded, shape)
  x2_encoded = np.reshape(x2_encoded, shape)

  target_encoded = x1_encoded
  result = [decoder.predict(x1_encoded)]

  delta = (x2_encoded - x1_encoded)/(1.0*n_steps)
  for _ in range(n_steps):
    target_encoded = target_encoded + delta
    result.append(decoder.predict(target_encoded))

  return result



  
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
    
def normalize(x):
  # utility function to normalize a tensor by its L2 norm
  return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def run(BUCKET_ID, DOWNLOAD, VERBOSE, n_imgs=20, noise_lvl=1):
  ## download data from cloud to local
  if DOWNLOAD:
    download_data(BUCKET_ID)

  ## load model from local files  
  autoencoder = load_model(BUCKET_ID)  
  
  ## generate encoder and decoder from autoencoder
  encoder = load_encoder(autoencoder)  
  decoder = load_decoder(autoencoder)
  if VERBOSE == True: 
    autoencoder.summary()
    encoder.summary()
    decoder.summary()

  ## load MNIST test data
  x, labels = load_data(n_imgs)

  ## get latent representations
  #x_encoded = get_encoded(encoder, x)
  x_encoded = encoder.predict(x)
  '''
  ## generate filter visualization
  latent_rank, latent_viz, filter_index = generate_latent_viz(x_encoded, decoder)

  ## plot results
  n_rows, n_cols = 8, 4  
  f, axes = plt.subplots(n_rows+1, n_cols, sharey=True, figsize=(5,10))    
  
  for i in range(32):
    axes[i / n_cols, i % n_cols].imshow(latent_viz[i][0,:,:,0])      
    axes[i / n_cols, i % n_cols].set_title("{:02.0f},{:04.2f}".format(filter_index[i],latent_rank[i]))


  result_auto = autoencoder.predict(x)
  result_deco = decoder.predict(x_encoded)
  axes[n_rows, 0].imshow(x[0,:,:,0])      
  axes[n_rows, 1].imshow(result_auto[0,:,:,0])      
  axes[n_rows, 2].imshow(result_deco[0,:,:,0])      

  fig_name = 'viz.pdf'
  f.savefig(fig_name) 
  '''
  ## plot results
  n_rows, n_cols = n_imgs/2, n_imgs/2
  f, axes = plt.subplots(n_rows, n_cols, sharey=True, figsize=(8,8))    

  for j in range(n_rows):
    result = interpolate_latent(x_encoded[2*j], x_encoded[2*j+1], decoder, n_steps=n_cols-1)    
    for i in range(n_cols):
      axes[j, i].imshow(result[i][0,:,:,0])      
      axes[j, i].set_xticklabels([])
      axes[j, i].set_yticklabels([])
    print labels[2*j], labels[2*j+1]
  fig_name = 'interpolate_many.pdf'
  f.savefig(fig_name) 
  

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  #mnist_AE_20170829_184249
  #mnist_DN_20170830_220943
  parser.add_argument('--BUCKET_ID', help='Specify bucket id.', default='mnist_DN_20170830_220943')
  parser.add_argument('--DOWNLOAD', help='Flag to specify if data is to be downloaded from cloud.', action='store_true', default=False)
  parser.add_argument('--n_imgs', help='Number of images to evaluate.', default=30, type=int)
  parser.add_argument('--noise_lvl', help='Number of noise iterations to add.', default=1, type=int)
  parser.add_argument('--VERBOSE', help='Flag to specify verbosity.', action='store_true', default=False)
  args = parser.parse_args()
  arguments = args.__dict__

  run(**arguments)