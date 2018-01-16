'''
Deep Convolutional Generative Adversarial Network (DC-GAN) implementation
Original code: https://github.com/jacobgil/keras-dcgan
Uses Keras with a tensorflow backend

Usage:

Training 
$ python dcgan.py --mode train --batch_size <batch_size> 

Generation 
$ python dcgan.py --mode generate --batch_size <batch_size> 

'''

import os
import matplotlib as mpl
# https://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined/40931739#40931739
if os.environ.get('DISPLAY','') == '':    
    print('No display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from tensorflow.python.lib.io import file_io

import data_sources
import numpy as np
import csv
import argparse
import math

def generator_model():
    model = Sequential()
    model.add(Dense(1024, input_dim=100))
    model.add(Activation('tanh'))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model


def discriminator_model():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), padding='same', input_shape=(28, 28, 1)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))                # floor
    height = int(math.ceil(float(num)/width))  # ceil
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img[:, :, 0]
    return image


def save_model_architecture(model, job_dir, model_name):
    model_json = model.to_json()
    with file_io.FileIO(job_dir + model_name + "_arch.json", mode='w') as json_file:
        json_file.write(model_json)


def save_model(model, job_dir, model_name):
    if 'gs://' in job_dir:
        model.save(model_name + '.h5', True)
        with file_io.FileIO(model_name + '.h5', mode='r') as input_f:
            with file_io.FileIO(job_dir + model_name + '.h5', mode='w') as output_f:
                output_f.write(input_f.read())        
    else:
        model.save(job_dir + model_name + '.h5', True)        


def save_loss(loss_arr, job_dir):
    if 'gs://' in job_dir:
        with open('loss.csv', 'a') as f:
            writer = csv.writer(f)
            for loss in loss_arr:
                writer.writerows([loss]) 
        with file_io.FileIO('loss.csv', mode='r') as input_f:
            with file_io.FileIO(job_dir + 'loss.csv', mode='w') as output_f:
                output_f.write(input_f.read())
    else:
        with open(job_dir + 'loss.csv', 'a') as f:
            writer = csv.writer(f)
            for loss in loss_arr:
                writer.writerows([loss]) 


def save_image(fig, job_dir, fig_name):    
    fig_name = fig_name + ".png"  
    if 'gs://' not in job_dir:
        fig.savefig(job_dir + fig_name)                 
    else:
        fig.savefig(fig_name)                 
        with file_io.FileIO(fig_name, mode='r') as input_f:
            with file_io.FileIO(job_dir + fig_name, mode='w') as output_f:
                output_f.write(input_f.read())    


def moving_average(a, n=3):
    ret = np.cumsum(np.asarray(a, dtype=float))
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:]/n


def plot_loss(loss_arr, job_dir, epoch):
    
    d_loss, g_loss = [], []
    
    if 'gs://' in job_dir:
        loss_path = 'loss.csv'        
    else: 
        loss_path = job_dir + 'loss.csv'            
    with file_io.FileIO(loss_path, mode='r') as csvfile:
        csvReader = csv.reader(csvfile)
        for row in csvReader:
            d_loss.append(row[0])
            g_loss.append(row[1])

    x = np.linspace(0,epoch+1, len(d_loss))
    n=80
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].plot(x, d_loss)
    ax[0].plot(x[n-1:], moving_average(d_loss, n=n))
    ax[0].set_title('Discriminator loss')
    ax[1].plot(x, g_loss)
    ax[1].plot(x[n-1:], moving_average(g_loss, n=n))
    ax[1].set_title('GAN loss')    
    save_image(fig=fig, job_dir=job_dir, fig_name='loss')
    plt.close(fig)


def load_data(dataset, BUCKET_NAME, job_dir):
    print("Loading and pre-processing data")    
    if dataset == 'mnistf':
      if 'gs://' in job_dir:                
          data_path = 'gs://{}/data/fashion_mnist'.format(BUCKET_NAME)        
      else:
          if os.path.isfile('gcloud.local.run.sh'):
              data_path = '../data/fashion_mnist/' # Path is relative to gcloud.local.run.sh           
          else:
              data_path = '../../data/fashion_mnist/' # Path is relative to dcgan.py
      (x_train, y_train), (x_test, y_test) = data_sources.load_mnistf(path=data_path)
      x_train = np.reshape(x_train, (np.shape(x_train)[0], 28, 28))
      x_test = np.reshape(x_test, (np.shape(x_test)[0], 28, 28))
    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = data_sources.load_mnist()          
    else:
        (x_train, y_train), (x_test, y_test) = data_sources.load_mnist()          

    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    x_train = x_train[:, :, :, None]
    x_test = x_test[:, :, :, None]
    
    return (x_train, y_train), (x_test, y_test)


def train(BATCH_SIZE=128, dataset='mnist', BUCKET_NAME='rc_bucket', 
          n_epochs=50, job_dir='tmp/', **kwargs):

    ########## File structure prep ##########    
    if 'gs://' not in job_dir:
        if not os.path.exists(job_dir):
            os.makedirs(job_dir)
        if os.path.exists(job_dir+'loss.csv'):
            os.remove(job_dir+'loss.csv')

    ########## Load data ##########
    (x_train, _), (x_test, y_test) = load_data(dataset=dataset, 
                                               BUCKET_NAME=BUCKET_NAME, 
                                               job_dir=job_dir)
    
    ########## build models ##########
    print("Building models")
    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)    
    d.trainable = True    
    d.compile(loss='binary_crossentropy', optimizer=d_optim)

    ########## serialize model to JSON ##########
    save_model_architecture(d, job_dir, 'discriminator')
    save_model_architecture(g, job_dir, 'generator')
    save_model_architecture(d_on_g, job_dir, 'GAN')

    ########## train GAN ##########
    print("Training GAN")
    for epoch in range(n_epochs):
        #loss_arr = []
        print("Epoch is", epoch)
        print("Number of batches", int(x_train.shape[0]/BATCH_SIZE))
        for index in range(int(x_train.shape[0]/BATCH_SIZE)):                        
            # train discriminator, evaluate discriminator
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            image_batch = x_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = g.predict(noise, verbose=0)
            if index == 0: # periodically save generator output
                image = combine_images(generated_images)
                image = image*127.5+127.5                
                fig = plt.figure()
                plt.imshow(image, cmap='Greys_r')                        
                save_image(fig=fig, job_dir=job_dir, fig_name=str(epoch)+"_"+str(index))                                
                plt.close(fig)

            x = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = d.train_on_batch(x, y)                        

            # train generator, evaluate on full stack            
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
            d.trainable = True

            # store loss
            loss_arr.append([d_loss, g_loss])                        

        print("epoch %d batch %d. d_loss: %f, g_loss: %f" % (epoch, index, d_loss, g_loss))            
  
        ########## save loss and models ##########
        save_loss(loss_arr , job_dir)        
        save_model(g, job_dir, 'generator')
        save_model(d, job_dir, 'discriminator')
        plot_loss(loss_arr, job_dir, epoch)


def generate(BATCH_SIZE, nice=False):
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator.h5')
    if nice:
        d = discriminator_model()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('discriminator')
        noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 100))
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5    
    save_image(image=image, job_dir='', fig_name="generated_image")    
    plt.close(fig)


def get_args():
    parser = argparse.ArgumentParser()    

    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--dataset", type=str, default='mnist')
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.add_argument('--job-dir', help='GCS location to write checkpoints and export models', default=None)
    
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":                
        if args.job_dir == None:
            args.job_dir = "../tmp/local_train_dc_gan_" + args.dataset + "/"
        train(BATCH_SIZE=args.batch_size, 
              dataset=args.dataset,                            
              n_epochs=args.n_epochs, 
              job_dir=args.job_dir)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)