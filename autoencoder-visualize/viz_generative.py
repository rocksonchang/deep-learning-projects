'''Visualization of the filters of VGG16, via gradient ascent in input space.

This script can run on CPU in a few minutes (with the TensorFlow backend).

Results example: http://i.imgur.com/4nj4KjN.jpg
'''
from __future__ import print_function

from scipy.misc import imsave
import numpy as np
import time
#from keras.applications import vgg16
from keras import backend as K
from tensorflow.python.lib.io import file_io
from keras.models import model_from_json

# dimensions of the generated pictures for each filter.
##img_width = 128
##img_height = 128
img_width, img_height, img_chns = 28, 28, 1

# the name of the layer we want to visualize
# (see model definition at keras/applications/vgg16.py)
##layer_name = 'block5_conv1'
layer_name = 'conv2d_3'

# util function to convert a tensor into a valid image


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def load_model(BUCKET_ID = 'mnist_DN_20170830_220943'):
  # load json and create model
  arch_filename = BUCKET_ID + "/" + BUCKET_ID + "_arch.json"
  with file_io.FileIO(arch_filename, mode='r') as f:    
    loaded_model = model_from_json(f.read())    
  # load weight into model
  weights_filename = BUCKET_ID + "/" + BUCKET_ID + ".hdf5"
  loaded_model.load_weights(weights_filename)
  print("Loaded model from disk")
  
  return loaded_model


# build the VGG16 network with ImageNet weights
##model = vgg16.VGG16(weights='imagenet', include_top=False)
model  = load_model()
model.summary()
print('Model loaded.')


# this is the placeholder for the input images
input_img = model.input

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


kept_filters = []
for filter_index in range(0, 16):
    # we only scan through the first 200 filters,
    # but there are actually 512 of them
    print('Processing filter %d' % filter_index)
    start_time = time.time()

    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    if K.image_data_format() == 'channels_first':
        loss = K.mean(layer_output[:, filter_index, :, :])
    else:
        loss = K.mean(layer_output[:, :, :, filter_index])

    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)

    # this function returns the loss and grads given the input picture
    # need to add learning_phase argument to specify training (1) or testing (0): https://github.com/fchollet/keras/issues/2417
    iterate = K.function([input_img, K.learning_phase()], [loss, grads])

    # step size for gradient ascent
    step = 0.01

    # we start from a gray image with some random noise
    if K.image_data_format() == 'channels_first':
        input_img_data = np.random.random((1, img_chns, img_width, img_height))
    else:
        input_img_data = np.random.random((1, img_width, img_height, img_chns))
    #input_img_data = (input_img_data - 0.5) * 20 + 128
    #input_img_data = (input_img_data - 0.5)

    # we run gradient ascent for 20 steps
    for i in range(300):
        loss_value, grads_value = iterate([input_img_data, 0])
        input_img_data += grads_value * step

        print('Current loss value:', loss_value)
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break

    # decode the resulting input image
    #if loss_value > 0:
    img = deprocess_image(input_img_data[0])
    kept_filters.append((img, loss_value))
    end_time = time.time()
    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

'''
import pickle
with open('data_{:02.0f}.pkl'.format(filter_index), 'wb') as f:
  pickle.dump(kept_filters, f)
with open('data_{:02.0f}.pkl'.format(filter_index), 'rb') as f:
  loaded_filters = pickle.load(f)
  kept_filters = loaded_filters
'''

# we will stich the best 64 filters on a 8 x 8 grid.
n = 4

# the filters that have the highest loss are assumed to be better-looking.
# we will only keep the top 64 filters.
kept_filters.sort(key=lambda x: x[1], reverse=True)
kept_filters = kept_filters[:n * n]

# build a black picture with enough space for
# our 8 x 8 filters of size 128 x 128, with a 5px margin in between
margin = 5
width = n * img_width + (n - 1) * margin
height = n * img_height + (n - 1) * margin
stitched_filters = np.zeros((width, height, 3))

# fill the picture with our saved filters
for i in range(n):    
    for j in range(n):
        img, loss = kept_filters[i * n + j]
        stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                         (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

# save the result to disk
imsave('stitched_filters_%dx%d_conv2d_3.png' % (n, n), stitched_filters)
