from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils.data_utils import get_file
from keras import backend as K
import numpy as np
import tensorflow as tf
import os
import glob
from scipy import misc
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import copy
import pandas as pd

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
np.random.seed(1337)

TF_WEIGHTS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v1.0/rcmalli_vggface_tf_weights_tf_ordering.h5'

def VGGFace(weights='vggface', classes=2622):
    input_shape = (224, 224, 3)

    # Block 1
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='conv1_1', input_shape=input_shape))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool1'))

    # Block 2
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='conv2_1'))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool2'))

    # Block 3
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_1'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_2'))
    model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool3'))

    # Block 4
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_1'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_2'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool4'))

    # Block 5
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv5_1'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv5_2'))
    model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool5'))

    # Classification block
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dense(classes, activation='softmax', name='fc8'))

    # load weights
    if weights == 'vggface':
        print('K.image_dim_ordering:', K.image_dim_ordering())
        weights_path = get_file('rcmalli_vggface_tf_weights_tf_ordering.h5',
                TF_WEIGHTS_PATH,
                cache_subdir='models')
        model.load_weights(weights_path)

        for layer in model.layers:
            layer.trainable = False
    else:
        weights_path = weights
        print(classes, weights_path)
        model.load_weights(weights_path)

    return model

def identify(attendant_users, user_names, faces, weights, classes):
    model = VGGFace(weights, classes)

    columns = ["Unknown"] + attendant_users
    names = []
    proba = []

    for idx, filepath in enumerate(faces):
        name = user_names[idx]
        
        im = load_image(filepath)
        im = np.expand_dims(im, axis=0)
        out = model.predict(im)

        names.append(name)
        proba.append(out[0])
    
    K.clear_session()
    return pd.DataFrame(proba, index=names, columns=columns)

def augment_images(faces, dir, count = 40):
    datagen = ImageDataGenerator(
            rotation_range=25,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.4,
            zoom_range=0.1,
            channel_shift_range=0.4,
            horizontal_flip=True,
            fill_mode='nearest')

    for k, v in faces.items():
        directory = '{0}/{1}'.format(dir, k)
        if os.path.exists(directory):
            continue
        else:
            os.makedirs(directory)
        for filepath in v['images']:
            img = load_img(filepath)  # this is a PIL image
            x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
            x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=directory, save_prefix=k, save_format='jpg'):
                i += 1
                if i >= count:
                    break

def load_image(filepath):
    im = misc.imread(filepath)
    im = misc.imresize(im, (224, 224)).astype(np.float32)
    aux = copy.copy(im)
    im[:, :, 0] = aux[:, :, 2]
    im[:, :, 2] = aux[:, :, 0]
    im[:, :, 0] -= 93.5940
    im[:, :, 1] -= 104.7624
    im[:, :, 2] -= 129.1863
    return im

def load_data(faces, users, unknown_users = []):
    train_X = []
    train_Y = []
    
    for idx, name in enumerate(users):
        k = name
        v = faces[name]
        for filepath in v['images']:
            train_X.append(load_image(filepath))
            id = idx + 100
            if name in unknown_users:
                id = -1
            train_Y.append(id)
            
    train_X = np.asarray(train_X)
    train_Y = pd.Series(train_Y)
    train_Y = train_Y.astype('category')
    train_Y = train_Y.cat.codes
    train_Y = np.asarray(train_Y)
    return train_X, train_Y

def load_train(attendant_dir, attendant_users, unknown_dir = "", unknown_users = []):
    import random
    faces = dict()
    for idx, name in enumerate(attendant_users):
        images = glob.glob("{0}/{1}/*.jpg".format(attendant_dir, name))
        random.shuffle(images)
        images = images[0:40]
        faces[name] = {
            'images': images
        }
    for idx, name in enumerate(unknown_users):
        images = glob.glob("{0}/{1}/*.jpg".format(unknown_dir, name))
        random.shuffle(images)
        t = int(40 / len(unknown_users))
        images = images[0:t]
        faces[name] = {
            'images': images
        }
    return faces

def load_users(dir):
    attendant_users = []
    for uid in os.listdir("{0}".format(dir)):
        if ".h5" in uid or "augmented" in uid:
            continue
        attendant_users.append(uid)

    faces = dict()
    for idx, name in enumerate(attendant_users):
        faces[name] = {
            'images': glob.glob("{0}/{1}/*.jpg".format(dir, name))
        }

    return attendant_users, faces

def hungarian(df):
    from munkres import Munkres, print_matrix, make_cost_matrix

    M = df.values.tolist()
    matrix = M
    m = Munkres()
    cost_matrix = make_cost_matrix(matrix, lambda cost: 1 - cost) # get maximum
    indexes = m.compute(cost_matrix)

    total = 0
    coord = []
    for row, column in indexes:
        value = matrix[row][column]
        total += value
        coord.append([row, column])
    return total, coord
