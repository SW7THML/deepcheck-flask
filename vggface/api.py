from scipy import misc
import copy
import numpy as np
import pandas as pd
import glob
from keras.layers import Flatten, Dense
import os
from keras.optimizers import SGD
from vggface import vggface
from keras import backend as K

pd.options.display.float_format = '{:.4f}'.format

def load_unknown_users(unknown_dir):
    unknown_users = []
    for name in os.listdir("{0}".format(unknown_dir)):
        unknown_users.append(name)
    return unknown_users

def train(uid, dir):
    print('train function called')
    unknown_dir = "./unknown"
    unknown_users = load_unknown_users(unknown_dir)
    attendant_users, faces = vggface.load_users(dir)
    vggface.augment_images(faces, dir + "/augmented")
    faces = vggface.load_train(dir + "/augmented", attendant_users, unknown_dir, unknown_users)
    train_X, train_Y = vggface.load_data(faces, attendant_users + unknown_users, unknown_users)

    model = vggface.VGGFace()
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(len(attendant_users) + 1, activation='softmax', name='new_fc8'))
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=["accuracy"])
    model.fit(train_X, train_Y, batch_size=8, nb_epoch=25 * int(len(train_Y) / 60), shuffle=True, verbose=1)
    #scores = model.evaluate(train_X, train_Y, verbose=0)
    #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    model.save_weights("{0}/{1}.h5".format(dir, uid))
    K.clear_session()

    return True

def validate(uid):
    print('validate function called')

def identify(uid, dir, test_dir, face_ids):
    print('identify function called')
    unknown_dir = "./unknown"
    unknown_users = load_unknown_users(unknown_dir)
    attendant_users, faces = vggface.load_users(dir)

    test_users = face_ids
    print(test_dir, test_users)

    images = glob.glob("{0}/*.jpg".format(test_dir))
    print(images)

    columns = ["Unknown"] + attendant_users
    unknown_e = 2 * (1.0 / len(columns))
    df = vggface.identify(attendant_users, test_users, images, "{0}/{1}.h5".format(dir, uid), len(attendant_users) + 1)
    print(df)
    total, coord = vggface.hungarian(df[df.iloc[:, 0] < unknown_e])
    print("total", total, "coordnates", coord)

    names = df[df.iloc[:, 0] < unknown_e].index
    faces = []
    for x, y in coord:
        face = dict()
        face['faceId'] = names[x]
        face['candidates'] = []
        if y > 0:
            face['candidates'].append({
                    'personId': columns[y],
                    'confidence': df[columns[y]][x]
                })
        faces.append(face)
    for face_id in list(set(face_ids) - set(names)):
        face = dict()
        face['faceId'] = face_id
        face['candidates'] = []
        faces.append(face)

    return faces
