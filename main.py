from flask import Flask, request, Response, jsonify
app = Flask(__name__)

# import sys
# sys.path.append('./face-recognition')
# sys.path.append('./face-detection')

import numpy as np
from face_recognition.lib.preprocessing import ImageAugmentator
from face_recognition.lib.utils import Loader, Maximizer
from face_recognition.lib.model import FaceRecognitionClassifier
from face_detection.lib.model import FaceDetectionRegressor
import glob
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def load_faces_to_tensor(dir, user_list):
  X = []

  for user in user_list:
    glob_path = "{dir}/{user}/*.jpg".format(dir=dir, user=user)

    image_path_list = glob.glob(glob_path)
    print(image_path_list)

    loader = Loader()
    for image_path in image_path_list:
      images = loader.load_image(image_path)
      images = img_to_array(images)
      X.append(images)

  X = np.asarray(X)

  return X

from os import mkdir
from datetime import datetime
from random import randrange
def timestamp():
  stamp = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
  return str(randrange(100, 1000)) + '-' + stamp 

from urllib import urlopen
import cv2
@app.route('/demo', methods=['GET', 'POST'])
def demo():
  width = int(request.form('width'))
  height = int(arequestrgs.form('height'))
  file = request.files['photo']

  weight_path = "."
  img_dir = timestamp()
  mkdir(img_dir)

  file.save(img_dir + '/img.jpg')
  img = cv2.imread(img_dir + '/img.jpg')
  img = cv2.resize(img, (width, height)) 
  cv2.imwrite(img_Dir + '/img.jpg', img)

  detector = FaceDetectionRegressor(weight_path)
  predictions = detector.predict(img_dir)

  print predictions
  #TODO face detection using 'url'
  faces = []
  for prediction in predictions[0][0]:
    faces.append({
      "left": prediction[0],
      "width": prediction[1] - prediction[0],
      "top": prediction[2],
      "height": prediction[3] - prediction[2],
      })

  data = {
          'msg': 'detect',
          'faces': faces
          }

  return jsonify(data)


@app.route('/detect', methods=['GET', 'POST'])
def detect():
  args = request.args
  url = args.get('url') # 'url'

  weight_path = "/home/deepcheck/Workspace/live/deepcheckflask"
  img_dir = timestamp()
  mkdir(img_dir)

  req = urlopen(url)
  img = req.read()

  file = open(img_dir + '/img.jpg', 'wb')
  file.write(img)
  file.close()

  detector = FaceDetectionRegressor(weight_path)
  predictions = detector.predict(img_dir)

  #TODO face detection using 'url'
  faces = []
  for prediction in predictions[0][0]:
    faces.append({
      "left": prediction[0],
      "width": prediction[1] - prediction[0],
      "top": prediction[2],
      "height": prediction[3] - prediction[2],
      })

  data = {
          'msg': 'detect',
          'faces': faces
          }

  return jsonify(data)

@app.route('/train', methods=['GET', 'POST'])
def train():
  args = request.args

  uid = args.get('uid')
  dir = args.get('dir')

  course_name = uid
  user_list = os.listdir(dir)

  X = load_faces_to_tensor(dir, user_list)
  y = range(0, len(X))

  augmentator = ImageAugmentator(count=40)
  X, y = augmentator.augment(X, y)

  model = FaceRecognitionClassifier(user_list)
  model.fit(X, y)

  weight_path = "{dir}/{course_name}.h5".format(dir=dir, course_name=course_name)
  model.save_weights(weight_path)

  data = {
          'msg': 'success'
          }

  return jsonify(data)

@app.route('/identify', methods=['GET', 'POST'])
def identify():
  args = request.args

  uid = args.get('uid')
  dir = args.get('dir')
  test_dir = args.get('test_dir')

  course_name = uid
  user_list = [dir_name for dir_name in os.listdir(dir) if os.path.isdir(os.path.join(dir, dir_name))]
  test_list = [dir_name for dir_name in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, dir_name))]

  X = load_faces_to_tensor(test_dir, test_list)

  # init model & load weight
  model = FaceRecognitionClassifier(user_list)
  weight_path = "{dir}/{course_name}.h5".format(dir=dir, uid=uid, course_name=course_name)
  model.load_weights(weight_path)

  prediction = model.predict(X)

  faces = []

  for i, label in enumerate(prediction):
    face = dict()
    face['faceId'] = test_list[i]
    face['candidates'] = []
    face['candidates'].append({
          'personId': user_list[label],
          'confidence': 0.7#df[columns[y]][x]
      })
    faces.append(face)

  return jsonify(faces)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
