from flask import Flask, request, Response, jsonify
app = Flask(__name__)

from vggface import api as vggface 

@app.route('/')
def root():
    return 'Hello World!'

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    args = request.args
    url = args.get('url') # 'url'

    #TODO face detection using 'url'
    faces = []
    faces.append({
        "width": 78,
        "height": 78,
        "left": 394,
        "top": 54
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

    if vggface.train(uid, dir):
        data = {
                'msg': 'success'
                }
    else:
        data = {
                'msg': 'failure'
                }

    return jsonify(data)

@app.route('/identify', methods=['GET', 'POST'])
def identify():
    args = request.args
    uid = args.get('uid')
    dir = args.get('dir')
    test_dir = args.get('test_dir')
    face_ids = args.get('face_ids')
    face_ids = face_ids.replace('", "', ',')
    face_ids = face_ids.replace('["', '')
    face_ids = face_ids.replace('"]', '')
    face_ids = face_ids.split(",")

    faces = vggface.identify(uid, dir, test_dir, face_ids)
    data = faces

    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)