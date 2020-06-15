import base64
import os
import os.path
import numpy as np
import sys
import subprocess
from datetime import datetime
from flask import Flask, jsonify, request
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.backend import clear_session

app = Flask(__name__)
model = None

# 建立 class 名稱陣列
cls_list = ['EE-7F-A-000', 'EE-7F-A-180', 'EE-7F-B-000', 'EE-7F-B-180', 'EE-7F-C-000', 'EE-7F-C-180', 'EE-7F-D-090',
            'EE-7F-E-000', 'EE-7F-E-090', 'EE-7F-E-180', 'EE-7F-E-270', 'EE-7F-F-000', 'EE-7F-F-270', 'EE-7F-G-000',
            'EE-7F-G-180', 'EE-7F-H-000', 'EE-7F-H-180', 'EE-7F-I-000', 'EE-7F-I-180', 'EE-7F-J-180', 'EE-7F-J-270',
            'EE-7F-K-090', 'EE-8F-A-000', 'EE-8F-A-180', 'EE-8F-B-090', 'EE-8F-C-000', 'EE-8F-C-180', 'EE-8F-C-270',
            'EE-8F-D-000', 'EE-8F-D-180', 'EE-8F-E-000', 'EE-8F-E-180', 'EE-8F-F-000', 'EE-8F-F-180', 'EE-8F-G-180',
            'EE-8F-G-270']


def save_image(current_time):
    path = 'dataset/test/user_' + str(current_time) + '.jpg'
    file = dict(request.form)
    img_data = file['data']
    img_data = img_data.replace(' ', '+')
    img_json = img_data.encode()
    with open(path, 'wb') as fh:
        fh.write(base64.decodebytes(img_json))
    return path


def pre_process_image(path):
    img = image.load_img(path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    return np.expand_dims(img_array, axis=0)


def get_current_time():
    now = datetime.now()
    return now.strftime('%Y%m%d-%H%M%S')


def recognition(predict_input):
    response = []
    predict = model.predict(predict_input)[0]
    top_index = predict.argsort()[::-1][:5]
    for index in top_index:
        print('    {:.3f}  {}'.format(predict[index], cls_list[index]))
        class_split = cls_list[index].split('-')
        content = {'building': class_split[0],
                   'floor': class_split[1],
                   'position': class_split[2],
                   'degree': class_split[3]}
        response.append(content)

    return response


@app.route('/recognize-initial', endpoint='recognize-initial')
def recognize_initial():
    global model
    model = load_model('model/EE7F+8F_model-inceptionResNetV2-final.h5')
    path = 'initial/initial.jpg'
    predict_input = pre_process_image(path)
    result = recognition(predict_input)
    clear_session()
    return jsonify(result)


@app.route('/post-test', methods=['GET', 'POST'], endpoint='post-test')
def post_test():
    return jsonify('post-test')


@app.route('/recognize', methods=['POST'], endpoint='recognize')
def recognize():
    if request.method == 'POST':
        # get current time
        current_time = get_current_time()

        # save image and get file path
        path = save_image(current_time)

        # pre-process image
        predict_input = pre_process_image(path)

        # recognition
        result = recognition(predict_input)

        # delete file
        os.remove(path)

        # clear_session
        clear_session()

        return jsonify(result)


ip = subprocess.check_output([sys.executable, 'get_ip.py']).decode()
app.run(host=ip, port=5000, debug=True)

