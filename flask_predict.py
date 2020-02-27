import base64
import os
import os.path
import numpy as np
import sys
import subprocess
import tensorflow as tf
from datetime import datetime
from flask import Flask, jsonify, request
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.backend import clear_session

app = Flask(__name__)
model = None

# 建立 class 名稱陣列
cls_list = ['A-000', 'A-180', 'B-000', 'B-180', 'C-090', 'D-000', 'D-090', 'D-180', 'D-270', 'E-000', 'E-270', 'F-000',
            'F-180', 'G-000', 'G-180', 'H-000', 'H-180', 'I-180', 'I-270', 'J-090']

chinese_list = ['背對舊實驗室門前', '面對舊實驗室門前', '瑞光門前面對飲水機', '瑞光門前背對飲水機', '電梯前背對窗戶',
                '面對飲水機', '飲水機前面對牆壁', '背對飲水機', '飲水機前面對窗戶', '長走廊靠飲水機端', '長走廊面對窗戶',
                '長廊中後段面前', '長廊中後段面後', '長廊中段面前', '長廊中段面後', '長廊中前段面前', '長廊中前段面後',
                '長廊前段面後', '長廊前段面窗', '煥宗門前面鏡子']


def load_our_model():
    global model
    global graph
    model = load_model('model/EE7F_model-vgg16-final.h5')
    graph = tf.get_default_graph()


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
    predict = model.predict(predict_input)[0]
    top_index = predict.argsort()[::-1][:2]
    for index in top_index:
        print('    {:.3f}  {}  {}'.format(predict[index], cls_list[index], chinese_list[index]))

    return chinese_list[top_index[0]]


@app.route('/recognize-initial', endpoint='recognize-initial')
def recognize_initial():
    path = 'dataset/test/initial.jpg'
    predict_input = pre_process_image(path)
    with graph.as_default():
        result = recognition(predict_input)
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
        with graph.as_default():
            result = recognition(predict_input)

        # delete file
        os.remove(path)

        return jsonify(result)


load_our_model()
ip = subprocess.check_output([sys.executable, 'get_ip.py']).decode()
app.run(host=ip, port=5000, debug=True)

