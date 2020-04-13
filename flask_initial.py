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
# cls_list = ['EE-7F-A-000', 'EE-7F-A-180', 'EE-7F-B-000', 'EE-7F-B-180', 'EE-7F-C-000', 'EE-7F-C-180', 'EE-7F-D-090',
#             'EE-7F-E-000', 'EE-7F-E-090', 'EE-7F-E-180', 'EE-7F-E-270', 'EE-7F-F-000', 'EE-7F-F-270', 'EE-7F-G-000',
#             'EE-7F-G-180', 'EE-7F-H-000', 'EE-7F-H-180', 'EE-7F-I-000', 'EE-7F-I-180', 'EE-7F-J-180', 'EE-7F-J-270',
#             'EE-7F-K-090', 'EE-8F-A-000', 'EE-8F-A-180', 'EE-8F-B-090', 'EE-8F-C-000', 'EE-8F-C-180', 'EE-8F-C-270',
#             'EE-8F-D-000', 'EE-8F-D-180', 'EE-8F-E-000', 'EE-8F-E-180', 'EE-8F-F-000', 'EE-8F-F-180', 'EE-8F-G-180',
#             'EE-8F-G-270']
#
# chinese_list = ['7F_背對舊實驗室門前', '7F_面對舊實驗室門前', '7F_瑞光門前面對飲水機', '7F_瑞光門前背對飲水機',
#                 '7F_樓梯面對飲水機', '7F_樓梯面對舊實驗室', '7F_電梯前背對窗戶', '7F_面對飲水機', '7F_飲水機前面對牆壁',
#                 '7F_背對飲水機', '7F_飲水機前面對窗戶', '7F_長走廊靠飲水機端', '7F_長走廊面對窗戶', '7F_長廊中後段面前',
#                 '7F_長廊中後段面後', '7F_長廊中段面前', '7F_長廊中段面後', '7F_長廊中前段面前', '7F_長廊中前段面後',
#                 '7F_長廊前段面後', '7F_長廊前段面窗', '7F_煥宗門前面鏡子', '8F_樓梯面前', '8F_樓梯面後', '8F_電梯前面',
#                 '8F_飲水機面前', '8F_飲水機面後', '8F_背對飲水機', '8F_化學檯子面前', '8F_化學檯子面後', '8F_瑞光LAB面前',
#                 '8F_瑞光LAB面後', '8F_伯奇辦公室面前', '8F_伯奇辦公室面後', '8F_長走廊前段面後', '8F_長走廊前段面樓梯']

cls_list = ['EE-7F-A-000', 'EE-7F-A-180', 'EE-7F-B-000', 'EE-7F-B-180', 'EE-7F-C-000', 'EE-7F-C-180', 'EE-7F-D-090',
            'EE-7F-E-000', 'EE-7F-E-090', 'EE-7F-E-180', 'EE-7F-E-270', 'EE-7F-F-000', 'EE-7F-F-270', 'EE-7F-G-000',
            'EE-7F-G-180', 'EE-7F-H-000', 'EE-7F-H-180', 'EE-7F-I-000', 'EE-7F-I-180', 'EE-7F-J-180', 'EE-7F-J-270',
            'EE-7F-K-090']

chinese_list = ['7F_背對舊實驗室門前', '7F_面對舊實驗室門前', '7F_瑞光門前面對飲水機', '7F_瑞光門前背對飲水機',
                '7F_樓梯面對飲水機', '7F_樓梯面對舊實驗室', '7F_電梯前背對窗戶', '7F_面對飲水機', '7F_飲水機前面對牆壁',
                '7F_背對飲水機', '7F_飲水機前面對窗戶', '7F_長走廊靠飲水機端', '7F_長走廊面對窗戶', '7F_長廊中後段面前',
                '7F_長廊中後段面後', '7F_長廊中段面前', '7F_長廊中段面後', '7F_長廊中前段面前', '7F_長廊中前段面後',
                '7F_長廊前段面後', '7F_長廊前段面窗', '7F_煥宗門前面鏡子']


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
        print('    {:.3f}  {}  {}'.format(predict[index], cls_list[index], chinese_list[index]))
        class_split = cls_list[index].split('-')
        content = {'building': class_split[0],
                   'floor': class_split[1],
                   'position': class_split[2],
                   'degree': class_split[3],
                   'chinese': chinese_list[index]}
        response.append(content)

    return response


@app.route('/recognize-initial', endpoint='recognize-initial')
def recognize_initial():
    global model
    model = load_model('model/EE7F_model-inceptionV3-final.h5')
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

