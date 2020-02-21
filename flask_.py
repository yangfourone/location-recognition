import base64
import os
import os.path
import numpy as np
import sys
import subprocess
from flask import Flask, jsonify, request
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image

app = Flask(__name__)

# 建立 class 名稱陣列
cls_list = ['A-000', 'A-180', 'B-000', 'B-180', 'C-090', 'D-000', 'D-090', 'D-180', 'D-270', 'E-000', 'E-270', 'F-000',
            'F-180', 'G-000', 'G-180', 'H-000', 'H-180', 'I-180', 'I-270', 'J-090']


@app.route("/post-test", methods=["GET", "POST"])
def post_test():
    if request.method == "POST":
        parameter = dict(request.form)
        value = parameter["pic"]
        return jsonify(value)


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        path = "dataset/test/user.jpg"
        file = dict(request.form)
        img_data = file["data"]
        img_data = img_data.replace(" ", "+")
        img_json = img_data.encode()

        with open(path, "wb") as fh:
            fh.write(base64.decodebytes(img_json))

        # recognition
        recognition(path)

        # delete file
        os.remove(path)

        return jsonify('value')


def recognition(path):
    # 載入訓練好的模型
    net = load_model('model/EE7F_model-vgg16-final.h5')
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    pred = net.predict(x)[0]
    top_inds = pred.argsort()[::-1][:3]
    print(path)
    for i in top_inds:
        print('    {:.3f}  {}'.format(pred[i], cls_list[i]))


ip = subprocess.check_output([sys.executable, "get_ip.py"]).decode()
app.run(host=ip, port=5000, debug=True)
