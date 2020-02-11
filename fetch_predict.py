from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import sys
import numpy as np
import time
import os.path
import cv2

# 從參數讀取圖檔路徑
files = sys.argv[1:]

# 載入訓練好的模型
net = load_model('model/EE7F_model-resnet50-final.h5')

# Class 分類名稱
cls_list = ['A-000', 'A-180', 'B-000', 'B-180', 'C-090', 'D-000', 'D-090', 'D-180', 'D-270', 'E-000', 'E-270', 'F-000',
            'F-180', 'G-000', 'G-180', 'H-000', 'H-180', 'I-180', 'I-270', 'J-090']

# 照片儲存路徑
saveFolder = 'dataset/test/'

# 擷取頻率 (每x秒1張)
frameRate = 3

# 圖檔 index
index = 0


# 辨識每一張圖
def predict(file_name):
    img = image.load_img(file_name, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    pred = net.predict(x)[0]
    top_inds = pred.argsort()[::-1][:5]
    for i in top_inds:
        print('    {:.3f}  {}'.format(pred[i], cls_list[i]))


# 儲存每一張辨識的圖
def save_frame(file_index):
    write_name = saveFolder + 'image_' + str(file_index) + '.jpg'
    print('save image_' + str(file_index))
    cv2.imwrite(write_name, frame)
    return write_name


# 鏡頭來源
vc = cv2.VideoCapture(0)

print('start!')

if not os.path.isdir(saveFolder):
    os.mkdir(saveFolder)

while True:
    # delay x 秒
    time.sleep(frameRate)

    rval, frame = vc.read()

    index += 1
    file_name = save_frame(index)
    predict(file_name)
