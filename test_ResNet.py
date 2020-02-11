from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import sys
import os.path
import numpy as np

# 從參數讀取圖檔路徑
files = []
path = sys.argv[1:]
fileName = os.listdir(path[0])
for name in fileName:
    files.append(path[0] + '/' + name)

# 載入訓練好的模型
net = load_model('model/EE7F_model-resnet50-final.h5')

# 建立 class 名稱陣列
cls_list = ['A-000', 'A-180', 'B-000', 'B-180', 'C-090', 'D-000', 'D-090', 'D-180', 'D-270', 'E-000', 'E-270', 'F-000',
            'F-180', 'G-000', 'G-180', 'H-000', 'H-180', 'I-180', 'I-270', 'J-090']

# 辨識每一張圖
for f in files:
    img = image.load_img(f, target_size=(224, 224))
    if img is None:
        continue
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    pred = net.predict(x)[0]
    top_inds = pred.argsort()[::-1][:5]
    print(f)
    for i in top_inds:
        print('    {:.3f}  {}'.format(pred[i], cls_list[i]))
