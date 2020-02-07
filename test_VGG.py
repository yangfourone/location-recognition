from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import sys
import numpy as np

# 從參數讀取圖檔路徑
files = sys.argv[1:]

# 載入訓練好的模型
net = load_model('model/model-vgg16-final.h5')

cls_list = ['P0_d000', 'P0_d090', 'P0_d180', 'P0_d270', 'P1_d000', 'P1_d090', 'P1_d180', 'P1_d270', 'P2_d000',
            'P2_d090', 'P2_d180', 'P2_d270', 'P3_d000', 'P3_d090', 'P3_d180', 'P3_d270', 'P4_d000', 'P4_d090',
            'P4_d180', 'P4_d270', 'P5_d000', 'P5_d090', 'P5_d180', 'P6_d000', 'P6_d090', 'P6_d270']

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
