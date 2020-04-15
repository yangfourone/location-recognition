from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import sys
import os.path
import numpy as np
from datetime import datetime


def get_current_time():
    now = datetime.now()
    return now.strftime('%Y%m%d-%H%M%S')


# 宣告計算正確率之參數
groundTruth = []
total = 0
correct = 0

# 建立儲存結果的 txt 檔
current_time = get_current_time()
result_file_name = "result/EE7F+8F_inceptionResNetV2_" + current_time + ".txt"
fp = open(result_file_name, 'w', encoding='UTF-8')

# 從參數讀取圖檔路徑
files = []
path = sys.argv[1:]
fileName = os.listdir(path[0])
for name in fileName:
    files.append(path[0] + '/' + name)
    groundTruth.append((name[0:11]).replace('.jpg', ''))

# 載入訓練好的模型
net = load_model('model/EE7F+8F_model-inceptionResNetV2-final.h5')

# 建立 class 名稱陣列
cls_list = ['EE-7F-A-000', 'EE-7F-A-180', 'EE-7F-B-000', 'EE-7F-B-180', 'EE-7F-C-000', 'EE-7F-C-180', 'EE-7F-D-090',
            'EE-7F-E-000', 'EE-7F-E-090', 'EE-7F-E-180', 'EE-7F-E-270', 'EE-7F-F-000', 'EE-7F-F-270', 'EE-7F-G-000',
            'EE-7F-G-180', 'EE-7F-H-000', 'EE-7F-H-180', 'EE-7F-I-000', 'EE-7F-I-180', 'EE-7F-J-180', 'EE-7F-J-270',
            'EE-7F-K-090', 'EE-8F-A-000', 'EE-8F-A-180', 'EE-8F-B-090', 'EE-8F-C-000', 'EE-8F-C-180', 'EE-8F-C-270',
            'EE-8F-D-000', 'EE-8F-D-180', 'EE-8F-E-000', 'EE-8F-E-180', 'EE-8F-F-000', 'EE-8F-F-180', 'EE-8F-G-180',
            'EE-8F-G-270']

# cls_list = ['EE-7F-A-000', 'EE-7F-A-180', 'EE-7F-B-000', 'EE-7F-B-180', 'EE-7F-C-000', 'EE-7F-C-180', 'EE-7F-D-090',
#             'EE-7F-E-000', 'EE-7F-E-090', 'EE-7F-E-180', 'EE-7F-E-270', 'EE-7F-F-000', 'EE-7F-F-270', 'EE-7F-G-000',
#             'EE-7F-G-180', 'EE-7F-H-000', 'EE-7F-H-180', 'EE-7F-I-000', 'EE-7F-I-180', 'EE-7F-J-180', 'EE-7F-J-270',
#             'EE-7F-K-090']

# 辨識每一張圖
for f in files:
    img = image.load_img(f, target_size=(224, 224))
    if img is None:
        continue
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    predict = net.predict(x)[0]
    top_index = predict.argsort()[::-1][:3]
    # 顯示預測結果
    print('\n' + f, file=fp)
    for index in top_index:
        print('    {:.3f}  {}'.format(predict[index], cls_list[index]), file=fp)
    # 計算正確率
    if cls_list[top_index[0]] == groundTruth[total]:
        correct += 1
    total += 1

# cd to 'dataset/test' and 'rm .DS_Store'
print('\ntotal: ' + str(total) + ', correct: ' + str(correct), file=fp)
print('Rate: ' + str((correct/total)*100) + '%\n', file=fp)
