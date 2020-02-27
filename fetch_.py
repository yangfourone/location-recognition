import time
import os.path
import cv2


# 照片儲存路徑
saveFolder = 'dataset/test/'

# 擷取頻率 (每x秒1張)
frameRate = 3

# 圖檔 index
index = 0


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
