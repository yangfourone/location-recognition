import cv2
import os.path

# -------------parameters---------------
# video folder name
dir = 'video'
# path about saving images
saveFolder = 'dataset/train'
# fetch frequency
frameRate = 5
# total files count in folder
fileAmount = len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])
# all files name in folder
fileName = os.listdir(dir)


def getFrame(sec):
    videoCap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames, image = videoCap.read()
    if hasFrames:
        fileNameSplit = file.split('.')
        saveDir = saveFolder + '/' + fileNameSplit[0]
        if not os.path.isdir(saveDir):
            os.mkdir(saveDir)
        write_name = saveDir + '/image_' + str(count) + '.jpg'
        cv2.imwrite(write_name, image)
    return hasFrames


for file in fileName:
    capturePath = dir + '/' + file
    videoCap = cv2.VideoCapture(capturePath)
    sec = 0
    count = 1
    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)
