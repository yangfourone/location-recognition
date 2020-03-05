import os
import shutil

# video folder name
dir = 'video'
# move from
moveFrom = 'dataset/train'
# move to
moveTo = 'dataset/valid'
# total files count in folder
fileAmount = len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])
# all files name in folder
fileName = os.listdir(dir)
# how much proportion for testing data
rate = 0.3



for file in fileName:
    fileNameSplit = file.split('.')
    moveFromDir = moveFrom + '/' + fileNameSplit[0]
    moveToDir = moveTo + '/' + fileNameSplit[0]
    if not os.path.isdir(moveToDir):
        os.mkdir(moveToDir)
    # total files count in folder
    fileAmount = len([name for name in os.listdir(moveFromDir) if os.path.isfile(os.path.join(moveFromDir, name))])
    testingAmount = int(fileAmount * rate)
    # move file
    for index in range(testingAmount):
        shutil.move(moveFrom + '/' + fileNameSplit[0] + '/image_' + str(index + 1) + '.jpg', moveTo + '/' + fileNameSplit[0] + '/image_' + str(index + 1) + '.jpg')