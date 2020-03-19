import os
import shutil
import math

# 路徑：從哪邊移動
# move from
moveFrom = 'dataset/train'

# 路徑：移動到哪邊
# move to
moveTo = 'dataset/valid'

# 總共有幾個影片檔
# total files count in folder
fileAmount = len([name for name in os.listdir(moveFrom) if os.path.isfile(os.path.join(moveFrom, name))])

# 所有影片的名字
# all files name in folder
fileName = os.listdir(moveFrom)

# 多少比例的資料會被移動
# how much proportion for testing data
rate = 0.3

# 影片的主要分類名稱總共有幾個字母
# numbers of characters in video name
videoTitleCharacterNum = 11

for file in fileName:
    targetFolder = file[0:videoTitleCharacterNum]
    moveFromDir = moveFrom + '/' + targetFolder
    moveToDir = moveTo + '/' + targetFolder

    # 如果沒有相對應的資料夾就創造新資料夾
    # make a new folder if there is no correct folder
    if not os.path.isdir(moveToDir):
        os.mkdir(moveToDir)

    # 算出該資料夾總共有幾個檔案
    # total files count in folder
    fileAmount = len([name for name in os.listdir(moveFromDir) if os.path.isfile(os.path.join(moveFromDir, name))])
    testingAmount = int(fileAmount * rate)

    # 移動檔案
    # move file
    for index in range(testingAmount):
        # 平均分散的移動
        # moving average
        shutil.move(moveFromDir + '/image_' + str(math.floor((index + 1)/rate)) + '.jpg',
                    moveToDir + '/image_' + str(math.floor((index + 1)/rate)) + '.jpg')
