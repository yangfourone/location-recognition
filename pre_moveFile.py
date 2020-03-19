import os
import shutil

# 影片的來源資料夾
# video folder name
videoSource = 'video'

# 路徑：從哪邊移動
# move from
moveFrom = 'dataset/train'

# 路徑：移動到哪邊
# move to
moveTo = 'dataset/valid'

# 總共有幾個影片檔
# total files count in folder
fileAmount = len([name for name in os.listdir(videoSource) if os.path.isfile(os.path.join(videoSource, name))])

# 所有影片的名字
# all files name in folder
fileName = os.listdir(videoSource)

# 多少比例的資料會被移動
# how much proportion for testing data
rate = 0.3

for file in fileName:
    fileNameSplit = file.split('.')
    moveFromDir = moveFrom + '/' + fileNameSplit[0]
    moveToDir = moveTo + '/' + fileNameSplit[0]

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
        shutil.move(moveFrom + '/' + fileNameSplit[0] + '/image_' + str(round((index + 1)/rate)) + '.jpg',
                    moveTo + '/' + fileNameSplit[0] + '/image_' + str(round((index + 1)/rate)) + '.jpg')
