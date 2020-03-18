import cv2
import os.path

# 影片的來源資料夾
# video folder name
videoSource = 'video'

# 圖片儲存的位置
# path about saving images
saveFolder = 'dataset/train'

# 圖片擷取的頻率 (秒)
# fetch frequency (second)
frameRate = 5

# 影片的主要分類名稱總共有幾個字母
# numbers of characters in video name
videoTitleCharacterNum = 4

# 總共有幾個影片檔
# total files count in folder
fileAmount = len([name for name in os.listdir(videoSource) if os.path.isfile(os.path.join(videoSource, name))])

# 所有影片的名字
# all files name in folder
fileName = os.listdir(videoSource)

# 上一個處理的影片名字
# last video title
lastVideoTitle = ''


def getFrame():
    videoCap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames, image = videoCap.read()
    if hasFrames:
        # 製作儲存位置
        # make file saving path
        savePath = saveFolder + '/' + targetFolder

        # 如果沒有相對應的資料夾就創造新資料夾
        # make a new folder if there is no correct folder
        if not os.path.isdir(savePath):
            os.mkdir(savePath)

        # 依照編號寫到正確的資料夾
        # write file with index into correct folder
        write_name = savePath + '/image_' + str(count) + '.jpg'
        cv2.imwrite(write_name, image)

    return hasFrames


for file in fileName:
    # 建立影片分類的資料夾名稱, 例如: Apple.MOV, Apple-1.MOV, Apple-2.MOV 都會被寫入在 Apple 的資料夾內
    # make a folder by video main category, e.g. Apple.MOV, Apple-1.MOV, Apple-2.MOV will be writen in folder Apple
    targetFolder = file[0:videoTitleCharacterNum]

    # 如果同分類之影片已經切割完成, 則將檔案名稱計數重置為1
    # if the same main category videos are separated successfully, then initial the file index
    if lastVideoTitle != targetFolder:
        count = 1

    # 更新上一個影片的主要分類
    # update last video's main category
    lastVideoTitle = targetFolder

    # 選取要擷取的影片位置
    # get the video path
    capturePath = videoSource + '/' + file
    videoCap = cv2.VideoCapture(capturePath)

    # 重置抓取的開始時間
    # reset capture time
    sec = 0

    success = getFrame()
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame()
