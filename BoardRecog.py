import sys
sys.path.append('c:/users/matt/appdata/local/programs/python/python36-32/lib/site-packages')

import cv2
import numpy as np
import os

#Constants
PATH='C:/Users/Matt/Desktop/Chessboards/'
HARRISBLOCKSIZE=4
HARRISAPERTURE=3
HARRISCORNERRESPONSE=0.1
HARRISFILTER=0.1

def cycleImg(img):
    cv2.imshow('chessboard',img)
    if cv2.waitKey(0):
        cv2.destroyAllWindows()

for filename in os.listdir(PATH):  
    img = cv2.imread(PATH + filename)
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0,255,0])
    cycleImg(img)
    imgGrey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    imgGrey = np.float32(imgGrey)
    cornermap = cv2.cornerHarris(imgGrey,HARRISBLOCKSIZE,HARRISAPERTURE,HARRISCORNERRESPONSE)

    img[cornermap>HARRISFILTER*cornermap.max()]=[0,0,255]
    cycleImg(img)

    img = cv2.Canny(img,50,150,apertureSize = 3)
    cycleImg(img)
    
    height, width = img.shape
    img=np.zeros((height,width,3), np.uint8)
    img[cornermap>0.04*cornermap.max()]=[255,255,255]
    cycleImg(img)

    img = cv2.erode(img, None)
    img = cv2.dilate(img, None)

    cycleImg(img)

