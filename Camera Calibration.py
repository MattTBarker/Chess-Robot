import cv2
import numpy as np
import os

PATH='C:/Users/Matt/Desktop/Testboards/'

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

size=None

for fname in os.listdir(PATH):
    img = cv2.imread(PATH+fname)
    img = img[:,:img.shape[1]//2]
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if size is None:
        size=gray.shape[::-1]

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2=cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7,6), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, size,None,None)

print(mtx)
print(dist)

np.save(PATH+"Camera_Calibration_Mtx",mtx)
np.save(PATH+"Camera_Calibration_Dist",dist)
