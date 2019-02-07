import sys
sys.path.append('c:/users/matt/appdata/local/programs/python/python36-32/lib/site-packages')

import cv2
import numpy as np
import os

#Constants
PATH='C:/Users/Matt/Desktop/Chessboards/'
HARRISBLOCKSIZE=3
HARRISAPERTURE=3
HARRISCORNERRESPONSE=0.02
HARRISFILTER=0.08
MINCORNERDISTANCE=10
CLEANEDGEBLOCKSIZE=10
CLEANEDGEBIAS=1

#(((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) )/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)), ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)))

def cycleImg(img):
    cv2.imshow('chessboard',img)
    if cv2.waitKey(0) & 0xFF:
        cv2.destroyAllWindows()

#Averages nearby centroids into a single point
def getMergedCentroidClusters(centroids, distance=5):
    corners = list({(int(l[0]), int(l[1])) for l in centroids})
    while True:
        flag=True
        
        size=len(corners)
        i=0
        while i < size: 
            corner1 = corners[i]
            for corner in corners:
                if corner==corner1:
                    continue
                if abs(corner[0]-corner1[0])<distance and abs(corner[1]-corner1[1])<distance:
                    corners.remove(corner)
                    corner1=(corner[0]+corner1[0]//2, corner[1]+corner1[1]//2)
                    size-=1
                    flag=False
            i+=1
        if(flag):
            break
    return corners      

def softDilate(img, pixels=1):
    height, width = img.shape[:2]
    img1 = np.zeros([height,width,1],dtype=img.dtype)

    for y in range(pixels,height-pixels):
        for x in range(pixels, width-pixels):
            for i in range(-pixels, pixels+1):
                for j in range(-pixels, pixels+1):
                    img1[y,x]|=img[y+i, x+j]
                    img1[y,x]//=0.9
    return img1

def getFilteredCentroids(img, centroids, samplingRate, threshold, imgTest):
    img = cv2.dilate(img,np.ones((3,3),np.uint8),iterations = 1)

    height, width = img.shape[:2]
    distance = 0.1*max(img.shape[:2])

    centroids = [corner for corner in centroids if corner[0]!=0 and corner[0]!=width and corner[1]!=0 and corner[1]!=height]

    height-=1
    width-=1
    dist=0.05*min(height, width)

    lines=[]

    for corner in centroids:     
        for corner1 in centroids:     
            if abs(corner[0]-corner1[0])<distance and abs(corner[1]-corner1[1])<distance:
                continue

            rho = None
            edgeIntercept = None
            edgeIntercept1 = None

            if corner[0]-corner1[0]==0:
                rho=0.5*np.pi
                edgeIntercept = (corner[0], 0)
                edgeIntercept1 = (corner[0], height)
            else:                                                     
                m=(corner[1]-corner1[1])/(corner[0]-corner1[0])
                c=corner[1]-m*corner[0]
                rho=np.arctan2(corner[1]-corner1[1], corner[0]-corner1[0])        
            
                if c >= 0 and c <= height:  #intersection check for line X=0
                    edgeIntercept = (0, c)
                elif m*width + c >=0 and m*width + c <= height: #intersection check for line X=width
                    edgeIntercept = (width, m*width + c)
                elif -c/m >= 0 and -c/m <= width: #intersection check for line Y=0
                    edgeIntercept = (-c/m, 0)

                if c >= 0 and c <= height and edgeIntercept != (0, c):
                    edgeIntercept1 = (0, c)
                elif m*width + c >=0 and m*width + c <= height and edgeIntercept != (width, m*width + c):
                    edgeIntercept1 = (width, c)
                elif -c/m >= 0 and -c/m <= width and edgeIntercept != (-c/m, 0):
                    edgeIntercept1 = (-c/m, 0)
                else: #must intercept at Y=height
                    edgeIntercept1 = ((height-c)/m, height)

            xStep = (edgeIntercept1[0]-edgeIntercept[0])/samplingRate
            yStep = (edgeIntercept1[1]-edgeIntercept[1])/samplingRate

            edgeIntercept = tuple(map(int,edgeIntercept))
            edgeIntercept1 = tuple(map(int,edgeIntercept1))

            lineValue=0

            for i in range(samplingRate):
                x=int(edgeIntercept[0]+i*xStep)
                y=int(edgeIntercept[1]+i*yStep)
                lineValue += max(img[y, x]-CLEANEDGEBIAS*cv2.mean(img[max(0,y-CLEANEDGEBLOCKSIZE):min(height,y+CLEANEDGEBLOCKSIZE), max(0,x-CLEANEDGEBLOCKSIZE):min(width,x+CLEANEDGEBLOCKSIZE)])[0], 0)

            flag=True
            i=0
            while i<len(lines):
                line=lines[i]
                if edgeIntercept[0]>=line[0][0]-dist and edgeIntercept[0]<=line[0][0]+dist and edgeIntercept[1]>=line[0][1]-dist and edgeIntercept[1]<=line[0][1]+dist:
                    if edgeIntercept1[0]>=line[1][0]-dist and edgeIntercept1[0]<=line[1][0]+dist and edgeIntercept1[1]>=line[1][1]-dist and edgeIntercept1[1]<=line[1][1]+dist:
                        flag=False
                        if lineValue > line[2]:
                            lines[i]=(edgeIntercept, edgeIntercept1, lineValue)
                        else:
                            break
                i+=1
            if flag:
                lines.append((edgeIntercept, edgeIntercept1, lineValue))
                
    lines = set(lines)

#Angle filter probably unnecessary

##    angles = [[1, lines[0][3]]]
##
##    for line in lines:
##        flag=False
##        i=0
##        while i<len(angles):
##            if abs(line[3]-angles[i][1])%(2*np.pi)<0.05*np.pi:
##                flag=True
##                break
##            i+=1
##        if flag:
##            angles[i][1]=angles[i][1]+line[3]
##            angles[i][1]%=2*np.pi
##            angles[i][1]/=2
##            angles[i][0]+=1
##        else:
##            angles.append([1, line[3]])
##
##    angle=max(angles, key=lambda x: x[0])[1]
##    angles = [x for x in angles if abs(angle-x[1])%(2*np.pi)>0.35*np.pi]
##    angle1=max(angles, key=lambda x: x[0])[1]
##             
##    lines = [line for line in lines if abs(angle-line[3])%(2*np.pi)<0.15*np.pi or abs(angle1-line[3])%(2*np.pi)<0.15*np.pi]

    maxLineValue=max(lines, key=lambda l: l[2])[2]*threshold
    lines = [[line[0], line[1], np.arctan2(line[0][1]-line[1][1], line[0][0]-line[1][0])%np.pi, set(), 0] for line in lines if line[2] >= maxLineValue]

    imgTest1=imgTest.copy()
    for line in lines:
        cv2.line(imgTest1,line[0],line[1],(0,0,255),2)
    cycleImg(imgTest1)




#Fast Angle Filter
    
    for line in lines:
        for line1 in lines:
            if abs(line[2]-line1[2])%np.pi<0.01*np.pi:
                line[4]+=1
                continue
                
            intersectX=((line1[0][0]*line1[1][1]-line1[0][1]*line1[1][0])*(line[0][0]-line[1][0])-(line1[0][0]-line1[1][0])*(line[0][0]*line[1][1]-line[0][1]*line[1][0]))/((line1[0][0]-line1[1][0])*(line[0][1]-line[1][1])-(line1[0][1]-line1[1][1])*(line[0][0]-line[1][0]))
            intersectY=((line1[0][0]*line1[1][1]-line1[0][1]*line1[1][0])*(line[0][1]-line[1][1])-(line1[0][1]-line1[1][1])*(line[0][0]*line[1][1]-line[0][1]*line[1][0]))/((line1[0][0]-line1[1][0])*(line[0][1]-line[1][1])-(line1[0][1]-line1[1][1])*(line[0][0]-line[1][0]))
            intersection = (int(intersectX), int(intersectY))
            if intersection[0]>0 and intersection[0]<width and intersection[1]>0 and intersection[1]<height:
                continue
            line[3].add(intersection)


    for line in lines:
        bestScore=0
        for point in line[3]:
            accuracy = np.sqrt(abs(height//2-point[1])**2 + abs(width//2-point[0])**2)*0.1
            i=0
            for point1 in line[3]:
                if abs(point[0]-point1[0])<accuracy and abs(point[1]-point1[1])<accuracy:
                    i+=1
            if i>bestScore:
                bestScore=i
        line[4]+=bestScore

    maxAngleVariance=max(lines, key=lambda l: l[4])[4]*0.5
    lines = [(line[0], line[1], line[2]) for line in lines if line[4] >= maxAngleVariance]

#Janky af, works well on mostly complete boards, poorly on boards with many missing edges




    for line in lines:
        cv2.line(imgTest,line[0],line[1],(0,0,255),2)
    cycleImg(imgTest)

def regenerateMissingEdges(lines):
    
        

for filename in os.listdir(PATH):
    img = cv2.imread(PATH + filename)
    cycleImg(img)

    imgGrey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgGrey = np.float32(imgGrey)

    cornerMap = cv2.cornerHarris(imgGrey,HARRISBLOCKSIZE,HARRISAPERTURE,HARRISCORNERRESPONSE)
    cornerMap = cv2.threshold(cornerMap,0.04*cornerMap.max(),255,cv2.THRESH_BINARY)[1]
    cornerMap = np.uint8(cornerMap)

    centroidArray = cv2.connectedComponentsWithStats(cornerMap)[3]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    imgCanny = cv2.Canny(img,50,150,apertureSize = 3)

    corners = getMergedCentroidClusters(centroidArray, MINCORNERDISTANCE)
    getFilteredCentroids(imgCanny, corners, 50, 0.5, img)
