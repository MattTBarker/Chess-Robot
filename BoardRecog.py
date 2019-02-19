import sys
sys.path.append('c:/users/matt/appdata/local/programs/python/python36-32/lib/site-packages')

import cv2
import numpy as np
import os

#Constants
PATH='C:/Users/Matt/Desktop/Testboards/'
                        #RECOMMENDED
HARRISAPERTURE=3        #3
CORNERDETECTIONTYPE=1   #0
HARRISTHRESHOLD=0.01    #0.16
CHESSTHRESHOLD=0.9      #0.5
LINETHRESHOLD=0.3       #0.5
MINCORNERDISTANCE=10    #10
CANNYDILATION=3         #3
CLEANEDGEBLOCKSIZE=10   #10
CLEANEDGEBIAS=1.5       #1
LINESAMPLERATE=50       #50
LINESCORINGTYPE=1       #0
MINLINELENGTH=4         #4
CONVERGENCEACCURACY=0.3 #0.2

CAMERA_MATRIX=np.load("Camera_Calibration_Mtx.npy")
CAMERA_DISTORTION=np.load("Camera_Calibration_Dist.npy")

def cycleImg(img):
    cv2.imshow('chessboard',img)
    if cv2.waitKey(0) & 0xFF:
        cv2.destroyAllWindows()

#Takes an image and transforms to a field of how chessboard-corner-like each pixel is
#Args - (greyscale integer image, how far to sample from each pixel, board rotation (False for straight, True for diagonal))
#Return - [uint corner mask]
def cornerChessboard(img, buffer=3, transpose=False):
    height, width = img.shape[:2]
    imgChess = np.zeros([height,width], dtype=int)

    for i in range(buffer, height-buffer):
        for j in range(buffer, width-buffer):
            score = 0
            score1 = 0

            score+=abs(img[i-buffer, j-buffer]-img[i-buffer, j+buffer])
            score+=abs(img[i-buffer, j+buffer]-img[i+buffer, j+buffer])
            score+=abs(img[i+buffer, j+buffer]-img[i+buffer, j-buffer])
            score+=abs(img[i+buffer, j-buffer]-img[i-buffer, j-buffer])
            score-=2*abs(img[i-buffer, j-buffer]-img[i+buffer, j+buffer])
            score-=2*abs(img[i-buffer, j+buffer]-img[i+buffer, j-buffer])
            score-=abs(img[i+buffer, j]-img[i, j+buffer])
            score-=abs(img[i, j+buffer]-img[i-buffer, j])
            score-=abs(img[i-buffer, j]-img[i, j-buffer])
            score-=abs(img[i, j-buffer]-img[i+buffer, j])
            score+=2*abs(img[i+buffer, j]-img[i-buffer, j])
            score+=2*abs(img[i, j+buffer]-img[i, j-buffer])

            if transpose:
                imgChess[i, j]=max(0,-1*score)
            else:
                imgChess[i, j]=max(0,score)

    return imgChess.astype("uint8")

#Averages nearby centroids into a single point
def getMergedCentroidClusters(centroids, distance=20):
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
    
#Takes an image and returns the chess board aligned edges in that image
#Args - (image as numpy array, pixel interval between checking pixel value along suspected edges, threshold floating point for what qualifies as an edge from 0-1)
#Return - [(x,y),(x1,y1),rho]
def getFilteredEdges(img, samplingRate, threshold, cornerType):
    imgTest=img.copy()

    if cornerType==0:
        cornerMap = cv2.cornerHarris(np.float32(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)),3,HARRISAPERTURE,0.02)
    elif cornerType==1:
        cornerMap = cornerChessboard(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype("int"))
    else:
        raise Exception('corner type undefined')
    cornerMap = cv2.threshold(cornerMap,HARRISTHRESHOLD*cornerMap.max() if cornerType==0 else CHESSTHRESHOLD*cornerMap.max(), 255,cv2.THRESH_BINARY)[1]
    cornerMap = np.uint8(cornerMap)

##    cornerMap = cv2.erode(cornerMap,np.ones((1,1)),iterations = 1)
    cornerMap = cv2.dilate(cornerMap,np.ones((4,4)),iterations = 1)

    #centroids = getMergedCentroidClusters(cv2.connectedComponentsWithStats(cornerMap)[3], MINCORNERDISTANCE)
    centroids = cv2.connectedComponentsWithStats(cornerMap)[3]

    height, width = img.shape[:2]
    distance = 0.1*max(img.shape[:2]) #Minimum distance between centroids to be viable for line check

    imgCanny = cv2.Canny(img,50,150,apertureSize = 3)
    cycleImg(imgCanny)
    imgCannyTest = cv2.cvtColor(imgCanny,cv2.COLOR_GRAY2RGB)
    imgCannyTest[cornerMap>0.01*cornerMap.max()]=[0,0,255]
    cycleImg(imgCannyTest)
    img = cv2.dilate(imgCanny,np.ones((CANNYDILATION,CANNYDILATION),np.uint8),iterations = 1)

    height-=1
    width-=1
    dist=0.1*min(height, width) #Distance for merging similar lines

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

            lineLength=0
            for i in range(samplingRate):
                x=int(edgeIntercept[0]+i*xStep)
                y=int(edgeIntercept[1]+i*yStep)
                pointValue=img[y, x]-CLEANEDGEBIAS*cv2.mean(img[max(0,y-CLEANEDGEBLOCKSIZE):min(height,y+CLEANEDGEBLOCKSIZE), max(0,x-CLEANEDGEBLOCKSIZE):min(width,x+CLEANEDGEBLOCKSIZE)])[0]
                if pointValue > 0:
                    lineLength+=1
                    if LINESCORINGTYPE == 0 or lineLength>=MINLINELENGTH:
                        lineValue += pointValue
                else:
                    lineLength=0

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

    maxLineValue=max(lines, key=lambda x: x[2])[2]*threshold
    lines = [[line[0], line[1], np.arctan2(line[0][1]-line[1][1], line[0][0]-line[1][0])%np.pi, set(), 0] for line in lines if line[2] >= maxLineValue]

#Fast Angle Filter
    
    for line in lines:
        for line1 in lines:
            if abs(line[2]-line1[2])%np.pi<0.01*np.pi:
                line[4]+=1
                continue

            #TODO intersection equation is wrong
                
            intersectX=((line1[0][0]*line1[1][1]-line1[0][1]*line1[1][0])*(line[0][0]-line[1][0])-(line1[0][0]-line1[1][0])*(line[0][0]*line[1][1]-line[0][1]*line[1][0]))/((line1[0][0]-line1[1][0])*(line[0][1]-line[1][1])-(line1[0][1]-line1[1][1])*(line[0][0]-line[1][0]))
            intersectY=((line1[0][0]*line1[1][1]-line1[0][1]*line1[1][0])*(line[0][1]-line[1][1])-(line1[0][1]-line1[1][1])*(line[0][0]*line[1][1]-line[0][1]*line[1][0]))/((line1[0][0]-line1[1][0])*(line[0][1]-line[1][1])-(line1[0][1]-line1[1][1])*(line[0][0]-line[1][0]))
            intersection = (int(intersectX), int(intersectY))
            if intersection is None:
                continue
            if intersection[0]>0 and intersection[0]<width and intersection[1]>0 and intersection[1]<height:
                continue
            line[3].add(intersection)


    for line in lines:
        bestScore=0
        for point in line[3]:
            accuracy = np.sqrt(abs(height//2-point[1])**2 + abs(width//2-point[0])**2)*CONVERGENCEACCURACY
            i=0
            for point1 in line[3]:
                if abs(point[0]-point1[0])<accuracy and abs(point[1]-point1[1])<accuracy:
                    i+=1
            if i>bestScore:
                bestScore=i
        line[4]+=bestScore

    maxAngleVariance=max(lines, key=lambda x: x[4])[4]*0.5
    lines = [(line[0], line[1], line[2]%np.pi) for line in lines if line[4] >= maxAngleVariance]

#Janky af, works well on mostly complete boards, poorly on boards with many missing edges


    

    for line in lines:
        cv2.line(imgTest,line[0],line[1],(0,0,255),2)
##        for line1 in lines:
##            if ((line1[0][0]-line1[1][0])*(line[0][1]-line[1][1])-(line1[0][1]-line1[1][1])*(line[0][0]-line[1][0])) != 0 and ((line1[0][0]-line1[1][0])*(line[0][1]-line[1][1])-(line1[0][1]-line1[1][1])*(line[0][0]-line[1][0])) != 0:
##                intersectX=((line1[0][0]*line1[1][1]-line1[0][1]*line1[1][0])*(line[0][0]-line[1][0])-(line1[0][0]-line1[1][0])*(line[0][0]*line[1][1]-line[0][1]*line[1][0]))/((line1[0][0]-line1[1][0])*(line[0][1]-line[1][1])-(line1[0][1]-line1[1][1])*(line[0][0]-line[1][0]))
##                intersectY=((line1[0][0]*line1[1][1]-line1[0][1]*line1[1][0])*(line[0][1]-line[1][1])-(line1[0][1]-line1[1][1])*(line[0][0]*line[1][1]-line[0][1]*line[1][0]))/((line1[0][0]-line1[1][0])*(line[0][1]-line[1][1])-(line1[0][1]-line1[1][1])*(line[0][0]-line[1][0]))
##                intersection = (int(intersectX), int(intersectY))
##                cv2.line(imgTest,(intersection[0],intersection[1]),(intersection[0],intersection[1]+5),(0,255,0),3)
    cycleImg(imgTest)

    return lines

def getVanishingPoint(lines):
    lines=[[line[0], line[1], line[2], set()] for line in lines]
    for line in lines:
        parallel=0
        for line1 in lines:
            if abs(line[2]-line1[2])%np.pi<0.01*np.pi:
                parallel+=1
                continue

            if parallel>=3:
                return None

            #TODO intersection equation refactored
                   
            intersectX=((line1[0][0]*line1[1][1]-line1[0][1]*line1[1][0])*(line[0][0]-line[1][0])-(line1[0][0]-line1[1][0])*(line[0][0]*line[1][1]-line[0][1]*line[1][0]))/((line1[0][0]-line1[1][0])*(line[0][1]-line[1][1])-(line1[0][1]-line1[1][1])*(line[0][0]-line[1][0]))
            intersectY=((line1[0][0]*line1[1][1]-line1[0][1]*line1[1][0])*(line[0][1]-line[1][1])-(line1[0][1]-line1[1][1])*(line[0][0]*line[1][1]-line[0][1]*line[1][0]))/((line1[0][0]-line1[1][0])*(line[0][1]-line[1][1])-(line1[0][1]-line1[1][1])*(line[0][0]-line[1][0]))
            intersection = (int(intersectX), int(intersectY))
            line[3].add(intersection)

    vanishingPoint=None
    for line in lines:
        bestScore=0
        for point in line[3]:
            accuracy = np.sqrt((np.mean((line[0][1], line[1][1]))//2-point[1])**2 + (np.mean((line[0][0], line[1][0]))//2-point[0])**2)*0.1
            i=0
            for point1 in line[3]:
                if abs(point[0]-point1[0])<accuracy and abs(point[1]-point1[1])<accuracy:
                    i+=1
            if i>bestScore:
                bestScore=i
                vanishingPoint = point

    return vanishingPoint

#TODO
#NEEDS BUG TESTING
#Takes an array of edges and fills in the missing lines, then removes the least conforming lines to produce a 10x10 grid
#Args - (line array in the form [(x, y), (x, y), rho])
#Return - [sorted vertical line array of length 10 in the form [(min x, min y), (max x, max y)], sorted horizontal line array of length 10 in the form [(min x, min y), (max x, max y)]
def getBoardLines(lines):
    lines.sort(key=lambda x: x[2])

    #TODO - Factor out redundant code
    
    lines1=None
    lines2=None
    gaps=[abs(line[2]-lines[(i-1)%len(lines)][2]) for i, line in enumerate(lines)]

    maxGap=max(gaps)
    maxGapIndex=gaps.index(maxGap)
    gaps[maxGapIndex]=0
    maxGap1=max(gaps)
    maxGapIndex1=gaps.index(maxGap1)

    
    if maxGap1 < 0.5*maxGap:
        lines1=lines[:maxGapIndex]
        lines2=lines[maxGapIndex:]
    else:
        lines1=lines[:maxGapIndex]
        lines2=lines[maxGapIndex:maxGapIndex1]
        lines1+=lines[maxGapIndex1:]

    vLines=None
    hLines=None

    if abs(90-np.mean([line[2] for line in lines1])) >= abs(90-np.mean([line[2] for line in lines2])):
        vLines=lines1
        hLines=lines2
    else:
        vLines=lines2
        hLines=lines1

    vLines = [(min(line[:2], key=lambda x: x[1]), max(line[:2], key=lambda x: x[1]), line[2]) for line in vLines]
    hLines = [(min(line[:2], key=lambda x: x[0]), max(line[:2], key=lambda x: x[0]), line[2]) for line in hLines]

    vLines.sort(key=lambda x: x[0][0])
    hLines.sort(key=lambda x: x[0][1])
    vLines.sort(key=lambda x: x[0][1])
    hLines.sort(key=lambda x: x[0][0])

    vVanishingPoint=getVanishingPoint(vLines)
    hVanishingPoint=getVanishingPoint(hLines)

    distances=[]
    if vVanishingPoint is not None:
        for i in range(len(vLines)-1):
            distances.append(np.mean(((vVanishingPoint[1] - vLines[i+1][0][1])/(vVanishingPoint[1] - vLines[i][0][1]),(vVanishingPoint[1] - vLines[i+1][1][1])/(vVanishingPoint[1] - vLines[i][1][1]))))
    else:
        for i in range(len(vLines)-1):
            distances.append(np.mean((vLines[i+1][0][1]/vLines[i][0][1],vLines[i+1][1][1]/vLines[i][1][1]))) 
               
    vDistances = distances.copy()
    distances.sort()
    maxDist=distances[-1]-distances[0]
    distances1 = [maxDist//distance for distance in distances]

    vSquareSize=None

    for i in range(len(distances)):
        if distances1.count(distances1[i])>=3:
           vSquareSize=distances[i]
           break
        

    distances=[]
    if hVanishingPoint is not None:
        for i in range(len(hLines)-1):
            distances.append(np.mean(((hVanishingPoint[0] - hLines[i+1][0][0])/(hVanishingPoint[0] - hLines[i][0][0]),(hVanishingPoint[0] - hLines[i+1][1][0])/(hVanishingPoint[0] - hLines[i][1][0]))))
    else:
        for i in range(len(hLines)-1):
            distances.append(np.mean((hLines[i+1][0][0]/hLines[i][0][0],hLines[i+1][1][0]/hLines[i][1][0])))

    hDistances = distances.copy()
    distances.sort()
    maxDist=distances[-1]-distances[0]
    distances1 = [maxDist//distance for distance in distances]

    hSquareSize=None

    for i in range(len(distances)):
        if distances1.count(distances1[i])>=3:
           hSquareSize=distances[i]
           break

    if vSquareSize is None or hSquareSize is None:
        return None

    #Fill in missing lines

    newLines=[]
    for i in range(len(vLines)-1):
        missingLines = int(round(vDistances[i]/vSquareSize))
        if missingLines <= 1:
            continue

        for i1 in range(missingLines):
            minXPos=vLines[i][0][0] + (vLines[i+1][0][0]-vLines[i+1][0][0])//missingLines
            maxXPos=vLines[i][1][0] + (vLines[i+1][1][0]-vLines[i+1][1][0])//missingLines
            minYPos=vLines[i][0][1] + (vLines[i+1][0][1]-vLines[i+1][0][1])//missingLines
            maxYPos=vLines[i][1][1] + (vLines[i+1][1][1]-vLines[i+1][1][1])//missingLines
            newLines.append(((minXPos, minYPos), (maxXPos, maxYPos)))

            #TODO
            #Doesn't yet fit lines to borders
            

    i=0
    while i < len(vLines)-1:
        if vDistances[i] < vSquareSize:
            del vLines[i]
            del vDistances[i]
            continue
        i+=1

    vLines+=newLines

    newLines=[]
    for i in range(len(hLines)-1):
        missingLines = int(round(hDistances[i]/hSquareSize))
        if missingLines == 1:
            continue

        for i1 in range(missingLines):
            minXPos=vLines[i][0][0] + (vLines[i+1][0][0]-vLines[i+1][0][0])//missingLines
            maxXPos=vLines[i][1][0] + (vLines[i+1][1][0]-vLines[i+1][1][0])//missingLines
            minYPos=vLines[i][0][1] + (vLines[i+1][0][1]-vLines[i+1][0][1])//missingLines
            maxYPos=vLines[i][1][1] + (vLines[i+1][1][1]-vLines[i+1][1][1])//missingLines
            newLines.append(((minXPos, minYPos), (maxXPos, maxYPos)))

            #TODO
            #Doesn't yet fit lines to borders
            

    i=0
    while i < len(hLines)-1:
        if hDistances[i] < hSquareSize:
            del hLines[i]
            del hDistances[i]
            continue
        i+=1

    hLines+=newLines

    vLines.sort(key=lambda x: x[0][0])
    hLines.sort(key=lambda x: x[0][1])
    vLines.sort(key=lambda x: x[0][1])
    hLines.sort(key=lambda x: x[0][0])
    
    return [line[:2] for line in vLines], [line[:2] for line in hLines]
        
#NEEDS BUG TESTING
def getPieces(img):
    img = [pix[3] for pix in img]
    return np.indicies(img.max())



#NEEDS BUG TESTING
#Takes an 2 arrays of parralel edges and an array of piece coordinates and returns a 8x8 matrix of board states
#Args - (vertical line array in the form [(x, y), (x, y)], horizontal line array in the form [(x, y), (x, y)], piece array in the form [(x, y), value])
#Return - [8x8 board state matrix]
def getBoardState(lines1, lines2, pieces):
    squares=np.zeros(8,8,1)

    i=0
    while i < len(lines1):
        for piece in pieces:
            if piece[0][0]>line1[i]:
                i1=0
                while i1 < len(lines2):
                    if piece[0][1]>line2[i1]:
                            squares[i, i1] = piece[1]

    return squares
    
        

for filename in os.listdir(PATH):
    img = cv2.imread(PATH + filename)
    img = img[:,img.shape[1]//2:]
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    cycleImg(img)
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(CAMERA_MATRIX,CAMERA_DISTORTION,(w,h),0,(w,h))
    img = cv2.undistort(img, CAMERA_MATRIX, CAMERA_DISTORTION, None, newcameramtx)
    cycleImg(img)
    lines=getFilteredEdges(img, LINESAMPLERATE, LINETHRESHOLD, CORNERDETECTIONTYPE)
##    vLines, hLines=getBoardLines(lines)
##    for line in hLines:
##        print(line)
##        cv2.line(img,line[0],line[1],(0,0,255),2)
##    cycleImg(img)
