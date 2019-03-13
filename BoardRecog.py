import cv2
import numpy as np
from os import listdir

import random as rand

#Constants
PATH='C:/Users/Matt/Desktop/Chess Robot/Testboards/'
DEBUG=False

                        #RECOMMENDED
HARRISAPERTURE=3        #3
CORNERDETECTIONTYPE=0   #0 (HARRIS)
HARRISTHRESHOLD=0.01    #0.16
CHESSTHRESHOLD=0.8      #0.5
LINETHRESHOLD=0.2       #0.5
MINCORNERDISTANCE=10    #10
CANNYDILATION=5         #3
CLEANEDGEBLOCKSIZE=10   #10
CLEANEDGEBIAS=2.0       #1
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
#Args - (greyscale integer image, how far to sample from each pixel)
#Return - [uint corner mask]
def cornerChessboard(img, buffer=2):
    height, width = img.shape[:2]
    imgChess = np.zeros([height,width], dtype=int)

    for i in range(buffer, height-buffer):
        for j in range(buffer, width-buffer):
            score = 0

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

            imgChess[i, j]=score

    if np.mean(imgChess) > 0:
        imgChess*=-1
    imgChess[imgChess < 0] = 0

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

#Args - (Tuple representing point on line in form (pointX, pointY), Tuple representing point on line in form (point1X, point1Y))
#Return - boolean truth of divergence, x coord, y coord
def getIntersection(line, line1):
    line = line[:2]
    line1 = line1[:2]

    det = (line1[0][0]-line1[1][0])*(line[0][1]-line[1][1])-(line1[0][1]-line1[1][1])*(line[0][0]-line[1][0])

    if det == 0:
        return False, None, None

    intersectX=((line1[0][0]*line1[1][1]-line1[0][1]*line1[1][0])*(line[0][0]-line[1][0])-(line1[0][0]-line1[1][0])*(line[0][0]*line[1][1]-line[0][1]*line[1][0]))//det
    intersectY=((line1[0][0]*line1[1][1]-line1[0][1]*line1[1][0])*(line[0][1]-line[1][1])-(line1[0][1]-line1[1][1])*(line[0][0]*line[1][1]-line[0][1]*line[1][0]))//det

    return True, intersectX, intersectY
    
#Takes an image and returns the chess board aligned edges in that image
#Args - (image as numpy array, pixel interval between checking pixel value along suspected edges, threshold floating point for what qualifies as an edge from 0-1)
#Return - [(x,y),(x1,y1),rho]
def getFilteredEdges(img, samplingRate, threshold, cornerType):
    imgTest=img.copy()

    height, width = img.shape[:2]
    height-=1
    width-=1

    if cornerType==0:
        cornerMap = cv2.cornerHarris(np.float32(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)),3,HARRISAPERTURE,0.02)
    elif cornerType==1:
        cornerMap = cornerChessboard(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype("int"))
    else:
        raise Exception('corner type undefined')
    cornerMap = cv2.threshold(cornerMap,HARRISTHRESHOLD*cornerMap.max() if cornerType==0 else CHESSTHRESHOLD*cornerMap.max(), 255,cv2.THRESH_BINARY)[1]
    cornerMap = np.uint8(cornerMap)

    cornerMap = cv2.erode(cornerMap,np.ones((1,1)),iterations = 1)
    cornerMap = cv2.dilate(cornerMap,np.ones((4,4)),iterations = 1)
    
    centroids = cv2.connectedComponentsWithStats(cornerMap)[3]
    
    distance = 0.2*max(img.shape[:2]) #Minimum distance between centroids to be viable for line check

    imgCanny = cv2.Canny(img,50,150,apertureSize = 3)
    if DEBUG:
        cycleImg(imgCanny)
    if DEBUG:
        imgCannyTest = cv2.cvtColor(imgCanny,cv2.COLOR_GRAY2RGB)
        imgCannyTest[cornerMap>0]=[0,0,255]
        cycleImg(imgCannyTest)
        mask=np.zeros((height+1, width+1), dtype=int)
        for corner in centroids:
            mask[int(corner[1]),int(corner[0])]=255
        mask = cv2.dilate(mask.astype("uint8"),np.ones((4,4)),iterations = 1)
        imgCannyTest[mask==255]=[255,0,0]
        cycleImg(imgCannyTest)
    img = cv2.dilate(imgCanny,np.ones((CANNYDILATION,CANNYDILATION),np.uint8),iterations = 1)

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
                        if lineValue > line[3]:
                            lines[i]=(edgeIntercept, edgeIntercept1, np.arctan2(edgeIntercept[1]-edgeIntercept1[1], edgeIntercept[0]-edgeIntercept1[0])%np.pi, lineValue)
                        else:
                            break
                i+=1
            if flag:
                lines.append((edgeIntercept, edgeIntercept1, np.arctan2(edgeIntercept[1]-edgeIntercept1[1], edgeIntercept[0]-edgeIntercept1[0])%np.pi, lineValue))
                
    lines = set(lines)

    redundantLines=[]
    for line in lines:
        for line1 in lines:
            ret, intersectX, intersectY = getIntersection(line, line1)
            if not ret or abs(line[2]-line1[2])%np.pi>0.2*np.pi:
                continue
            if intersectX>0 and intersectX<width and intersectY>0 and intersectY<height:
                redundantLines.append(min(line, line1, key = lambda x: x[3]))
        
    maxLineValue=max(lines, key=lambda x: x[3])[3]*threshold
    lines = [[line[0], line[1], line[2], set(), 0] for line in lines if line[3] >= maxLineValue and line not in redundantLines]
    

#Fast Angle Filter
    
    for line in lines:
        for line1 in lines:
            if abs(line[2]-line1[2])%np.pi<0.05*np.pi:
                line[4]+=1
                continue
                
            intersection = getIntersection(line, line1)[1:]
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

    minAngleScore=max(lines, key=lambda x: x[4])[4]*0.5
    lines = [(line[0], line[1], line[2]%np.pi) for line in lines if line[4] >= minAngleScore]

#Janky af, works well on mostly complete boards, poorly on boards with many missing edges


    
    if DEBUG:
        for line in lines:
            cv2.line(imgTest,line[0],line[1],(0,0,255),2)
            for line1 in lines:
                if ((line1[0][0]-line1[1][0])*(line[0][1]-line[1][1])-(line1[0][1]-line1[1][1])*(line[0][0]-line[1][0])) != 0 and ((line1[0][0]-line1[1][0])*(line[0][1]-line[1][1])-(line1[0][1]-line1[1][1])*(line[0][0]-line[1][0])) != 0:
                    intersectX=((line1[0][0]*line1[1][1]-line1[0][1]*line1[1][0])*(line[0][0]-line[1][0])-(line1[0][0]-line1[1][0])*(line[0][0]*line[1][1]-line[0][1]*line[1][0]))/((line1[0][0]-line1[1][0])*(line[0][1]-line[1][1])-(line1[0][1]-line1[1][1])*(line[0][0]-line[1][0]))
                    intersectY=((line1[0][0]*line1[1][1]-line1[0][1]*line1[1][0])*(line[0][1]-line[1][1])-(line1[0][1]-line1[1][1])*(line[0][0]*line[1][1]-line[0][1]*line[1][0]))/((line1[0][0]-line1[1][0])*(line[0][1]-line[1][1])-(line1[0][1]-line1[1][1])*(line[0][0]-line[1][0]))
                    intersection = (int(intersectX), int(intersectY))
                    cv2.line(imgTest,(intersection[0],intersection[1]),(intersection[0],intersection[1]+5),(0,255,0),3)
        cycleImg(imgTest)

    return lines

#Returns the avaerage convergence point of the collection of lines passed
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
            
            intersection = getIntersection(line, line1)[1:]
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
#NEEDS BUG TESTING - for boards with vanishing points
#Takes an array of edges and fills in the missing lines, then removes the least conforming lines to produce a 10x10 grid
#Args - (line array in the form [(x, y), (x, y), rho])
#Return - [sorted vertical line array of length 10 in the form [(min x, min y), (max x, max y)], sorted horizontal line array of length 10 in the form [(min x, min y), (max x, max y)]
def getBoardLines(lines):
##    lines=[]
##    
##    height, width = img.shape[:2]
##
##    height, width = height - 1, width - 1
##
##    hRange = list(range(9))
##    wRange = list(range(9))
##
##    rand.shuffle(hRange)
##    rand.shuffle(wRange)
##
##    direction = [rand.randrange(2) for i in hRange]
##    for i in hRange:
##        if rand.randrange(6)!=0:
##            lines.append(((int(i*width/8), 0 if direction[i] == 0 else height), (int(i*width/8), 0 if direction[i] == 1 else height), np.pi*0.5))
##
##    direction = [rand.randrange(2) for i in wRange]
##    for i in wRange:
##        if rand.randrange(6)!=0:
##            lines.append(((0 if direction[i] == 0 else width, int(i*height/8)), (0 if direction[i] == 1 else width, int(i*height/8)), 0))
    
    lines.sort(key=lambda x: x[2])

    #TODO - Factor out redundant code (although probably not a good idea if it would make this method any harder to read than it already is)
    
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

    if not lines1 or not lines2:
        return False, None, None
    

    if abs(90-np.mean([line[2] for line in lines1])) >= abs(90-np.mean([line[2] for line in lines2])):
        vLines=lines2
        hLines=lines1
    else:
        vLines=lines1
        hLines=lines2

    vLines = [(min(line[:2], key=lambda x: x[1]), max(line[:2], key=lambda x: x[1]), line[2]) for line in vLines]
    hLines = [(min(line[:2], key=lambda x: x[0]), max(line[:2], key=lambda x: x[0]), line[2]) for line in hLines]

    vLines.sort(key=lambda x: x[0][0])
    hLines.sort(key=lambda x: x[0][1])
    vLines.sort(key=lambda x: x[0][1])
    hLines.sort(key=lambda x: x[0][0])

    lines=None
    del lines

    if DEBUG:
        imgTest=img.copy()
        for line in vLines:
            cv2.line(imgTest,line[0],line[1],(0,0,255),2)
        for line in hLines:
            cv2.line(imgTest,line[0],line[1],(255,0,0),2)

        cycleImg(imgTest)

    
    vVanishingPoint=getVanishingPoint(vLines)
    hVanishingPoint=getVanishingPoint(hLines)

    distances=[]
    #Distances to vanishing point calculated as ratios to conserve perspective
    if hVanishingPoint is not None:
        for i in range(len(vLines)-1):
            distances.append(np.mean(((hVanishingPoint[0] - vLines[i+1][0][0])/(hVanishingPoint[0] - vLines[i][0][0]),(vVanishingPoint[0] - vLines[i+1][1][0])/(hVanishingPoint[0] - vLines[i][1][0]))))
    else:
        for i in range(len(vLines)-1):
            distances.append(np.mean((vLines[i+1][0][0]-vLines[i][0][0],vLines[i+1][1][0]-vLines[i][1][0])))
               
    vDistances = distances.copy()
    distances.sort()
    if hVanishingPoint is not None:
        maxDist=distances[-1]-distances[0]
    else:
        maxDist=np.mean((vLines[-1][0][0]-vLines[0][0][0],vLines[-1][1][0]-vLines[0][1][0]))
    distances1 = [maxDist//(distance+1) for distance in sorted(distances, reverse=True)]

    vSquareSize=None

    for i in range(len(distances)):
        if distances1.count(distances1[i])>=3:
           vSquareSize=maxDist/distances1[i]
           break
        

    distances=[]
    #Distances to vanishing point calculated as ratios due conserve perspective
    if vVanishingPoint is not None:
        for i in range(len(hLines)-1):
            distances.append(np.mean(((vVanishingPoint[1] - hLines[i+1][0][1])/(vVanishingPoint[1] - hLines[i][0][1]),(vVanishingPoint[1] - hLines[i+1][1][1])/(vVanishingPoint[1] - hLines[i][1][1]))))
    else:
        for i in range(len(hLines)-1):
            distances.append(np.mean((hLines[i+1][0][1]-hLines[i][0][1],hLines[i+1][1][1]-hLines[i][1][1])))

    hDistances = distances.copy()
    distances.sort()
    if hVanishingPoint is not None:
        maxDist=distances[-1]-distances[0]
    else:
        maxDist=np.mean((hLines[-1][0][1]-hLines[0][0][1],hLines[-1][1][1]-hLines[0][1][1]))
    distances1 = [maxDist//distance for distance in sorted(distances, reverse=True)]

    hSquareSize=None

    for i in range(len(distances)):
        if distances1.count(distances1[i])>=3:
           hSquareSize=maxDist/distances1[i]
           break

    if vSquareSize is None or hSquareSize is None:
        return False, None, None

    #Fill in missing lines

    newLines=[]
    for i in range(len(vLines)-1):
        missingLines = int(round(vDistances[i]/vSquareSize))
        if missingLines <= 1:
            continue

        for i1 in range(missingLines):
            minXPos=vLines[i][0][0] + i1*(vLines[i+1][0][0]-vLines[i][0][0])//missingLines
            maxXPos=vLines[i][1][0] + i1*(vLines[i+1][1][0]-vLines[i][1][0])//missingLines
            minYPos=vLines[i][0][1] + i1*(vLines[i+1][0][1]-vLines[i][0][1])//missingLines
            maxYPos=vLines[i][1][1] + i1*(vLines[i+1][1][1]-vLines[i][1][1])//missingLines
            newLines.append(((minXPos, minYPos), (maxXPos, maxYPos)))

            #TODO
            #Doesn't yet fit lines to borders
            

    i=0
    while i < len(vLines)-1:
        if vDistances[i] < vSquareSize and False:   #BUG with board cleanup
            del vLines[i]
            del vDistances[i]
            continue
        i+=1

    vLines+=newLines

    newLines=[]
    for i in range(len(hLines)-1):
        missingLines = int(round(hDistances[i]/hSquareSize))
        if missingLines <= 1:
            continue

        for i1 in range(missingLines):
            minXPos=hLines[i][0][0] + i1*(hLines[i+1][0][0]-hLines[i][0][0])//missingLines
            maxXPos=hLines[i][1][0] + i1*(hLines[i+1][1][0]-hLines[i][1][0])//missingLines
            minYPos=hLines[i][0][1] + i1*(hLines[i+1][0][1]-hLines[i][0][1])//missingLines
            maxYPos=hLines[i][1][1] + i1*(hLines[i+1][1][1]-hLines[i][1][1])//missingLines
            newLines.append(((minXPos, minYPos), (maxXPos, maxYPos)))

            #TODO
            #Doesn't yet fit lines to borders
            

    i=0
    while i < len(hLines)-1:
        if hDistances[i] < hSquareSize and False:   #BUG with board cleanup
            del hLines[i]
            del hDistances[i]
            continue
        i+=1

    hLines+=newLines

    vLines.sort(key=lambda x: x[0][0])
    hLines.sort(key=lambda x: x[0][1])
    vLines.sort(key=lambda x: x[0][1])
    hLines.sort(key=lambda x: x[0][0])
    
    return True, [line[:2] for line in vLines], [line[:2] for line in hLines]
        
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
    
        

for filename in listdir(PATH):
    img = cv2.imread(PATH + filename)
    img = img[:,img.shape[1]//2:]
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    if DEBUG:
        cycleImg(img)
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(CAMERA_MATRIX,CAMERA_DISTORTION,(w,h),0,(w,h))
    img = cv2.undistort(img, CAMERA_MATRIX, CAMERA_DISTORTION, None, newcameramtx)
    if DEBUG:
        cycleImg(img)
    lines=getFilteredEdges(img, LINESAMPLERATE, LINETHRESHOLD, CORNERDETECTIONTYPE)
    ret, vLines, hLines=getBoardLines(lines)
    if DEBUG and ret:
        imgTest=img.copy()
        for line in hLines:
            cv2.line(imgTest,line[0],line[1],(0,0,255),2)
        cycleImg(imgTest)

        imgTest=img.copy()
        for line in vLines:
            cv2.line(imgTest,line[0],line[1],(0,0,255),2)
        cycleImg(imgTest)
