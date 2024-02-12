import cv2
import numpy as np

def getcontours(img,cThr=[100,100],showCanny=False,minArea=1000,filter=0,draw=False):
    imgGrey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGrey,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,cThr[0],cThr[1])
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=3)
    imgThre = cv2.erode(imgDial,kernel,iterations=2)
    imgThre_tmp = cv2.resize(imgThre,(0,0),None,0.5,0.5)
    if showCanny:cv2.imshow('Canny',imgThre_tmp)

    contours , hiearchy = cv2.findContours(imgThre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    finalcontours = []

    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.02*peri,True)
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    finalcontours.append([ len(approx),area,approx,bbox,i])
            else:
                finalcontours.append([len(approx), area, approx, bbox, i])
    finalcontours = sorted(finalcontours,key=lambda x:x[1] , reverse=True)

    if draw:
        for con in finalcontours:
            cv2.drawContours(img,con[4],-1,(0,0,255),3)

    return img , finalcontours

def reorder(points):
    print(points.shape)
    pointsNew=np.zeros_like(points)
    points = points.reshape((4,2))
    add = points.sum(1)
    pointsNew[0] = points[np.argmin(add)]
    pointsNew[3] = points[np.argmax(add)]
    diff=np.diff(points,axis=1)
    pointsNew[1] = points[np.argmin(diff)]
    pointsNew[2] = points[np.argmax(diff)]
    return pointsNew


def warpimg(img,points,w,h,pad=20):
    print('x')
    points = reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgwarp = cv2.warpPerspective(img,matrix,(w,h))
    imgwarp = imgwarp[pad:imgwarp.shape[0]-pad,pad:imgwarp.shape[1]-pad]

    return imgwarp

def findDist(pt1,pt2):
    return ((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)**0.5

