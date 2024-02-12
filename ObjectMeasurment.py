import cv2
import utilis
import numpy as np

##############################

webcam = True
path = 'test2.jpg'
cap = cv2.VideoCapture(0)
cap.set(10,160)
cap.set(3,1920)
cap.set(4,1080)

scale = 3
wP = 210 * scale
hP = 297 * scale


while True:
    if webcam:success,img = cap.read()
    else: img = cv2.imread(path)

    img , conts=utilis.getcontours(img,showCanny=True,cThr=[175,175],
                                           minArea=5000,filter=4,draw=True)

    if len(conts) !=0 :
        biggest = conts[0][2]
        #print(biggest)
        imgwarp = utilis.warpimg(img, biggest, wP , hP)
        img2, conts2 = utilis.getcontours(imgwarp, cThr=[50, 50],
                                        minArea=2000,draw=True)
        if len(conts2)!=0:
            for obj in conts2:
                cv2.polylines(img2,[obj[2]],True,(0,255,0),2)
                nPoints = utilis.reorder(obj[2])
                nW = utilis.findDist(nPoints[0][0]//scale , nPoints[1][0]//scale)
                nH = utilis.findDist(nPoints[0][0] // scale, nPoints[2][0] // scale)
                print(f"W={nW} H={nH}")

        cv2.imshow('A4',img2)



    img = cv2.resize(img,(0,0),None,0.5,0.5)
    cv2.imshow('Original',img)
    cv2.waitKey(1)
