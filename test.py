import cv2
import mediapipe as mp
import time
import handTrackModule as htm
import math
import os
import numpy as np

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)


detector=htm.handDetector(detectionCon=0.65,maxHands=1)
folderPath = "PainterHeader"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
drawColor = (255, 0, 255)
brushThickness = 15
eraserThickness = 100
xp,yp=0,0

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]
imgCanvas = np.zeros((720, 1280, 3), np.uint8)


while True:

    #1import image
    success,img=cap.read()
    img=cv2.flip(img,1)
    

    #2 find landmarks
    img=detector.findHands(img)
    lmList=detector.findPosition(img)

    if(len(lmList)!=0):

        #tip of index and middle finger
        x1,y1=lmList[8][1:]
        x2,y2=lmList[12][1:]
    
        #3 check which finger up

        fingers=detector.fingersUp()
        #print(fingers)


        #4 check select mode 1st 2nd finger up
        if(fingers[1] and fingers[2]):
            #print("Selection mode")
            xp,yp=0,0
            
            if(y1<125):
                if(250<x1<450):
                    header=overlayList[0]
                    drawColor = (255, 0, 255)
                elif(550<x1<750):
                    header=overlayList[1]
                    drawColor = (255, 0, 0)
                elif(888<x1<950):
                    header=overlayList[2]
                    drawColor = (0, 255, 0)
                elif(1050<x1<1200):
                    header=overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img,(x1,y1-25),(x2,y2+25),drawColor,cv2.FILLED)


        #5 drawing mode 1st finger up
        if fingers[1] and fingers[2]==0:
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
            #print("Drawing")
            
            if(xp==0 and yp==0):
                xp,yp=x1,x2

            if drawColor == (0,0,0):
                cv2.line(img,(xp,yp),(x1,y1),drawColor,eraserThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,eraserThickness)
            else:
                cv2.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brushThickness)
            
            xp,yp=x1,y1


    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)


    
    img[0:125, 0:1280] = header
    #   img=cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("Image",img)
    #cv2.imshow("canvas",imgCanvas)
    cv2.waitKey(1)