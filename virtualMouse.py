import cv2
import mediapipe as mp
import time
import handTrackModule as htm
import math
import autopy
import numpy as np


wCam, hCam = 1280, 720
frameR=300
smoothening=7
plocX,plocY=0,0
clocX,clocY=0,0

wScreen,hScreen=autopy.screen.size()
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector=htm.handDetector(detectionCon=0.65,maxHands=1)

while True:
    success,img=cap.read()
    img = cv2.flip(img, 1)
    
    #step1 find hand landmarks
    img=detector.findHands(img)
    lmList = detector.findPosition(img)

    #2get the tip of index and middle finger
    #cv2.rectangle(img,(frameR,frameR),(wCam-frameR,hCam-frameR),(255,0,255),2)
    if(len(lmList)!=0):
        x1,y1=lmList[8][1:]
        x2,y2=lmList[12][1:]
        
        #3 check ehich finger is up
        fingers=detector.fingersUp()

        #4 if only index finger moving mode
        if(fingers[1]==1 and fingers[2]==0):
            #5 convert coordinates
            
            x3=np.interp(x1,(frameR,wCam-frameR),(0,wScreen))
            y3=np.interp(y1,(200,hCam-200),(0,hScreen))

            #6 smoothen values
            clocX=plocX+(x3-plocX)/smoothening
            clocY=plocY+(y3-plocY)/smoothening
            #7 move mouse
            autopy.mouse.move(wScreen-clocX,clocY)
            cv2.circle(img,(x1,y1),15,(255,0,255),(cv2.FILLED))
            plocX,plocY=clocX,clocY

        #8 click mode index and middle fingers up
        if(fingers[1]==1 and fingers[2]==1):
            length,img,lineinfo=detector.findDistance(8,12,img)
            print(length)
            if(length<40):
                cv2.circle(img,(lineinfo[4],lineinfo[5]),15,(0,255,0),(cv2.FILLED))
                autopy.mouse.click()
    #9 
    cv2.imshow("Virtual Mouse", img)
    if cv2.waitKey(1) == ord('q'):
        break
    if cv2.getWindowProperty("Virtual Mouse", cv2.WND_PROP_VISIBLE)<1:
        break
    
cap.release()
cv2.destroyAllWindows()