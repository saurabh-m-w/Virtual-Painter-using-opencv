import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

prevTime=0
currTime=0


while True:
    success,img=cap.read()
    
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLnd in results.multi_hand_landmarks:
            x1,y1,x2,y2=0,0,0,0
            for id,lm in enumerate(handLnd.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x * w),int(lm.y * h)
                #print(id,cx,cy)
                if(id==4):
                    x1=cx
                    y1=cy
                if(id==8):
                    x2,y2=cx,cy
            print((((x2 - x1 )**2) + ((y2-y1)**2) )**0.5)

            mpDraw.draw_landmarks(img,handLnd,mpHands.HAND_CONNECTIONS)

    currTime=time.time()
    fps=1/(currTime-prevTime)
    prevTime=currTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,255,),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)

