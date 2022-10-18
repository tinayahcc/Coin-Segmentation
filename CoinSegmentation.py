from typing import Counter
import cv2
import numpy as np
from numpy.core.numeric import count_nonzero

cap = cv2.VideoCapture('M5 Term1\image\Coin.mp4')

while (cap.read()):
    check , frame = cap.read() 
    coin_pic=frame[:1080,0:1920]

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray_blur=cv2.GaussianBlur(gray,(15,15),0)
    thresh=cv2.adaptiveThreshold(gray_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,1)
    kernel=np.ones((3,3),np.uint8)
    closing=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations=4)
    
    result_img=closing.copy()
    contours,hierachy=cv2.findContours(result_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    counter=0

    for cnt in contours:
        area=cv2.contourArea(cnt)
        if area<5000 or area>35000:
            continue
        ellipse=cv2.fitEllipse(cnt)
        cv2.ellipse(coin_pic,ellipse,(0,255,0),5)
        counter+=1

    cv2.putText(coin_pic,str(counter),(10,100),cv2.FONT_HERSHEY_COMPLEX,4,(255,0,0),2,cv2.LINE_4)
    cv2.imshow('Show coin',coin_pic)

    if cv2.waitKey(1) & 0xFF == ord('e'):
        break
    
cap.release() 
cv2.destroyAllWindows()