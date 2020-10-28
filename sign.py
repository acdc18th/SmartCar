import numpy as np
import cv2

def tracking():
    cap=cv2.VideoCapture(0)
    
    lower_yellow=np.array([110, 100, 100])
    upper_yellow=np.array([130, 255, 255])
    lower_green=np.array([50, 100, 100])
    upper_green=np.array([70, 255, 255])
    lower_red=np.array([-10, 100, 100])
    upper_red=np.array([10, 255, 255])

    while True:
        ret, frame=cap.read()
        hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #yellow_range=cv2.inRange(hsv, lower_yellow, upper_yellow)
        #green_range=cv2.inRange(hsv, lower_green, upper_green)
        red_range=cv2.inRange(hsv, lower_red, upper_red)

        #yellow_result=cv2.bitwise_and(frame, frame, mask=yellow_range)
        red_result=cv2.bitwise_and(frame, frame, mask=red_range)
        #green_result=cv2.bitwise_and(frame, frame, mask=green_range)

        cv2.imshow("Original", frame)
        cv2.imshow("Result",red_result)

        k = cv2.waitKey(30) & 0xff
        if k == 27: # press 'ESC' to quit
            break
