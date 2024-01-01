import numpy as np
import cv2

# location = 'E:/OpenCV/haarcascade_russian_plate_number.xml'
PlateClassifer = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

frameWidth = 640
frameHeight = 480
minArea = 500
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
# cap.set(10, 150)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    NumberPlate = PlateClassifer.detectMultiScale(gray, 1.1, 4)
    for(x,y,w,h) in NumberPlate:
        area = w * h
        if area > minArea:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 3)
            cv2.putText(frame, 'Number Plate Detected', (x,y), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)



    cv2.imshow('Window', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
