import time
import cv2
from deepface import DeepFace
from datetime import datetime


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
f = open('emotion_labels.txt', 'a')
while True:
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ret,frame = cap.read()
    result = DeepFace.analyze(frame,actions="emotion",enforce_detection=False)
            
        
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.1,4)
        
        
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
        
        dominant_emotion = result[0]['dominant_emotion']
        print(dominant_emotion)
        data=dominant_emotion+' '+current_time
        # Add the emotion label to the image
        f.write(data+'\n')
        cv2.putText(frame,data,(x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xff == ord('q'):
        f.close()
        break
    time.sleep(10)

cap.release()
cv2.destroyAllWindows()

