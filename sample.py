#with one frame
# import cv2
# from deepface import DeepFace

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# cap = cv2.VideoCapture(0)

# # Capture a single frame
# ret, frame = cap.read()

# # Find the dominant emotion in the frame
# result = DeepFace.analyze(frame, actions="emotion", enforce_detection=False)
# dominant_emotion = result[0]['dominant_emotion']
# #dominant_emotion = result['dominant_emotion']
# print('Dominant emotion:', dominant_emotion)

# # Draw a rectangle around the detected faces and label the dominant emotion
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# for (x, y, w, h) in faces:
#     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
#     cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

# # Display the result
# cv2.imshow('frame', frame)
# key = cv2.waitKey(0)

# # Release the camera and destroy the window
# cap.release()
# cv2.destroyAllWindows()

# # Close the window if the user pressed 'q'
# if key == ord('q'):
#     cv2.destroyAllWindows()
# face emotion detection in live camera

#with schedule
# import sched
# import time
# import cv2
# from deepface import DeepFace

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# cap = cv2.VideoCapture(0)
# def capture_emotions():
#     ret,frame = cap.read()
#     result = DeepFace.analyze(frame,actions="emotion",enforce_detection=False)
        
    
#     gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

#     faces = face_cascade.detectMultiScale(gray,1.1,4)
    
    
#     for (x,y,w,h) in faces:
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
    
#         dominant_emotion = result[0]['dominant_emotion']
#         print(dominant_emotion)
        
#         # Add the emotion label to the image
#         cv2.putText(frame,str(dominant_emotion) ,(x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

#         cv2.imshow('frame',frame)
   
#         if cv2.waitKey(1) & 0xff == ord('q'):
#             cap.release()
#             cv2.destroyAllWindows()
#             break
# s = sched.scheduler(time.time, time.sleep)
# def schedule_func(sc):
#     # execute my_func()
#     capture_emotions()
#     s.enter(15, 1, schedule_func, (sc,))

# s.enter(5, 1, schedule_func, (s,))
# cap.release()
# cv2.destroyAllWindows()

# import cv2
# from deepface import DeepFace
# from datetime import datetime


# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# cap = cv2.VideoCapture(0)
# with open('emotion_log.txt', 'a') as f:
#     while True:
#         current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         ret,frame = cap.read()
#         result = DeepFace.analyze(frame,actions="emotion",enforce_detection=False)
            
        
#         gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

#         faces = face_cascade.detectMultiScale(gray,1.1,4)
        
        
#         for (x,y,w,h) in faces:
#             cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
        
#             dominant_emotion = result[0]['dominant_emotion']
#             print(dominant_emotion)
#             data=dominant_emotion+' '+current_time
#             # Add the emotion label to the image
#             f.write(data)
#             cv2.putText(frame,data,(x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

#         cv2.imshow('frame',frame)

#         if cv2.waitKey(1) & 0xff == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()
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
# import time
# import cv2
# from deepface import DeepFace
# from datetime import datetime
# import threading

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# cap = cv2.VideoCapture(0)
# f = open('emotion_labels.txt', 'a')

# def capture_frames():
#     while True:
#         ret,frame = cap.read()
#         cv2.imshow('frame',frame)
#         if cv2.waitKey(1) & 0xff == ord('q'):
#             f.close()
#             cap.release()
#             cv2.destroyAllWindows()
#             break

# def write_emotions():
#     while True:
#         current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         ret,frame = cap.read()
#         result = DeepFace.analyze(frame,actions="emotion",enforce_detection=False)
#         gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray,1.1,4)
#         for (x,y,w,h) in faces:
#             cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
#             dominant_emotion = result[0]['dominant_emotion']
#             data=dominant_emotion+' '+current_time
#             f.write(data+'\n')
#             cv2.putText(frame,data,(x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
#         time.sleep(10)

# t1 = threading.Thread(target=capture_frames)
# t2 = threading.Thread(target=write_emotions)

# t1.start()
# t2.start()

#from jpeg
# import cv2
# from deepface import DeepFace

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# # Load the JPEG image
# img = cv2.imread('image.jpg')

# # Find the dominant emotion in the image
# result = DeepFace.analyze(img, actions="emotion", enforce_detection=False)
# dominant_emotion = result[0]['dominant_emotion']
# print('Dominant emotion:', dominant_emotion)

# # Draw a rectangle around the detected faces and label the dominant emotion
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# for (x, y, w, h) in faces:
#     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
#     cv2.putText(img, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

# # Display the result
# cv2.imshow('image', img)
# key = cv2.waitKey(0)

# # Close the window if the user pressed 'q'
# if key == ord('q'):
#     cv2.destroyAllWindows()