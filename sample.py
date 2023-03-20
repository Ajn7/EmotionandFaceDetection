
# import cv2
# from deepface import DeepFace
# from datetime import datetime

# # Load trained model
# face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# face_recognizer.read('trained_model.xml')

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# cap = cv2.VideoCapture(0)

# #file for storing
# f = open('emotion_labels.txt', 'a')

# # contiuous shooting 
# while True:
#     current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

#     for (x, y, w, h) in faces:
#         # Extract the face region from the frame
#         face_roi = gray[y:y+h, x:x+w]

#         # Recognize the face using the trained model
#         person_id, confidence = face_recognizer.predict(face_roi)

#         # If the confidence is high enough, display the recognized name on the frame
#         if confidence < 100:
#             recognized_name = f'arjun {person_id}'
#             cv2.putText(frame, recognized_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#         else:
#             recognized_name = 'unknown person'
#         # Draw a rectangle around the detected face
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#         # Analyze the emotion of the face
#         result = DeepFace.analyze(frame,actions="emotion",enforce_detection=False)
#         dominant_emotion = result[0]['dominant_emotion']
#         print(dominant_emotion)
#         data=dominant_emotion+' '+current_time+' '+recognized_name
#         # Add data to the file
#         f.write(data+'\n')
#         cv2.putText(frame,data,(x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

#     # Display the frame
#     cv2.imshow('Recognition', frame)

#     # Check if the user has pressed the 'q' key to quit
#     if cv2.waitKey(1) == ord('q'):
#         break

# # close all
# cap.release()
# cv2.destroyAllWindows()
# f.close()


import cv2
from deepface import DeepFace
from datetime import datetime

# Load trained model
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trained_model.xml')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

#file for storing
f = open('emotion_labels.txt', 'a')

# contiuous shooting 
while True:
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face_roi = gray[y:y+h, x:x+w]

        # Recognize the face using the trained model
        person_id, confidence = face_recognizer.predict(face_roi)

        # If the confidence is high enough, display the recognized name on the frame
        if confidence < 100:
            
            if person_id == 1:
                recognized_name = f'arjun'
                cv2.putText(frame, recognized_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            elif person_id == 2:
                recognized_name = f'akash'
                cv2.putText(frame, recognized_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                recognized_name = 'Unknown face'
                cv2.putText(frame, recognized_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            recognized_name = 'Unknown face'
            cv2.putText(frame, recognized_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Analyze the emotion of the face
        result = DeepFace.analyze(frame,actions="emotion",enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion']
        print(dominant_emotion)
        data=dominant_emotion+' '+current_time+' '+recognized_name
        # Add data to the file
        f.write(data+'\n')
        cv2.putText(frame,data,(x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Display the frame
    cv2.imshow('Recognition', frame)

    # Check if the user has pressed the 'q' key to quit
    if cv2.waitKey(1) == ord('q'):
        break

# close all
cap.release()
cv2.destroyAllWindows()
f.close()

