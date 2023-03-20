import cv2

# Load the trained face recognition model
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trained_model.xml')

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start the video capture
cap = cv2.VideoCapture(0)

# Start a loop to capture frames and recognize faces
while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    # Iterate over each detected face
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

    # Display the frame
    cv2.imshow('Recognition', frame)

    # Check if the user has pressed the 'q' key to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

