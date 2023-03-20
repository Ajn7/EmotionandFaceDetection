import cv2
import os

# Set the ID of the person you want to recognize
person_id = 1

# Set the path to the directory where you want to save the images
save_path = f'dataset/arjun_{person_id}'

# Create the directory if it doesn't exist
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start the video capture
cap = cv2.VideoCapture(0)

# Initialize a counter for the number of images captured
count = 0

# Start a loop to capture images
while count < 10:
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

        # Save the face image with a unique filename
        filename = f'arjun_{person_id}_{count}.jpg'
        file_path = os.path.join(save_path, filename)
        cv2.imwrite(file_path, face_roi)

        # Increment the counter
        count += 1

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Capture', frame)

    # Check if the user has pressed the 'q' key to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()