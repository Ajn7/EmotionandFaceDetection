import cv2
import os
import numpy as np

# Set the path to the directory containing the dataset
dataset_path = 'dataset/'

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create an LBPH face recognizer object
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Initialize empty lists for the training data and labels
training_data = []
labels = []

# Loop over each subdirectory in the dataset directory
for subdir in os.listdir(dataset_path):
    # Get the path to the subdirectory
    subdir_path = os.path.join(dataset_path, subdir)

    # Get the label for the subdirectory
    label = int(subdir.split('_')[1])

    # Loop over each image file in the subdirectory
    for filename in os.listdir(subdir_path):
        # Get the path to the image file
        file_path = os.path.join(subdir_path, filename)

        # Load the image file and convert it to grayscale
        image = cv2.imread(file_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect the face in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        # Iterate over each detected face (there should be only one)
        for (x, y, w, h) in faces:
            # Extract the face region from the image
            face_roi = gray[y:y+h, x:x+w]

            # Add the face region and label to the training data and labels lists
            training_data.append(face_roi)
            labels.append(label)

# Train the LBPH face recognizer on the training data and labels
recognizer.train(training_data, np.array(labels))

# Save the trained model to a file
recognizer.save('trained_model.xml')