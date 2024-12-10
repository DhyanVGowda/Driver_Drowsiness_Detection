import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

# Initialize sound mixer
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load Haar cascades
face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

# Labels for predictions
lbl = ['Close', 'Open']

# Load the model
model = load_model('models/cnnCat2.keras')  # Update to the correct model path
path = os.getcwd()

# Start webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces and eyes
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    # Draw rectangles around detected eyes
    for (x, y, w, h) in right_eye:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for (x, y, w, h) in left_eye:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with detected eyes
    cv2.imshow('Detected Eyes', frame)

    # Skip frame if eyes are not detected
    if len(right_eye) == 0 or len(left_eye) == 0:
        continue
    
    # Process right eye
    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255.0
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)

        # Debugging: Visualize the right eye image
        cv2.imshow('Right Eye', r_eye[0])  # Show the processed right eye
        cv2.waitKey(1)  # Add a small delay to view the image

        rpred_probs = model.predict(r_eye)
        print(f"Right Eye Probabilities: {rpred_probs}")
        rpred = np.argmax(rpred_probs, axis=1)
        break

    # Process left eye
    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255.0
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)

        # Debugging: Visualize the left eye image
        cv2.imshow('Left Eye', l_eye[0])  # Show the processed left eye
        cv2.waitKey(1)  # Add a small delay to view the image

        lpred_probs = model.predict(l_eye)
        print(f"Left Eye Probabilities: {lpred_probs}")
        lpred = np.argmax(lpred_probs, axis=1)
        break

    # Debugging predictions
    print(f"Right Eye Prediction: {rpred}, Left Eye Prediction: {lpred}")

    if rpred[0] == 0 and lpred[0] == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score = max(score - 1, 0)
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score > 15:
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        try:
            sound.play()
        except:
            pass
        thicc = thicc + 2 if thicc < 16 else thicc - 2
        thicc = max(thicc, 2)
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()