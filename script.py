import cv2
import numpy as np

# Eye and Face Classifiers
eyeCascade = cv2.CascadeClassifier('./Classifiers/eye.xml')
faceCascade = cv2.CascadeClassifier('./Classifiers/face.xml')

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    # Converting the frame to Monochromatic for applying accurate and efficient openCV operations
    frame2gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecting the face and eyes using the respective classifiers
    face = faceCascade.detectMultiScale(frame2gray, 1.3, 5)
    eyes = eyeCascade.detectMultiScale(frame2gray, 1.3, 5)

    # For generating the threshold image
    ret, mask = cv2.threshold(frame2gray, 100, 255, cv2.THRESH_BINARY)

    # Drawing the rectangle for indicating the face with a title "Face !"
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Face !", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)

    # Drawing the rectangle for each of the eyes with a title "Eyes !"
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Eyes !", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

    # Displaying the Face Detector and Threshold window
    cv2.imshow('Threshold Image !!!!', mask)
    cv2.imshow('Face Detectorrr !!!!', frame)

    # Exit the window when Esc button is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
