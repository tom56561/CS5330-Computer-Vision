# Po-Shen Lee
# Date: 11/05/2024
# Lab 7 - Face + Eye Detection

import cv2

# Load the Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


def detect_and_display(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around each eye
            cv2.rectangle(roi_color, (ex, ey),
                          (ex + ew, ey + eh), (0, 0, 255), 2)

            # Invert the colors within the eye region, 20 points
            eye_region = roi_color[ey:ey + eh, ex:ex + ew]
            inverted_eye = cv2.bitwise_not(eye_region)
            roi_color[ey:ey + eh, ex:ex + ew] = inverted_eye

    cv2.imshow('Face and Eye Detection with Inverted Eyes', frame)


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    detect_and_display(frame)

    # Press 'Esc' to exit
    if cv2.waitKey(10) == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
