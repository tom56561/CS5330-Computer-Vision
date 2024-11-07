# Po-Shen Lee
# Date: 10/22/2024
# Lab - Simultaneous Video Streams with Canny Edge Detection

import cv2

# Initialize the webcam video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to grayscale (Canny edge detector works on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detector
    edges = cv2.Canny(gray, 100, 200)

    # Display the original frame in one window
    cv2.imshow('Original Video Stream', frame)

    # Display the Canny edge detector output in another window
    cv2.imshow('Canny Edge Detector Video Stream', edges)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()
