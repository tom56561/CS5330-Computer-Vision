# Eddie Lee
# Date: 10/30/2024
# HW3 - Movement Detection and Contours

import cv2

# Initialize video capture (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Read the first frame and convert it to grayscale
ret, prev_frame = cap.read()
if not ret:
    print("Error: Failed to capture initial frame.")
    cap.release()
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Loop over frames to process video
while True:
    # Read the current frame
    ret, curr_frame = cap.read()
    if not ret:
        print("Error: Failed to capture current frame.")
        break

    # Convert current frame to grayscale
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference between previous and current frame
    frame_diff = cv2.absdiff(prev_gray, curr_gray)

    # Threshold the difference image to create a binary image
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around significant contours
    for contour in contours:
        # Filter out small contours to reduce noise
        area = cv2.contourArea(contour)
        if area > 500:  # Adjust this threshold based on the desired sensitivity
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(curr_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the current frame with bounding boxes
    cv2.imshow("Movement Detection", curr_frame)
    cv2.imshow("Thresholded Difference", thresh)

    # Update the previous frame to the current frame
    prev_gray = curr_gray.copy()

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
