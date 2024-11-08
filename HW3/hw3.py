# Po-Shen Lee
# Date: 11/07/2024
# HW3 - Movement Detection and Contours (Enhanced for Significant Bounding Boxes)

import cv2

# Initialize video capture from the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize the first frame to None (to store the previous frame)
previous_frame = None

while True:
    # Read the current frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the current frame to grayscale for processing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Check if we have a previous frame to compare
    if previous_frame is None:
        # If no previous frame, set it and skip to the next iteration
        previous_frame = gray_frame
        continue

    # Calculate the absolute difference between the current frame and the previous frame
    frame_diff = cv2.absdiff(previous_frame, gray_frame)

    # Apply a binary threshold to the difference to get a binary image
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    # Find contours on the thresholded image
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours and find the largest one
    largest_contour = None
    max_area = 1500  # Set a minimum area threshold for significance

    for contour in contours:
        # Calculate the contour area
        area = cv2.contourArea(contour)
        # Only consider the contour if it meets the area threshold
        if area > max_area:
            max_area = area
            largest_contour = contour

    # Draw the bounding box for the largest significant contour, if found
    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame with bounding box and the thresholded frame for visualization
    cv2.imshow("Movement Detection", frame)
    cv2.imshow("Threshold", thresh)

    # Update the previous frame
    previous_frame = gray_frame

    # Press 'q' to quit the video stream
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture object and close display windows
cap.release()
cv2.destroyAllWindows()
