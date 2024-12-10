import pytesseract
import cv2


img = cv2.imread('images/earth-C.JPG', 0)
# cv2.imshow('Original Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# Apply OCR
text = pytesseract.image_to_string(img)
print("Extracted Text:")
print(text)
lines = text.split('\n')

for line in lines:
    line = line.strip()
    if line:
        print(line)
