import cv2

# Load image in grayscale
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('complex_red.png', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

# Apply fixed threshold to convert to binary image
thresh_value = 10
ret, thresh = cv2.threshold(img, thresh_value, 255, cv2.THRESH_BINARY)

# Display the binary image
cv2.imshow('Binary Image', thresh)
cv2.waitKey(0)