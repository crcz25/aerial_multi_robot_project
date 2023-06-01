import cv2
import numpy as np

# Define the callback function for the trackbars
def on_change(x):
    # Apply Hough transform to detect circles
    dp = cv2.getTrackbarPos("dp", "Trackbars")
    minDist = cv2.getTrackbarPos("minDist", "Trackbars")
    param1 = cv2.getTrackbarPos("param1", "Trackbars")
    param2 = cv2.getTrackbarPos("param2", "Trackbars")
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist, param1=param1, param2=param2, minRadius=0, maxRadius=0)

    # If no circles were detected, exit the function
    if circles is None:
        return

    # Find the largest circle
    circles = np.uint16(np.around(circles))
    # largest_circle = circles[0,0]
    # for circle in circles[0,:]:
    #     if circle[2] > largest_circle[2]:
    #         largest_circle = circle

    # Draw the largest circle on the original image
    img_circle = img.copy()
    # cv2.circle(img_circle, (largest_circle[0], largest_circle[1]), largest_circle[2], (0, 255, 0), 2)
    # Draw all circles
    for circle in circles[0,:]:
        cv2.circle(img_circle, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
    # Display the original image with the largest circle drawn on it
    cv2.imshow("Largest circle", img_circle)

# Load the image and convert to grayscale
img = cv2.imread('1.jpg')
# Resize image
img = cv2.resize(img, (640, 480))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Create a window with trackbars
cv2.namedWindow("Trackbars")
cv2.createTrackbar("dp", "Trackbars", 1, 2, on_change)
cv2.createTrackbar("minDist", "Trackbars", 100, 200, on_change)
cv2.createTrackbar("param1", "Trackbars", 100, 500, on_change)
cv2.createTrackbar("param2", "Trackbars", 100, 500, on_change)

# Call the on_change function once to initialize the display
on_change(0)

# Wait for a key press and exit
cv2.waitKey(0)
cv2.destroyAllWindows()
