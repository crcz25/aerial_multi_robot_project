import cv2
import numpy as np

# Load image
image = cv2.imread('shear.png', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
image = cv2.imread('drone_color.png', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
image = cv2.imread('canny_edge.png', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
image = cv2.imread('shear.png', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

# Find contours
# convert to binary image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Draw contours
output = image.copy()

# Filter the contours to only keep the ones that are likely to be gates
gate_contours = []
for contour in contours:
    # Compute the area and perimeter of the contour
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Compute the aspect ratio of the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)

    # Filter the contour based on its area, perimeter, and aspect ratio
    if area > 100 and perimeter > 50 and aspect_ratio > 0.5 and aspect_ratio < 1.5:
        gate_contours.append(contour)

for contour in gate_contours:
    # Compute the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)
    # Compute the height and width of the gate
    gate_height = h
    gate_width = w

    # Compute the aspect ratio of the gate
    aspect_ratio = gate_height / float(gate_width)

    # Determine the gate type based on its size and shape
    if gate_height > 100 and gate_width > 50 and aspect_ratio > 0.5 and aspect_ratio < 1.5:
        gate_type = "tall"
        # Draw the bounding box of the gate
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # Draw the type of the gate
        cv2.putText(output, gate_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    elif gate_height > 50 and gate_width > 100 and aspect_ratio > 1.5:
        gate_type = "wide"
        # Draw the bounding box of the gate
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # Draw the type of the gate
        cv2.putText(output, gate_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        gate_type = "unknown"
    


cv2.imshow('image', image)
cv2.imshow('output', output)
cv2.waitKey(0)
cv2.destroyAllWindows()