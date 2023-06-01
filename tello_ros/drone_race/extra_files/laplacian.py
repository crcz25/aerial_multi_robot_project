import numpy as np
import cv2 as cv

def show_image(name, image):
    cv.imshow(name, image)
    cv.waitKey(0)

def gate_detector(image):
    # Create a copy of the image
    image = image.copy()
    show_image("original", image)
    # Convert the image to grayscale
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Normalize the image
    image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    # Equalize the histogram of the image
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    # Blur the image to reduce noise
    image = cv.GaussianBlur(image, (3, 3), 0)
    # Detecte edges with laplacian of gaussian
    image = cv.Laplacian(image, cv.CV_64F, ksize=3)
    show_image("Laplacian", image)
    image = cv.convertScaleAbs(image)
    show_image("Laplacian Converted", image)
    image = cv.addWeighted(image, 1.5, image, 0, 0)
    # brightness = np.sum(image) / (image.shape[0] * image.shape[1])
    # minimum_brightness = 50
    # alpha = max(minimum_brightness / brightness, 1)
    show_image("addWeighted", image)
    # Apply median blur to reduce noise
    image = cv.medianBlur(image, 3)
    show_image("blur", image)
    # Apply Otsu's thresholding
    image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 7, -7)
    # _, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    show_image("threshold", image)
    # Apply morphological operations to remove noise and fill gaps
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    # image = cv.erode(image, kernel,iterations = 1)
    image = cv.morphologyEx(image , cv.MORPH_CLOSE, kernel, iterations=1)
    # image = cv.erode(image, kernel,iterations = 1)
    # image = cv.dilate(image, kernel,iterations = 1)
    # image = cv.morphologyEx(image , cv.MORPH_OPEN, kernel, iterations=1)
    show_image("morphologyEx", image)
    # Calculate the contours of the image 
    contours, hierarchies = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Reconvert the image to display the contours with color
    image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

    filtered_contours = contours
    # filtered_contours = sorted(filtered_contours, key=cv.contourArea, reverse=True)
    # filtered_contours = filtered_contours[:5]
    # # filtered_contours = []
    # for i in range(len(contours)):
    #     cv.drawContours(image, contours, i, (255, 0, 0), 3)
    #     if hierarchies[0][i][3] == -1:
    #         cv.drawContours(image, contours, i, (0, 255, 0), 3)
    #         pass
    #     else:
    #         cv.drawContours(image, contours, i, (0, 0, 255), 3)
    #         filtered_contours.append(contours[i])

    # cv.drawContours(image, contours, -1, (255, 0, 0), 3)

    gates = []
    boxes = [None]*len(filtered_contours)
    circles = [None]*len(filtered_contours)

    for i, c in enumerate(filtered_contours):
        rect = cv.minAreaRect(c)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        boxes[i] = box
        circles[i] = cv.minEnclosingCircle(c)

    # filter out duplicated bounding boxes or circles on the same location
    filtered_boxes = []
    filtered_circles = []
    for i in range(len(boxes)):
        is_duplicate = False
        for j in range(i+1, len(boxes)):
            dist = np.linalg.norm(boxes[i] - boxes[j])
            dist_normalized = dist / (image.shape[0] / 2)
            # print("dist DUPLICATED: ", dist_normalized)
            if dist_normalized < 0.5:
                is_duplicate = True
                break
        if not is_duplicate:
            filtered_boxes.append(boxes[i])
            filtered_circles.append(circles[i])

    for i, box, circle in zip(range(len(boxes)), boxes, circles):
        (cx_circle, cy_circle), radius_circle = circle
        x, y, w, h = cv.boundingRect(box)
        cx_box = x + w/2
        cy_box = y + h/2
        radius_circle_normalized = radius_circle / (image.shape[0] / 2)
        print("radius_circle_normalized: ", radius_circle_normalized)
        if 0.2 < radius_circle_normalized < 0.4:
        # if True:
            dis_x = np.abs(cx_circle - cx_box)
            dis_x_normalized = dis_x / (image.shape[1] / 2)
            dis_y = np.abs(cy_circle - cy_box)
            dis_y_normalized = dis_y / (image.shape[0] / 2)
            print("dis_x_normalized: ", dis_x_normalized)
            print("dis_y_normalized: ", dis_y_normalized)
            # if dis_x_normalized < 0.1 and dis_y_normalized < 0.1:
            if True:
                # Draw the bounding box
                cv.drawContours(image, [box], 0, (0, 255, 0), 3)
                # Draw the circle
                cv.circle(image, (int(cx_circle), int(cy_circle)), int(radius_circle), (0, 0, 255), 3)
                area = np.pi * radius_circle ** 2
                area = area / (image.shape[0] * image.shape[1])
                gates.append((int(cx_circle), int(cy_circle), radius_circle, area))


    # Sort the gates based on the area of the bounding box
    gates = sorted(gates, key=lambda x: x[3], reverse=True)
    # gates = gates[:1]
    # Draw the gates on the image
    # for gate in gates:
    #     cx, cy, radius, _ = gate
    #     cv.circle(image, (cx, cy), 10, (255, 0, 0), -1)
    #     cv.circle(image, (cx, cy), int(radius), (0, 255, 0), 3)
    return image

image = cv.imread("Drone Image_screenshot_11.04.2023.png", cv.IMREAD_ANYDEPTH | cv.IMREAD_ANYCOLOR)
res = gate_detector(image)
show_image("res", res)
cv.destroyAllWindows()