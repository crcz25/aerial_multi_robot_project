import cv2 as cv
import numpy as np
import random as rng

def background_foreground_separator(image, lower_range, upper_range):
    # Create a copy of the image
    image = image.copy()
    # image = cv.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    # Convert the image to HSV
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # Generate the mask
    mask = cv.inRange(image, lower_range, upper_range)
    # Apply morphological operations to remove noise and fill gaps
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)
    # mask = cv.erode(mask, kernel, iterations=2)
    # mask = cv.dilate(mask, kernel, iterations=2)
    # Extract the objects from the image
    image = cv.bitwise_and(image, image, mask=mask)
    # Convert the image to grayscale
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Equalize the histogram of the image
    # image = cv.equalizeHist(image)
    # Blur the image to reduce noise
    image = cv.GaussianBlur(image, (5, 5), 0)
    # Apply Adaptive Thresholding
    # image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    # Apply Otsu's thresholding
    _, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return image

def gate_detector(image):
    # Create a copy of the image
    image = image.copy()
    # If the image is not in grayscale, convert it
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Calculate the contours of the image 
    contours, hierarchies = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Reconvert the image to display the contours with color
    if len(image.shape) != 3:
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

    cv.drawContours(image, contours, -1, (255, 0, 0), 3)
    cnt = contours[2]
    epsilon = cv.arcLength(cnt,True)
    approx = cv.approxPolyDP(cnt,epsilon * 0.01,True)
    cv.drawContours(image, [approx], -1, (0, 0, 255), 3)
    filtered_contours = []
    # Draw the contours based on the hierarchy of the contours
    for i in range(len(contours)):
        # cv.drawContours(image, contours, i, (255, 0, 0), 3)
        if hierarchies[0][i][3] == -1:
            cv.drawContours(image, contours, i, (0, 255, 0), 3)
            pass
        else:
            cv.drawContours(image, contours, i, (0, 0, 255), 3)
            filtered_contours.append(contours[i])
    # # Find the contours that are rectangular
    # gates = []
    # for contour in filtered_contours:
    #     epsilon = 0.1*cv.arcLength(contour, True)
    #     approx = cv.approxPolyDP(contour, epsilon, True)
    #     perimeter = cv.arcLength(contour,True)
    #     cv.drawContours(image, [approx], 0, (255, 0, 0), 3)
    #     cv.drawContours(image, [perimeter], 0, (0, 0, 255), 3)
    #     # Create a bounding box around the contour
    #     x, y, w, h = cv.boundingRect(contour)
    #     # Calculate the aspect ratio of the bounding box
    #     aspect_ratio = float(w) / h
    #     # Calculate the area of the bounding box
    #     area = w * h
    #     # Normalize the area of the bounding box
    #     area = area / (image.shape[0] * image.shape[1])
    #     # If the aspect ratio is between 0.75 and 1.0, then the contour is a gate
    #     if 0.75 < aspect_ratio < 2.0:
    #         # Calculate the center of the gate
    #         cx = x + w // 2
    #         cy = y + h // 2
    #         # Add the gate to the list of gates
    #         gates.append((x, y, w, h, cx, cy, area))
    #         # Draw the bounding box around the gate
    #         cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
    #         # Draw the center of the gate
    #         cv.circle(image, (cx, cy), 10, (0, 255, 0), -1)
    # # Sort the gates based on the area of the bounding box
    # gates = sorted(gates, key=lambda x: x[6], reverse=True)
    # Draw the gates on the image
    # for gate in self.gates:
        # x, y, w, h, cx, cy, _ = gate
        # cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
        # cv.circle(image, (cx, cy), 10, (255, 0, 0), -1)
    return image

def stop_sign_detector(image, method = cv.RETR_TREE):
    # print("Checking stop sign")
    image = image.copy()
    # If the image is not in grayscale, convert it
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Detect the stop signs
    # detection = self.detector.detectMultiScale(image, scaleFactor=1.5, minNeighbors=10, minSize=(50, 50))
    # Calculate the contours of the image 
    contours, hierarchies = cv.findContours(image, method, cv.CHAIN_APPROX_SIMPLE)
    # Reconvert the image to display the contours with color
    if len(image.shape) != 3:
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    print("Number of contours: ", len(contours))
    print(hierarchies)
    # Draw the contours based on the hierarchy of the contours
    filtered_contours = []
    for i in range(len(contours)):
        # cv.drawContours(image, contours, i, (255, 0, 0), 3)
        print(hierarchies[0][i])
        if hierarchies[0][i][0] == -1:
            # cv.drawContours(image, contours, i, (0, 255, 0), 3)
            filtered_contours.append(contours[i])
    # Draw the bounding boxes around the stop signs
    stop_signs = []
    for cnt in filtered_contours:
        # Create a bounding box around the contour
        x, y, w, h = cv.boundingRect(cnt)
        # Calculate the area of the bounding box
        area = w * h
        # Normalize the area of the bounding box
        area = area / (image.shape[0] * image.shape[1])
        # Calculate the center of the stop sign
        cx = x + w // 2
        cy = y + h // 2
        # Add the stop sign to the list of stop signs
        stop_signs.append((x, y, w, h, cx, cy, area))
        # Draw the bounding box
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Draw the center of the stop sign
        cv.circle(image, (cx, cy), 10, (0, 255, 0), -1)
    # Sort the stop signs based on the area of the bounding box
    stop_signs = sorted(stop_signs, key=lambda x: x[6], reverse=True)
    return image

def detect_contours(image):
    # Create a copy of the image
    image = image.copy()
    # If the image is not in grayscale, convert it
    #Apply thresholding to create a binary image
    # thresh = cv.threshold(image, 127, 255, cv.THRESH_BINARY)[1]
    # Calculate the contours of the image 
    contours, hierarchies = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print(hierarchies)
    # Reconvert the image to display the contours with color
    if len(image.shape) != 3:
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

    # Draw the contours based on the hierarchy of the contours
    filtered_contours = []
    for i in range(len(contours)):
        # cv.drawContours(image, contours, i, (255, 0, 0), 3)
        if hierarchies[0][i][3] == -1:
            # cv.drawContours(image, contours, i, (0, 255, 0), 3)
            pass
        else:
            # cv.drawContours(image, contours, i, (0, 0, 255), 3)
            filtered_contours.append(contours[i])
    # Draw the contours
    # cv.drawContours(image, contours, -1, (255, 0, 0), 3)
    # Iterate over the contours and find the ones that match the gate shape and size
        # Find the contours that are rectangular
    gates = []
    for contour in filtered_contours:
        # Find the bounding box of the contour
        (x, y), radius = cv.minEnclosingCircle(contour)
        # Check if the circle has a certain radius
        if 50 < radius < 800:
            print("Radius: ", radius)
            center = (int(x),int(y))
            radius = int(radius)
            area = np.pi * radius ** 2
            area = area / (image.shape[0] * image.shape[1])
            # Draw the circle
            cv.circle(image,center,radius,(0,255,0),3)
            # Draw the center of the circle
            cv.circle(image,center,10,(0,0,255),-1)
            gates.append([center[0], center[1], radius, area])

    # Sort the gates based on the area of the bounding box
    gates = sorted(gates, key=lambda x: x[3], reverse=True)

    return image

def main():
    # Read the image
    image = cv.imread('4.jpg', cv.IMREAD_COLOR)
    image = cv.imread('5.jpg', cv.IMREAD_COLOR)
    # Convert to negative
    # Resize the image
    image = cv.resize(image, (0, 0), fx=0.2, fy=0.2)
    image = cv.imread('shear.png', cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
    image = cv.imread('Drone Image_screenshot_10.04.2023.png', cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
    image = cv.imread('Drone Image_screenshot_11.04.2023.png', cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
    # image = cv.imread('Drone Image post processing_screenshot_05.04.2023.png', cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
    
    # image = cv.imread('drone_color.png', cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
    # image = cv.imread('stop_sign_2.png', cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
    # image = cv.imread('shear.png', cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
    # image = cv.imread('stop_sign.png', cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
    # image = cv.imread('canny_edge.png', cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
    # # Create the lower and upper range for the color of the gate
    # lower_range = (30, 50, 50)
    # upper_range = (90, 255, 255)
    # Separate the background from the foreground
    # image = background_foreground_separator(image, lower_range, upper_range)
    # lower_range_red_1 = (0, 25, 25)
    # upper_range_red_1 = (10, 255, 255)
    # lower_range_red_2 = (160, 25, 25)
    # upper_range_red_2 = (180, 255, 255)
    # image_stop_sign_1 = background_foreground_separator(image, lower_range_red_1, upper_range_red_1)
    # image_stop_sign_2 = background_foreground_separator(image, lower_range_red_2, upper_range_red_2)
    # image_stop_sign = cv.add(image_stop_sign_1, image_stop_sign_2)
    # Detect the gate
    # image = detect_contours(image)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # image = cv.bitwise_not(image)
    image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    # Equalize the histogram of the image
    # cv.imshow('og', image)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_eq = clahe.apply(image)
    # image_eq = cv.equalizeHist(image)
    # cv.imshow('1', image_eq)
    image_blur = cv.GaussianBlur(image_eq, (5, 5), 0)
    # cv.imshow('2', image_blur)
    # cv.imshow('image grey', image)
    # # thresh = cv.threshold(image, 90, 120, cv.THRESH_BINARY)[1]
    # opImgx = cv.Sobel(image_blur, cv.CV_8U, 0, 1, ksize=3)
    # opImgy = cv.Sobel(image_blur, cv.CV_8U, 1, 0, ksize=3)
    # grad = cv.bitwise_or(opImgx, opImgy)
    # Detecte edges with laplacian
    # grad = cv.Laplacian(image_blur, cv.CV_8U, ksize=3)
    # Detecte edges with laplacian of gaussian
    # grad = cv.Laplacian(image_blur, cv.CV_8U, ksize=3)
    LoG = cv.Laplacian(image_blur, cv.CV_16S, ksize=5)
    LoG = cv.convertScaleAbs(LoG)
    LoG = cv.medianBlur(LoG, 5)
    cv.imshow('3', LoG)
    # thresh = cv.adaptiveThreshold(LoG, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    _, thresh = cv.threshold(LoG, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # cv.imshow('4', thresh)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    dilation = cv.dilate(thresh,kernel,iterations = 1)
    close = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=1)
    # cv.imshow('5', close)
    # opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=1)
    # close = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=1)
    # opening = cv.morphologyEx(close, cv.MORPH_OPEN, kernel, iterations=1)
    # opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    # opening = cv.morphologyEx(opening, cv.MORPH_OPEN, kernel)
    # opening = cv.morphologyEx(opening, cv.MORPH_OPEN, kernel)
    # closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    # opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
    # close = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    # erode = cv.erode(close, kernel)
    # opening = cv.morphologyEx(erode, cv.MORPH_OPEN, kernel)
    # cv.imshow('3', opening)
    contours, hierarchies = cv.findContours(close, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    image_f = cv.cvtColor(close, cv.COLOR_GRAY2RGB)
    # cv.drawContours(image_f, contours, -1, (255, 0, 0), 3)

    boxes = [None]*len(contours)
    minEllipse = [None]*len(contours)
    circles = [None]*len(contours)

    for i, c in enumerate(contours):
        rect = cv.minAreaRect(c)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        boxes[i] = box
        circles[i] = cv.minEnclosingCircle(c)
        if c.shape[0] > 5:
            minEllipse[i] = cv.fitEllipse(c)
    
    # filter out duplicated bounding boxes or circles on the same location
    filtered_boxes = []
    filtered_circles = []
    for i in range(len(boxes)):
        is_duplicate = False
        for j in range(i+1, len(boxes)):
            dist = np.sqrt((boxes[i].mean(axis=0)[0]-boxes[j].mean(axis=0)[0])**2 + (boxes[i].mean(axis=0)[1]-boxes[j].mean(axis=0)[1])**2)
            if dist < 10:  # adjust the threshold as needed
                is_duplicate = True
                break
        if not is_duplicate:
            filtered_boxes.append(boxes[i])
            filtered_circles.append(circles[i])

    # draw the filtered bounding boxes and circles on the image
    output = image_f.copy()
    for box in filtered_boxes:
        cv.drawContours(output, [box], 0, (0, 255, 0), 2)
    for circle in filtered_circles:
        cv.circle(output, (int(circle[0][0]), int(circle[0][1])), int(circle[1]), (0, 255, 0), 2)
    cv.imshow('output', output)

    filtered_contours = []
    for i in range(len(contours)):
        # cv.drawContours(image, contours, i, (255, 0, 0), 3)
        if hierarchies[0][i][3] == -1:
            # cv.drawContours(image, contours, i, (0, 255, 0), 3)
            pass
        else:
            # cv.drawContours(image, contours, i, (0, 0, 255), 3)
            filtered_contours.append(contours[i])
    filtered_contours = sorted(filtered_contours, key=cv.contourArea, reverse=True)
    filtered_contours = filtered_contours[:5]
    drawing = image_f.copy()
    gates = []


    for i, c in enumerate(filtered_contours):
        if c.shape[0] > 5:
            color = (0, 255, 0)
            rect = cv.minAreaRect(c)
            (cx_circle, cy_circle), radius_circle = cv.minEnclosingCircle(c)
            # (cx_ellipse, cy_ellipse), (a_ellipse, b_ellipse), angle_elipse = cv.fitEllipse(c)
            # Filter the ellipses that are closser to be a circle
            # ratio = min(a_ellipse, b_ellipse) / max(a_ellipse, b_ellipse)
            # Filter the contours that are too small to be a gate
            if 100 < radius_circle < 1000:
                pass
                # Calculate ratio of the bounding box
                # check if the center of the circle coincides with the center of the bounding box
            if np.abs(cx_circle - rect[0][0]) < 1 and np.abs(cy_circle - rect[0][1]) < 1:
                box = np.intp(cv.boxPoints(rect))
                # Draw the figures
                cv.drawContours(drawing, [box], 0, color, 3)
                cv.circle(drawing, (int(cx_circle), int(cy_circle)), int(radius_circle), color, 3)
                # cv.ellipse(drawing, (int(cx_ellipse), int(cy_ellipse)), (int(a_ellipse), int(b_ellipse)), angle_elipse, 0, 360, color, 3)
                area = np.pi * radius_circle ** 2
                area = area / (image.shape[0] * image.shape[1])
                gates.append((cx_circle, cy_circle, radius_circle, area))
    # # Sort the gates based on the area of the bounding box
    gates = sorted(gates, key=lambda x: x[3], reverse=True)
    cv.imshow('drawing', drawing)

    # Iterate through contours and find minimum-area rectangles
    # for cnt in contours:
        # approx = cv.approxPolyDP(cnt, 0.009 * cv.arcLength(cnt, True), True)
        # draws boundary of contours.
        # cv.drawContours(image_f, [approx], 0, (0, 0, 255), 5)

    #     # Used to flatted the array containing
    #     # the co-ordinates of the vertices.
    #     x = approx.ravel()[0]
    #     y = approx.ravel()[1] - 5
    #     # if len(approx) == 3:
    #         # cv.putText( image, "Triangle", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0) )
    #     if len(approx) == 4 :
    #         x, y , w, h = cv.boundingRect(approx)
    #         aspectRatio = float(w)/h
    #         print(aspectRatio)
    #         if aspectRatio >= 0.95 and aspectRatio < 1.05:
    #             cv.putText(image, "square", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
    #         else:
    #             cv.putText(image, "rectangle", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
    #     # elif len(approx) == 5 :
    #     #     cv.putText(image, "pentagon", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    #     # elif len(approx) == 10 :
    #     #     cv.putText(image, "star", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    #     else:
    #         cv.putText(image, "circle", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))

        # n = approx.ravel()
        # i = 0

        # for j in n:
        #     if i % 2 == 0:
        #         x = n[i]
        #         y = n[i + 1]

        #         # String containing the co-ordinates.
        #         string = str(x) + " " + str(y)

        #         if i == 0:
        #             # text on topmost co-ordinate.
        #             cv.putText(image, "Arrow tip", (x, y),
        #                         cv.FONT_HERSHEY_COMPLEX_SMALL,
        #                         1, (255, 0, 0), 1)

        #         else:
        #             # text on remaining co-ordinates.
        #             cv.putText(image, string, (x, y),
        #                         cv.FONT_HERSHEY_COMPLEX_SMALL,
        #                         1, (0, 255, 0), 1)
        #     i = i + 1


    # cv.imshow('final', image_f)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()