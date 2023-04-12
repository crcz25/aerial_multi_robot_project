import cv2 as cv
import numpy as np
from cv_bridge import CvBridgeError

class Mixin:
    def show_image(self, title, image, resize=False, width=640, height=480):
        # Resize the image
        if resize:
            image = cv.resize(image, (width, height))
        cv.imshow(title, image)
        cv.waitKey(1)

    def background_foreground_separator(self, image, lower_range, upper_range):
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

    def gate_detector(self, image):
        # Create a copy of the image
        image = image.copy()
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
        # Convert the image to absolute values
        image = cv.convertScaleAbs(image)
        image = cv.addWeighted(image, 1.5, image, 0, 0)
        # self.show_image("laplacian", image)
        # Apply median blur to reduce noise
        image = cv.medianBlur(image, 3)
        # self.show_image("blur", image)
        # Apply Otsu's thresholding
        image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 7, -7)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        image = cv.morphologyEx(image , cv.MORPH_CLOSE, kernel, iterations=1)
        image = cv.morphologyEx(image, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)), iterations=1)
        image = cv.erode(image, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)), iterations=1)
        image = cv.dilate(image, cv.getStructuringElement(cv.MORPH_RECT, (5, 5)), iterations=1)
        # image = cv.dilate(image, kernel, iterations=1)
        # _, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # self.show_image("threshold", image)
        # Apply morphological operations to remove noise and fill gaps
        # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        # image = cv.dilate(image, kernel,iterations = 1)
        # self.show_image("dilate", image)
        # image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel, iterations=1)
        # Calculate the contours of the image 
        contours, hierarchies = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # filtered_contours = []
        # for i in range(len(contours)):
        #     # cv.drawContours(image, contours, i, (255, 0, 0), 3)
        #     if hierarchies[0][i][3] == -1:
        #         # cv.drawContours(image, contours, i, (0, 255, 0), 3)
        #         pass
        #     else:
        #         # cv.drawContours(image, contours, i, (0, 0, 255), 3)
        #         filtered_contours.append(contours[i])

        filtered_contours = contours
        filtered_contours = sorted(filtered_contours, key=cv.contourArea, reverse=True)
        filtered_contours = filtered_contours[:5]

        # Reconvert the image to display the contours with color
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

        self.gates = []
        boxes = [None]*len(filtered_contours)
        circles = [None]*len(filtered_contours)

        for i, c in enumerate(filtered_contours):
            rect = cv.minAreaRect(c)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            boxes[i] = box
            circles[i] = cv.minEnclosingCircle(c)

        # Filter out duplicated bounding boxes
        # filtered_boxes = []
        # filtered_circles = []
        # for i in range(len(boxes)):
        #     is_duplicate = False
        #     x, y, w, h = cv.boundingRect(boxes[i])
        #     cx_box_i = x + w/2
        #     cy_box_i = y + h/2
        #     for j in range(i+1, len(boxes)):
        #         x, y, w, h = cv.boundingRect(boxes[j])
        #         cx_box_j = x + w/2
        #         cy_box_j = y + h/2
        #         dist = np.linalg.norm(np.array([cx_box_i, cy_box_i]) - np.array([cx_box_j, cy_box_j]))
        #         # If they are too close, then they are the same
        #         if dist < 80:
        #             is_duplicate = True
        #             break
        #     if not is_duplicate:
        #         filtered_boxes.append(boxes[i])
        #         filtered_circles.append(circles[i])

        for i, box, circle in zip(range(len(boxes)), boxes, circles):
            (cx_circle, cy_circle), radius_circle = circle
            x, y, w, h = cv.boundingRect(box)
            cx_box = x + w/2
            cy_box = y + h/2
            aspect_ratio = min(w, h) / max(w, h)
            # print("radius_circle_normalized: ", radius_circle_normalized)
            # if 0.25 < radius_circle_normalized < 0.75:
            # if True:
            if 0.8 < aspect_ratio < 1.0 and 80 < radius_circle < 400:
                print("radius_circle: ", radius_circle)
                print("aspect_ratio: ", aspect_ratio)
                dis_x = np.abs(cx_circle - cx_box)
                dis_x_normalized = dis_x / (image.shape[1] / 2)
                dis_y = np.abs(cy_circle - cy_box)
                dis_y_normalized = dis_y / (image.shape[0] / 2)
                # print("dis_x_normalized: ", dis_x_normalized)
                # print("dis_y_normalized: ", dis_y_normalized)
                area_circle = np.pi * radius_circle**2
                area_circle_normalized = area_circle / (image.shape[0] * image.shape[1])
                print("area_circle: ", area_circle)
                print("area_circle_normalized: ", area_circle_normalized)
                if dis_x_normalized < 0.15 and dis_y_normalized < 0.15 and area_circle_normalized < 0.55:
                # if True:
                    # Draw the bounding box
                    cv.drawContours(image, [box], 0, (0, 255, 0), 3)
                    # Draw the circle
                    # cv.circle(image, (int(cx_circle), int(cy_circle)), int(radius_circle), (0, 0, 255), 3)
                    # cv.putText(image, str(i), (int(cx_circle), int(cy_circle)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # cv.putText(image, str(i), (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # cv.putText(image, str(f"Radius: {radius_circle:.2f}"), (x, y+40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # cv.putText(image, str(f"Area: {area_circle:.2f}"), (x, y+100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    self.gates.append((int(cx_circle), int(cy_circle), radius_circle, area_circle_normalized))

        # Sort the gates based on the area of the bounding box
        self.gates = sorted(self.gates, key=lambda x: x[3], reverse=True)
        self.gates = self.gates[:1]
        # Draw the gate on the image
        for gate in self.gates:
            cx, cy, radius, area = gate
            cv.circle(image, (cx, cy), 10, (255, 0, 0), -1)
            cv.circle(image, (cx, cy), int(radius), (255, 0, 0), 3)
            cv.putText(image, "Gate", (cx - 20, cy - 20), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv.putText(image, "Area: {:.2f}".format(area), (cx - 20, cy + 20), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv.putText(image, "Radius: {:.2f}".format(radius), (cx - 20, cy + 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv.putText(image, "Center: ({}, {})".format(cx, cy), (cx - 20, cy + 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        return image

    def generate_grid(self, image):
        # Create a copy of the image
        image = image.copy()
        # If the image is not in RGB, convert it
        if len(image.shape) != 3:
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)

        # Divide the image into 3 rows and 3 columns
        rows, cols, _ = image.shape
        row_step = rows // 3
        col_step = cols // 3

        # Draw the grid
        for i in range(1, 3):
            cv.line(image, (0, i * row_step), (cols, i * row_step), (255, 255, 0), 5)
            cv.line(image, (i * col_step, 0), (i * col_step, rows), (255, 255, 0), 5)
        
        # Add a dot to the center of the image
        cv.circle(image, (cols // 2, rows // 2), 10, (0, 0, 255), -1)

        # Draw a line from the center of the image to the center of the gate
        if len(self.gates) > 0:
            # Draw a line from the center of the image to the center of the gate
            first_gate = self.gates[0]
            cx, cy, _, _ = first_gate
            cv.line(image, (cx, cy), (cols // 2, rows // 2), (0, 0, 255), 5)
        if len(self.stop_signs) > 0:
            # Draw a line from the center of the image to the center of the stop sign
            stop_sign = self.stop_signs[0]
            x, y, w, h, cx, cy, _ = stop_sign
            cv.line(image, (cx, cy), (cols // 2, rows // 2), (0, 0, 255), 5)
        return image

    def stop_sign_detector(self, image, method = cv.RETR_TREE):
        # print("Checking stop sign")
        image = image.copy()
        # If the image is not in grayscale, convert it
        if len(image.shape) == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            # Equalize the histogram of the image
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(image)
            # Blur the image to reduce noise
            image = cv.GaussianBlur(image, (3, 3), 0)
        # Detect the stop signs
        detection = self.detector.detectMultiScale(image, scaleFactor = 1.25, minNeighbors = 7, minSize = (80, 80), maxSize = (500, 500))
        # Calculate the contours of the image 
        # contours, hierarchies = cv.findContours(image, method, cv.CHAIN_APPROX_SIMPLE)
        # Reconvert the image to display the contours with color
        if len(image.shape) != 3:
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        # Draw the contours based on the hierarchy of the contours
        # filtered_contours = []
        # for i in range(len(contours)):
            # cv.drawContours(image, contours, i, (255, 0, 0), 3)
            # if hierarchies[0][i][0] == -1:
                # cv.drawContours(image, contours, i, (0, 255, 0), 3)
                # filtered_contours.append(contours[i])
        # Draw the bounding boxes around the stop signs
        self.stop_signs = []
        for cnt in detection:
            # Create a bounding box around the contour
            x, y, w, h = cnt
            # Calculate the area of the bounding box
            area = w * h
            # Normalize the area of the bounding box
            area = area / (image.shape[0] * image.shape[1])
            # Calculate the center of the stop sign
            cx = x + w // 2
            cy = y + h // 2
            # Add the stop sign to the list of stop signs
            self.stop_signs.append((x, y, w, h, cx, cy, area))
        # Sort the stop signs based on the area of the bounding box
        self.stop_signs = sorted(self.stop_signs, key=lambda x: x[6], reverse=True)
        if len(self.stop_signs) > 0:
            # Draw a line from the center of the image to the center of the stop sign
            stop_sign = self.stop_signs[0]
            x, y, w, h, cx, cy, _ = stop_sign
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Draw the center of the stop sign
            cv.circle(image, (cx, cy), 10, (0, 255, 0), -1)
        return image

    def image_sub_callback(self, data):
        # print("Image received")
        try:
            # Convert your ROS Image message to OpenCV2
            self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
