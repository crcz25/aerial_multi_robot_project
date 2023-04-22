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
        image = cv.cvtColor(image, cv.COLOR_HSV2BGR)
        # Equalize the histogram of the image
        # image = cv.equalizeHist(image)
        # Blur the image to reduce noise
        image = cv.GaussianBlur(image, (5, 5), 0)
        # Apply Adaptive Thresholding
        # image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
        # Apply Otsu's thresholding
        # _, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # self.show_image("background removal", image)
        return image

    def edge_detector(self, image):
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
        return image

    def gate_detector(self, image):
        # Create a copy of the image
        image = image.copy()
        image = self.edge_detector(image)
        # Calculate the contours of the image 
        contours, hierarchies = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # Sort the contours based on the area of the bounding box
        contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)[:10]
        # Reconvert the image to display the contours with color
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        # Find the gate
        self.gates = []
        for contour in contours:
            # Approximate the contour with a polygon
            epsilon = 0.01*cv.arcLength(contour,True)
            approx = cv.approxPolyDP(contour,epsilon,True)
            # Check if the approximated shape has 4 sides (rectangle gate)
            if len(approx) == 4 and cv.isContourConvex(approx):
                # Draw the contour of the square on the original image
                x, y, w, h = cv.boundingRect(approx)
                # Calculate the area of the gate
                area = cv.contourArea(contour)
                # Calculate the center of the gate
                cx = x + w / 2
                cy = y + h / 2
                # Save the gate
                self.gates.append((x, y, w, h, int(cx), int(cy), area))
            # Check if the approximated shape is a circle
            elif len(approx) >= 8:
                area = cv.contourArea(contour)
                perimeter = cv.arcLength(contour, True)
                roundness = 4 * np.pi * area / perimeter ** 2
                # Get the minimum enclosing circle
                if roundness >= 0.85:
                    (cx, cy), radius = cv.minEnclosingCircle(contour)
                    x, y, w, h = cv.boundingRect(approx)
                    # Save the gate
                    self.gates.append((x, y, w, h, int(cx), int(cy), area))

        # Draw the gate on the image
        if len(self.gates) > 0:
            gate = self.gates[0]
            x, y, w, h, cx, cy, area = gate
            cv.circle(image, (cx, cy), 10, (0, 355, 0), -1)
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)
            cv.putText(image, "Gate", (cx - 20, cy - 20), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv.putText(image, "Area: {:.2f}".format(area), (cx - 20, cy + 20), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv.putText(image, "Center: ({}, {})".format(cx, cy), (cx - 20, cy + 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        return image

    def generate_grid(self, image):
        # Create a copy of the image
        image = image.copy()
        # If the image is not in BGR, convert it
        if len(image.shape) != 3:
            image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

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
            x, y, w, h, cx, cy, _ = first_gate
            cv.line(image, (cx, cy), (cols // 2, rows // 2), (0, 255, 0), 5)
        if len(self.stop_signs) > 0:
            # Draw a line from the center of the image to the center of the stop sign
            stop_sign = self.stop_signs[0]
            x, y, w, h, cx, cy, _ = stop_sign
            cv.line(image, (cx, cy), (cols // 2, rows // 2), (0, 0, 255), 5)
        return image

    def stop_sign_detector(self, image):
        # print("Checking stop sign")
        image = image.copy()
        # Detect edges
        image = self.edge_detector(image)
        # Detect the stop signs
        # detection = self.detector.detectMultiScale(image, scaleFactor = 1.25, minNeighbors = 7, minSize = (80, 80), maxSize = (500, 500))
        # Calculate the contours of the image 
        contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # Sort the contours based on the area of the bounding box
        contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)[:5]
        # Reconvert the image to display the contours with color
        if len(image.shape) != 3:
            image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        # Find the stop signs
        self.stop_signs = []
        # Draw the contours based on the hierarchy of the contours
        for contour in contours:
            # Approximate the contour with a polygon
            epsilon = 0.01*cv.arcLength(contour,True)
            approx = cv.approxPolyDP(contour,epsilon,True)
            # Check if the approximated shape has 8 sides
            if 6 < len(approx) < 10 and cv.isContourConvex(approx):
                # Calculate the area of the contour
                area = cv.contourArea(contour)
                x, y, w, h = cv.boundingRect(contour)
                # Calculate the area of the bounding box
                area_box = w * h
                area_box = area_box / (image.shape[0] * image.shape[1])
                # Calculate the center of the bounding box
                cx = x + w // 2
                cy = y + h // 2
                self.stop_signs.append((x, y, w, h, cx, cy, area_box))

        # Draw the bounding boxes around the stop signs
        if len(self.stop_signs) > 0:
            # Draw a line from the center of the image to the center of the stop sign
            stop_sign = self.stop_signs[0]
            x, y, w, h, cx, cy, area = stop_sign
            cv.circle(image, (cx, cy), 10, (255, 0, 0), -1)
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv.putText(image, "Area: {:.2f}".format(area), (cx - 20, cy + 20), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv.putText(image, "Center: ({}, {})".format(cx, cy), (cx - 20, cy + 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        return image

    def image_sub_callback(self, data):
        # print("Image received")
        try:
            # Convert your ROS Image message to OpenCV2
            self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
