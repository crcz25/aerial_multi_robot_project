import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from cv_bridge import CvBridgeError

lower_range_green = (30, 50, 50)
upper_range_green = (90, 255, 255)

lower_range_red_1 = (0, 50, 50)
upper_range_red_1 = (10, 255, 255)

lower_range_red_2 = (170, 50, 50)
upper_range_red_2 = (180, 255, 255)

detector = cv.CascadeClassifier('haarcascade_stop.xml')

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
        image = cv.equalizeHist(image)
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
        # If the image is not in grayscale, convert it
        if len(image.shape) == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # Calculate the contours of the image 
        contours, hierarchies = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # Reconvert the image to display the contours with color
        if len(image.shape) != 3:
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        # Draw the contours based on the hierarchy of the contours
        # for i in range(len(contours)):
        #     # cv.drawContours(image, contours, i, (255, 0, 0), 3)
        #     if hierarchies[0][i][3] == -1:
        #         cv.drawContours(image, contours, i, (0, 255, 0), 3)
        #     else:
        #         cv.drawContours(image, contours, i, (0, 0, 255), 3)

        # Find the contours that are rectangular
        self.gates = []
        for contour in contours:
            # Create a bounding box around the contour
            x, y, w, h = cv.boundingRect(contour)
            # Calculate the aspect ratio of the bounding box
            aspect_ratio = float(w) / h
            # Calculate the area of the bounding box
            area = w * h
            # If the aspect ratio is between 0.75 and 1.0, then the contour is a gate
            if 0.75 < aspect_ratio < 2.0:
                # Calculate the center of the gate
                cx = x + w // 2
                cy = y + h // 2
                # Add the gate to the list of gates
                self.gates.append((x, y, w, h, cx, cy, area))
        # Sort the gates based on the area of the bounding box
        self.gates = sorted(self.gates, key=lambda x: x[6], reverse=True)
        # Draw the gates on the image
        for gate in self.gates:
            x, y, w, h, cx, cy, _ = gate
            cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv.circle(image, (cx, cy), 10, (255, 0, 0), -1)
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
            first_gate = self.gates[0]
            x, y, w, h, cx, cy, _ = first_gate
            cv.line(image, (cx, cy), (cols // 2, rows // 2), (0, 0, 255), 5)
        return image

    def center_gate(self):
        if len(self.gates) == 0:
            return
        # Get the first gate
        first_gate = self.gates[0]
        # Get the center of the gate
        cx = first_gate[4]
        cy = first_gate[5]
        # Get the center of the image
        rows, cols, _ = self.image.shape
        cx_image = cols // 2
        cy_image = rows // 2

        # Calculate the direction of the movement
        direction = np.subtract(np.array([cx, cy]), np.array([cx_image, cy_image]))
        unitary_direction = cv.normalize(direction, None, cv.NORM_L2)
        # Normalize the direction vector
        direction_unit = cv.normalize(direction, None, cv.NORM_L2)
        # Calculate the angle of the direction vector
        angle = np.arctan2(direction_unit[1], direction_unit[0])
        print(f"Angle: {angle}")
        # Since the robot is rotated 90 degrees in relation to the real world, rotate the direction vector
        rotation = np.array([[np.cos(np.pi / 2), -np.sin(np.pi / 2)],
                            [np.sin(np.pi / 2), np.cos(np.pi / 2)]])
        # Rotate the direction vector
        direction_unit = np.matmul(rotation, direction_unit)
        print(f"Direction: {direction}")
        # print(f"Rotation: {rotation}")
        print(f"Direction unit: {direction_unit[0], direction_unit[1]}")
        # Calculate the error in the x and y directions
        error_x = cx - cx_image
        error_y = cy - cy_image
        # Calculate the error in the x and y directions normalized
        error_normal_x = error_x / cx_image
        error_normal_y = error_y / cy_image
        # Calculate the unit vector in the x and y directions
        unit_x = direction_unit[0]
        unit_y = direction_unit[1]
        print(f"Error X: {error_x}")
        print(f"Error Y: {error_y}")
        print(f"Error Normal X: {error_normal_x}")
        print(f"Error Normal Y: {error_normal_y}")

        # Check the unit vector to know the direction of the movement to center the gate in the image
        if abs(error_normal_x) > 0.1:
            if unit_x > 0:
                print("Move right")
                steps = float(abs(error_normal_x) * self.speedx)
                print(f"Steps X: {steps}")
                self.move_right(steps)
            else:
                print("Move left")
                steps = float(abs(error_normal_x) * self.speedx)
                print(f"Steps X: {steps}")
                self.move_left(steps)
        elif abs(error_normal_y) > 0.1:
            if unit_y > 0:
                print("Move down")
                steps = float(abs(error_normal_y) * self.speedz)
                print(f"Steps Y: {steps}")
                self.move_down(steps)
            else:
                print("Move up")
                steps = float(abs(error_normal_y) * self.speedz)
                print(f"Steps Y: {steps}")
                self.move_up(steps)
        else:
            print("Centered in (x,y)")
            self.stop()
        return

    def stop_sign_detector(self, image):
        print("Checking stop sign")
        image = image.copy()
        rects = detector.detectMultiScale(image, scaleFactor=1.3, minNeighbors=10, minSize=(75, 75))
        for (x, y, w, h) in rects:
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Reconvert the image to display the contours with color
        return image

    def image_sub_callback(self, data):
        # print("Image received")
        try:
            # Convert your ROS Image message to OpenCV2
            self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            # Separate the stop sign from the background
            # # lower red
            # image_stop_sign_1 = self.background_foreground_separator(self.image, lower_range_red_2, upper_range_red_1)
            # # upper red
            # image_stop_sign_2 = self.background_foreground_separator(self.image, lower_range_red_2, upper_range_red_2)
            # # Combine the two images
            # image_stop_sign = cv.add(image_stop_sign_1, image_stop_sign_2)

            # Separate the gates from the background
            image_gates = self.background_foreground_separator(self.image, lower_range_green, upper_range_green)

            # Find the gates in the image
            gates = self.gate_detector(image_gates)
            # Find the stop sign in the image
            stop_sign = self.stop_sign_detector(self.image)

            # Generate the grid over the image
            image_grid = self.generate_grid(self.image)

            # Conatenate the images for display in a single image containing 4 images in 2 rows and 2 columns
            image_top = np.concatenate((gates, stop_sign), axis=1)
            image_bottom = np.concatenate((self.image, image_grid), axis=1)
            image = np.concatenate((image_top, image_bottom), axis=0)
            # Show the image
            self.show_image("Drone Image post detection", image, resize=True, width=960, height=720)

            # Check the movement of the drone to center the gate in the image
            self.center_gate()
        except CvBridgeError as e:
            print(e)
