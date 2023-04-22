from random import randint
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, String
from geometry_msgs.msg import Twist, Point, PoseStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

import rclpy
import _Camera as _Camera
import _Flight as _Flight
import _Utils as _Utils
import time
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

class Drone(Node, _Camera.Mixin, _Flight.Mixin, _Utils.Mixin):
    def __init__(self, sim=False, *args, **kwargs):
        super().__init__(f'drone_{randint(0, 1000)}')
        qos_policy = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                                          history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=1)
        # Create the bridge between ROS and OpenCV
        self.bridge = CvBridge()
        self.image = None
        # Internal Odometry Variables
        self.trajectory_odom = []
        self.theta_odom = 0
        self.odometry = Odometry()
        self.center = Point(x=0.0, y=0.0, z=0.0)
        self.curr_pos = Point(x=0.0, y=0.0, z=0.0)
        self.calculated_pos = Point(x=0.0, y=0.0, z=0.0)
        self.curr_angle = 0
        self.error = []
        # Set the environment (simulator or real)
        self.sim = sim
        # Execute the main node
        self.create_timer(0.1, self.main_node)

        # Track processing
        self.detector = cv2.CascadeClassifier('haarcascade_stop.xml')
        self.stop_signs = []
        self.gates = []
        self.centered = False
        self.close_enough = False
        self.moving = False

        self.model = kwargs['model']
        self.input_layer = kwargs['input_layer']
        self.output_layer = kwargs['output_layer']
        self.gate_color = kwargs['gate_color']
        
        self.colors_ranges = { 'red_1': [(0, 60, 0), (10, 255, 255)],
                              'red_2': [(160, 0, 0), (180, 255, 255)],
                               'green': [(30, 50, 50), (90, 255, 255)],
                               'blue': [(100, 100, 0), (130, 255, 255)],
                               'white': [(0, 0, 65), (0, 0, 145)],
                               'gray': [(0, 0, 0), (110, 125, 100)]
                                }

        # Set the variables according to the environment (simulator or real)
        if self.sim:
            from tello_msgs.srv import TelloAction
            # Flight Control Simulator
            self.publisher_twist = self.create_publisher(Twist, '/drone1/cmd_vel', 1)
            # To Send the takeoff and land commands
            self.cli = self.create_client(TelloAction, '/drone1/tello_action')
            while not self.cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('service not available, waiting again...')
            self.req = TelloAction.Request()
            # Camera Simulator
            self.image_sub = self.create_subscription(Image, '/drone1/image_raw', self.image_sub_callback,
                                                      qos_profile=qos_policy)
            # To get the odometry
            self.odom_sub = self.create_subscription(Odometry, '/drone1/odom', self.odom_callback,
                                                     qos_profile=qos_policy)
            # Set the speed
            self.speedx = 0.09
            self.speedy = 0.09
            self.speedz = 0.09
        else:
            # Flight Control Real World
            self.publisher_twist = self.create_publisher(Twist, '/control', 1)
            # To Send the takeoff and land commands
            self.publisher_takeoff = self.create_publisher(Empty, '/takeoff', 1)
            self.publisher_land = self.create_publisher(Empty, '/land', 1)
            # Camera Real World
            self.image_sub = self.create_subscription(Image, '/camera', self.image_sub_callback, qos_profile=qos_policy)
            # To get the odometry
            self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile=qos_policy)
            # Set the speed
            self.speedx = 20
            self.speedy = 20
            self.speedz = 20

    def track_processing(self):
        # Find the gates in the image based on the color if any
        image_gates = None

        if self.gate_color == 'red':
            image_gates = self.background_foreground_separator(self.image, self.colors_ranges['red_1'][0], self.colors_ranges['red_1'][1])
            image_gates = cv2.add(image_gates, self.background_foreground_separator(self.image, self.colors_ranges['red_2'][0], self.colors_ranges['red_2'][1]))
            # Apply the mask to the image to get the portion of the image
            image_gates = cv2.bitwise_and(self.image, self.image, mask=image_gates)
        elif self.gate_color == 'green':
            image_gates = self.background_foreground_separator(self.image, self.colors_ranges['green'][0], self.colors_ranges['green'][1])
            # Apply the mask to the image to get the portion of the image
            image_gates = cv2.bitwise_and(self.image, self.image, mask=image_gates)
        elif self.gate_color == 'blue':
            image_gates = self.background_foreground_separator(self.image, self.colors_ranges['blue'][0], self.colors_ranges['blue'][1])
            # Apply the mask to the image to get the portion of the image
            image_gates = cv2.bitwise_and(self.image, self.image, mask=image_gates)
        elif self.gate_color == 'white':
            image_gates = self.background_foreground_separator(self.image, self.colors_ranges['white'][0], self.colors_ranges['white'][1])
            # Apply the mask to the image to get the portion of the image
            image_gates = cv2.bitwise_and(self.image, self.image, mask=image_gates)
        else:
            image_gates = self.image.copy()
        gates = self.gate_detector(image_gates)

        # Find the stop sign in the image
        image_stop_sign_1 = self.background_foreground_separator(self.image, self.colors_ranges['red_1'][0], self.colors_ranges['red_1'][1])
        image_stop_sign_2 = self.background_foreground_separator(self.image, self.colors_ranges['red_2'][0], self.colors_ranges['red_2'][1])
        # image_stop_sign_3 = self.background_foreground_separator(self.image, self.colors_ranges['gray'][0], self.colors_ranges['gray'][1])
        # Combine the two images to get the complete stop sign
        image_stop_sign = cv2.add(image_stop_sign_1, image_stop_sign_2)
        # image_stop_sign = cv2.add(image_stop_sign, image_stop_sign_3)
        # Find the stop sign in the image
        stop_sign = self.stop_sign_detector(image_stop_sign)

        # Generate the grid over the image
        image_grid = self.generate_grid(self.image)

        # Conatenate the images for display in a single image containing 4 images in 2 rows and 2 columns
        image_top = np.concatenate((gates, stop_sign), axis=1)
        image_bottom = np.concatenate((self.image, image_grid), axis=1)
        image = np.concatenate((image_top, image_bottom), axis=0)
        # Show the image
        self.show_image("Drone Image post detection", image, resize=True, width=960, height=720)


    def center_object(self, cx, cy):
        # Get the center of the image
        rows, cols, _ = self.image.shape
        cx_image = cols // 2
        cy_image = rows // 2

        # Calculate the direction of the movement
        direction = np.subtract(np.array([cx, cy]), np.array([cx_image, cy_image]))
        # Normalize the direction vector
        direction_unit = cv2.normalize(direction, None, cv2.NORM_L2)
        # Calculate the angle of the direction vector
        angle = np.arctan2(direction_unit[1], direction_unit[0])
        # print(f"Angle: {angle}")
        # print(f"Direction: {direction}")
        # print(f"Direction unit: {direction_unit[0], direction_unit[1]}")
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
                steps = abs(error_normal_x) * self.speedx
                self.move_y(-steps)
            else:
                print("Move left")
                steps = abs(error_normal_x) * self.speedx
                self.move_y(steps)
        elif abs(error_normal_y) > 0.1:
            if unit_y > 0:
                print("Move down")
                steps = abs(error_normal_y) * self.speedz
                self.move_z(-steps)
            else:
                print("Move up")
                steps = abs(error_normal_y) * self.speedz
                self.move_z(steps)
        else:
            print("Centered in (x,y), approach the gate")
            self.centered = True
            self.stop()
        return

    def approach_object(self, area, threshold=0.15):
        if area > threshold:
            self.close_enough = True
            self.stop()
        else:
            self.move_x(self.speedx * 0.25)
        return

    def stop_drone(self, area):
        if area < 0.2:
            self.moving = True
            self.move_x(self.speedx * 0.5)
        else:
            self.moving = False
            self.centered = False
            self.stop()
            if self.sim:
                self.send_request_simulator('land')
            else:
                self.land()
            exit()
        return

    def pass_gate(self):
        self.moving = True
        self.move_x(self.speedx * 0.45)
        # self.create_timer(5.0, self.stop)
        rclpy.spin_once(self, timeout_sec=5.0)
        self.stop()
        return

    def find_depth(self, image):
        print("Finding depth")
        image = image.copy()
        N, C, H, W = self.input_layer.shape
        resized_image = cv2.resize(src=image, dsize=(W, H))
        input_data = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0).astype(np.float32)
        result = self.model({self.input_layer.any_name: input_data})[self.output_layer]
        depth_map = cv2.normalize(result[0], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]))
        depth_map = cv2.applyColorMap(np.uint8(255 * depth_map), cv2.COLORMAP_JET)
        self.show_image("depth", depth_map, resize=True, width=960, height=720)
        return depth_map

    def main_node(self):
        self.get_logger().info('Main node')
        # Check if the image is not empty
        if self.image is not None:
            # # Show the image
            # self.show_image("Drone Image", self.image, resize=True, width=960, height=720)
            # depth = self.find_depth(self.image)
            # Process the image
            self.track_processing()
            # If the area of the gate is bigger it means that the drone is close to the gate than the stop sign. Call the function approach_gate to move forward to the gate
            if len(self.gates) > 0:
                print("Gate Detected")
                cx, cy, radius, area =  self.gates[0]
                print(f"Area: {area}")
                if not self.centered:
                    print("Centering the gate")
                    #self.center_object(cx, cy)
                elif not self.close_enough:
                    print("Approaching the gate")
                    #self.approach_object(area, 0.45)
                elif not self.moving:
                    print("Passing the gate")
                    #self.pass_gate()
                else:
                    print("RESETING")
                    self.centered = False
                    self.close_enough = False
                    self.moving = False
                    # self.stop()
            elif len(self.stop_signs) > 0:
                print("Stop sign detected")
                x, y, w, h, cx, cy, area = self.stop_signs[0]
                print(f"Area: {area}")
                if not self.centered:
                    print("Centering the stop sign")
                    #self.center_object(cx, cy)
                elif not self.close_enough:
                    print("Approaching the stop sign")
                    #self.approach_object(area, 0.15)
                elif not self.moving:
                    print("Stopping the drone")
                    #self.stop_drone(area)
                else:
                    print("RESETING")
                    self.centered = False
                    self.close_enough = False
                    self.moving = False
                    #self.stop()
            else:
                print("Nothing detected")
                self.centered = False
                self.close_enough = False
                self.moving = False
                self.stop()
        return
