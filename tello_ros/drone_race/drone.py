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
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

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
        self.detect = True
        self.centered = False
        self.close_enough = False
        self.moving = False
        self.prev_gate = None
        self.curr_gate = None
        self.gate_found = False
        self.prev_stop = None
        self.curr_stop = None
        self.stop_sign_found = False
        self.searching = True
        self.goal_yaw = None
        self.direction = 0
        self.model = kwargs['model']
        self.input_layer = kwargs['input_layer']
        self.output_layer = kwargs['output_layer']
        self.gate_color = kwargs['gate_color']
        
        self.colors_ranges = {
            # 'white_background' : [(0, 0, 0), (180, 255, 160)],
            # 'gray': [(0, 0, 57), (180, 255, 255)],
            'red': [(0, 200, 0), (10, 255, 255)],
            'green': [(30, 50, 50), (90, 255, 255)],
            'blue': [(90, 100, 0), (140, 255, 255)],
            'white': [(0, 0, 65), (0, 0, 145)],
            'stop_sign': [(0, 50, 0), (180, 255, 50)],
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
        # Create an empty image to store the gates the same size and type as the original image
        image_gates = np.zeros(self.image.shape, dtype=np.uint8)

        # Find the gates in the image based on the color that is passed or all the colors
        if self.gate_color == 'all':
            # Iterate over all the colors except the gray
            for color in self.colors_ranges.keys():
                if color != 'stop_sign':
                    image_gates += self.background_foreground_separator(self.image, self.colors_ranges[color][0], self.colors_ranges[color][1])
        else:
            image_gates = self.background_foreground_separator(self.image, self.colors_ranges[self.gate_color][0], self.colors_ranges[self.gate_color][1])
        # Convert the image to grayscale
        image_gates = cv2.cvtColor(image_gates, cv2.COLOR_BGR2GRAY)
        # Apply threshold to the image
        image_gates = cv2.adaptiveThreshold(image_gates, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
        # Find the gates in the image
        gates = self.gate_detector(image_gates)

        # Find the stop sign in the image
        # image_stop_sign_1 = self.background_foreground_separator(self.image, self.colors_ranges['red_1'][0], self.colors_ranges['red_1'][1])
        image_stop_sign = self.background_foreground_separator(self.image, self.colors_ranges['stop_sign'][0], self.colors_ranges['stop_sign'][1])
        # image_stop_sign_3 = self.background_foreground_separator(self.image, self.colors_ranges['gray'][0], self.colors_ranges['gray'][1])
        # Combine the two images to get the complete stop sign
        # image_stop_sign = cv2.add(image_stop_sign_1, image_stop_sign_2)
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
        # print(f"Error X: {error_x}")
        # print(f"Error Y: {error_y}")
        # print(f"Error Normal X: {error_normal_x}")
        # print(f"Error Normal Y: {error_normal_y}")

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
            print("Centered in (x,y)")
            self.centered = True
            self.stop()
        return

    def approach_object(self, area, threshold=0.1):
        print(f"Area: {area}")
        # Calculate the error between the area of the gate and the threshold
        error = np.round(np.abs(0.40 - area), 2)
        print(f"Error: {error}")
        # Calculate the steps to move
        steps = error * self.speedx
        # Check if the drone is close enough to the gate
        if error > threshold:
            self.move_x(steps)
            self.close_enough = False
            self.moving = True
        else:
            self.moving = False
            self.close_enough = True
            self.goal_position = [self.current_position[0] + 1.7, self.current_position[1]]
            self.stop()
        return

    def stop_drone(self, area, threshold=0.1):
        # Calculate the error between the area of the gate and the threshold
        error = np.abs(0.45 - area) * 10
        print(f"Error: {error}")
        # Calculate the steps to move
        steps = error * self.speedx * 0.2
        # Check if the drone is close enough to the gate
        if error > threshold:
            self.move_x(steps)
            self.moving = True
            self.close_enough = False
            self.centered = False
        else:
            self.close_enough = True
            self.moving = False
            self.stop()
        # if self.sim:
        #     self.send_request_simulator('land')
        # else:
        #     self.land()
        # exit()
        return

    def pass_gate(self, threshold=0.1):
        # Set the moving flag
        # print(f"Goal position: {self.goal_position}")
        # Calculate the distance to the goal position
        distance = np.linalg.norm(np.subtract(self.goal_position, self.current_position))
        # print(f"Distance: {distance}")
        # Calculate the linear velocity
        # increase or reduce the linear velocity depending on the distance
        linear_velocity = self.speedx * distance if distance > threshold else self.speedx
        # print(f"Linear velocity: {linear_velocity}")

        # Move the drone
        if distance > threshold:
            self.move_x(linear_velocity)
            self.moving = True
        else:
            self.centered = False
            self.close_enough = False
            self.gate_found = False
            self.curr_gate = None
            self.moving = False
            self.searching = True
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
            if self.gate_found:
                print("Gate Detected")
                # self.direction = 0
                # self.goal_yaw = None
                _, _, _, _, cx, cy, area_gate =  self.curr_gate
                if not self.centered:
                    print("Centering the gate")
                    self.center_object(cx, cy)
                elif not self.close_enough:
                    print("Approaching the gate")
                    self.approach_object(area_gate)
                else:
                    print("Stopping the drone, it is close enough and centered, time to pass the gate")
                    self.pass_gate()
            elif self.stop_sign_found:
                print("Stop sign detected")
                # self.direction = 0
                # self.goal_yaw = None
                _, _, _, _, cx, cy, area_stop = self.curr_stop
                if not self.centered:
                    print("Centering the stop sign")
                    self.center_object(cx, cy)
                elif not self.close_enough:
                    print("Approaching the stop sign")
                    self.stop_drone(area_stop)
                    #self.approach_object(area_stop)
                else:
                    print("Stopping the drone, it is close enough and centered")
                    self.stop_drone(area_stop)
            else:
                print("Searching for a gate or stop sign")
                # Get the current yaw
                yaw = self.theta
                # Check if the goal yaw is None
                if self.goal_yaw is None:

                    # Calculate the goal yaw based on the direction to rotate +30 or -30 degrees
                    if self.direction == 0:
                        self.goal_yaw = yaw + np.deg2rad(30)
                    elif self.direction == 1:
                        self.goal_yaw = yaw - np.deg2rad(30)
                    else:
                        self.goal_yaw = yaw

                    # Shift the goal yaw to be between -pi and pi
                    angle = self.goal_yaw
                    angle_mod = angle % (2 * np.pi)
                    angle_shifted = (angle_mod - np.pi) % (2 * np.pi)
                    if angle_shifted > np.pi:
                        angle_shifted -= 2*np.pi
                    self.goal_yaw = angle_shifted

                # Calculate the error between the current yaw and the goal yaw
                error = np.abs(self.goal_yaw - yaw)
                # Calculate the steps to move
                steps = error * self.speedy

                print(f"DIRECTION: {self.direction}")
                print(f"Current yaw: {np.rad2deg(yaw)}, {yaw}")
                print(f"Goal yaw: {np.rad2deg(self.goal_yaw)}, {self.goal_yaw}")
                print(f"Error: {error}")

                # Check if the drone is close enough to the goal yaw
                if error > 0.1:
                    print(f"Rotating {steps} steps")
                    steps = -steps if self.direction == 0 else steps
                    self.rotate(steps)
                    self.moving = True
                else:
                    print("Stopping the drone")
                    self.moving = False
                    self.stop()
                    self.goal_yaw = None
                    self.direction = 1 if self.direction == 0 else 0
        return
