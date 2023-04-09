from random import randint
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
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
    def __init__(self, sim=False):
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
        lower_range_green = (30, 50, 50)
        upper_range_green = (90, 255, 255)

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

    def main_node(self):
        self.get_logger().info('Main node')
        # Check if the image is not empty
        if self.image is not None:
            # Process the image
            self.track_processing()

        # self.plot()
