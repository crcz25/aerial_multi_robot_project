import rclpy
import cv2 as cv
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridgeError


class Mixin:
    def image_sub_callback(self, data):
        # print("Image received")
        try:
            self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # Show the image
            cv.imshow("Image window", self.image)
            cv.waitKey(1)
        except CvBridgeError as e:
            print(e)
