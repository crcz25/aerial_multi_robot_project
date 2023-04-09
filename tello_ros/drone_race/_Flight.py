from std_msgs.msg import Empty
from geometry_msgs.msg import Twist, Point, PoseStamped

import numpy as np


class Mixin:
    def take_off(self):
        self.get_logger().info('Taking off')
        self.publisher_takeoff.publish(Empty())

    def land(self):
        self.get_logger().info('Landing')
        self.publisher_land.publish(Empty())

    def send_request_simulator(self, cmd):
        self.get_logger().info('Sending request {}'.format(cmd))
        self.req.cmd = cmd
        self.future = self.cli.call_async(self.req)

    def callback_future_simulator(self, future):
        self.get_logger().info('Response received')
        try:
            response = future.result()
            self.get_logger().info('Result TUT: {}'.format(response))
            if response.rc == 1:
                self.active_ = True
        except Exception as e:
            self.get_logger().error('Service call failed %r' % (e,))

    def odom_callback(self, msg):
        # self.get_logger().info('Saving odometry')
        self.odometry = msg
        if self.sim:
            self.trajectory_odom.append([msg.pose.pose.position.x, msg.pose.pose.position.y])
            self.theta_odom = np.arctan2(msg.pose.pose.position.y - self.center.y,
                                         msg.pose.pose.position.x - self.center.x)
        else:
            self.trajectory_odom.append([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
            self.theta_odom = np.arctan2(msg.twist.twist.linear.y - self.center.y,
                                         msg.twist.twist.linear.x - self.center.x)
        # Update the current position
        self.current_position = msg.pose.pose.position
        self.current_orientation = msg.pose.pose.orientation
        # print('Current position: {:.2f}, {:.2f}, {:.2f}'.format(self.current_position.x, self.current_position.y, self.current_position.z))
        # print('Current orientation: {:.2f} {:.2f} {:.2f} {:.2f}'.format(self.current_orientation.x, self.current_orientation.y, self.current_orientation.z, self.current_orientation.w))
        # print('Current theta: {:.2f}'.format(self.theta_odom))


    def move_forward(self):
        self.get_logger().info('Moving forward')

    def move_backward(self):
        self.get_logger().info('Moving backward')

    def move_x(self, steps=0.1):
        self.get_logger().info('Moving X')
        destination_twist = Twist()
        destination_twist.linear.x = float(steps)
        destination_twist.linear.y = 0.0
        destination_twist.linear.z = 0.0
        destination_twist.angular.z = 0.0
        self.publisher_twist.publish(destination_twist)

    def move_y(self, steps=0.1):
        self.get_logger().info('Moving Y')
        destination_twist = Twist()
        destination_twist.linear.x = 0.0
        destination_twist.linear.y = float(steps)
        destination_twist.linear.z = 0.0
        destination_twist.angular.z = 0.0
        self.publisher_twist.publish(destination_twist)

    def move_z(self, steps=0.1):
        self.get_logger().info('Moving Z')
        destination_twist = Twist()
        destination_twist.linear.x = 0.0
        destination_twist.linear.y = 0.0
        destination_twist.linear.z = float(steps)
        destination_twist.angular.z = 0.0
        self.publisher_twist.publish(destination_twist)

    def rotate(self, steps=0.1):
        self.get_logger().info('Rotating')
        destination_twist = Twist()
        destination_twist.linear.x = 0.0
        destination_twist.linear.y = 0.0
        destination_twist.linear.z = 0.0
        destination_twist.angular.z = float(steps)
        self.publisher_twist.publish(destination_twist)

    def stop(self):
        self.get_logger().info('Stopping')
        destination_twist = Twist()
        destination_twist.linear.x = 0.0
        destination_twist.linear.y = 0.0
        destination_twist.linear.z = 0.0
        destination_twist.angular.z = 0.0
        self.publisher_twist.publish(destination_twist)

