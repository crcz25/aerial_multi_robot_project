from std_msgs.msg import Empty
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
