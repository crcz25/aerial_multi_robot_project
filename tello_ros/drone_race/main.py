import rclpy
from drone import Drone


def main():
    rclpy.init()
    sim = True
    minimal_client = Drone(sim=sim)

    try:
        minimal_client.get_logger().info("Starting node")
        if sim:
            minimal_client.send_request_simulator('takeoff')
        else:
            minimal_client.take_off()
        while rclpy.ok():
            rclpy.spin_once(minimal_client)
    except KeyboardInterrupt:
        # Press Ctrl+C to stop the program
        pass
    finally:
        if sim:
            minimal_client.send_request_simulator('land')
        else:
            minimal_client.land()
        minimal_client.get_logger().info('Shutting down')
        minimal_client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
