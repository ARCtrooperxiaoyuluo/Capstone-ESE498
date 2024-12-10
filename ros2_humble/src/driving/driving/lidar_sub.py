# subscribe and print  yaw and dist
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

class LidarSubscriber(Node):
    def __init__(self):
        super().__init__('lidar_subscriber')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'lidar_data',
            self.listener_callback,
            10
        )
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        # Extract yaw and distance from the message data
        if len(msg.data) >= 2:
            yaw = msg.data[0]
            dist = msg.data[1]
            self.get_logger().info(f'Received yaw: {yaw:.2f}, distance: {dist:.2f}')

def main(args=None):
    rclpy.init(args=args)
    lidar_subscriber = LidarSubscriber()

    try:
        rclpy.spin(lidar_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        lidar_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

