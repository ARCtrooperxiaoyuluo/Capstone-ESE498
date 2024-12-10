# subscribe and print
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

# ROS2 Node and Subscriber Setup
class SpeedSubscriber(Node):
    def __init__(self):
        super().__init__('speed_subscriber')
        self.subscription = self.create_subscription(
            Float64,
            'wheel_speed',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        speed_kmh = msg.data  # Get the speed from the message
        self.get_logger().info(f'Speed: {speed_kmh:.4f} km/h')  # Print the speed

def main(args=None):
    rclpy.init(args=args)
    speed_subscriber = SpeedSubscriber()
    
    try:
        rclpy.spin(speed_subscriber)  # Keep the node running to listen to messages
    except KeyboardInterrupt:
        pass
    finally:
        speed_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
