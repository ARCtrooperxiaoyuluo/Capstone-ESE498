# publish keyboard
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sys
import termios
import tty

class KeyboardInputPublisher(Node):
    def __init__(self):
        super().__init__('keyboard_input_publisher')
        self.publisher_ = self.create_publisher(String, 'keyboard_input', 10)
        self.get_logger().info("Keyboard input publisher has been started. Press 'q' to quit.")
        self.loop()

    def get_key(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key

    def loop(self):
        while rclpy.ok():
            key = self.get_key()
            if key == 'q':  # Press 'q' to quit
                break
            msg = String()
            msg.data = key
            self.publisher_.publish(msg)
            self.get_logger().info(f'Published: "{key}"')

def main(args=None):
    rclpy.init(args=args)
    node = KeyboardInputPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
