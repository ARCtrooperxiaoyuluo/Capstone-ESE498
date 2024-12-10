import time
import math
import threading
import busio
import board
from adafruit_motor import servo
from adafruit_pca9685 import PCA9685
from simple_pid import PID
from std_msgs.msg import Float32, Float32MultiArray
import rclpy
from rclpy.node import Node

class WallFollowingCar(Node):
    def __init__(self):
        super().__init__('wall_following_car')

        self.target_distance = 600.0  # Target distance from the wall in mm
        self.distance_pid = PID(1.05, 0.02, 0.55, setpoint=0)

        self.actual_distance = 0.0
        self.actual_heading_angle = 0.0
        self.magnetic_heading = 0.0
        self.following_left_wall = False  # Track which wall is being followed

        # Start with right side LiDAR data
        self.data_subscriber = self.create_subscription(
            Float32MultiArray, 'lidar_250_290_data', self.data_callback, 10)

        # Subscribe to magnetic heading data
        self.heading_subscriber = self.create_subscription(
            Float32, 'compass_heading', self.magnetic_heading_callback, 10)

        # Servo setup
        i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(i2c)
        self.pca.frequency = 100
        self.servo = servo.Servo(self.pca.channels[14], min_pulse=500, max_pulse=2400)

        # Control loop thread
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()

    def data_callback(self, msg):
        self.actual_distance = msg.data[1]
        self.actual_heading_angle = msg.data[0]

    def magnetic_heading_callback(self, msg):
        self.magnetic_heading = msg.data

        # Switch from right-side LiDAR to left-side LiDAR when heading > 170Â°
        if self.magnetic_heading >= 290.0 or self.magnetic_heading <= 10.0:
            self.data_subscriber = self.create_subscription(
                Float32MultiArray, 'lidar_70_110_data', self.data_callback, 10)
            self.following_left_wall = True  # Now following the left wall
            print('left')
    def map_distance_to_heading(self, distance_error):
        """Map distance error to heading adjustment based on the wall being followed."""
        # Invert the sign of distance_error when following the left wall
        if self.following_left_wall:
            distance_error = -distance_error

        if -600 <= distance_error <= 600:
            return -0.065 * distance_error
        if distance_error > 600:
            return -39+5*math.log10(distance_error-599)
        elif distance_error < -600:
            return 39-5 * math.log10(abs(distance_error)-599)

    def control_loop(self):
        while rclpy.ok():
            distance_error = self.target_distance - self.actual_distance
            distance_error = int(distance_error)
            target_heading_angle = self.map_distance_to_heading(distance_error)
            control_effort = self.distance_pid(target_heading_angle - self.actual_heading_angle)

            servo_angle = 110 + control_effort
            print(servo_angle)
            servo_angle = max(40, min(180, servo_angle))

            self.servo.angle = servo_angle

            time.sleep(0.005)

def main(args=None):
    rclpy.init(args=args)
    node = WallFollowingCar()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
