# PID conrtoller, 2 threads
# need test 
import time
import threading
import busio
from board import SCL, SDA
from adafruit_motor import servo
from adafruit_pca9685 import PCA9685
from simple_pid import PID
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import math

class WallFollowingCar(Node):
    def __init__(self):
        super().__init__('wall_following_car')
        
        # PID controller initialization
        self.target_distance = 1000.0  # Target distance from the wall in millimeters
        self.distance_pid = PID(1.5, 0, 0, setpoint=0)  # PID setpoint is the heading angle difference (degrees)
        
        # Variables to hold actual values from ROS topic
        self.actual_distance = 0.0
        self.actual_heading_angle = 0.0
        
        # Set up ROS subscriber
        self.data_subscriber = self.create_subscription(
            Float32MultiArray, '/lidar_data', self.data_callback, 10)
        
        # Set up servo
        i2c = busio.I2C(SCL, SDA)
        self.pca = PCA9685(i2c)
        self.pca.frequency = 100
        self.servo = servo.Servo(self.pca.channels[14], min_pulse=500, max_pulse=2400)
        
        # Create and start the control loop thread
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.daemon = True  # Ensures the thread exits when the main program ends
        self.control_thread.start()

    def data_callback(self, msg):
        # Extract distance and heading angle from the array
        self.actual_distance = msg.data[1]  # Distance in mm
        self.actual_heading_angle = msg.data[0]  # Heading angle in degrees

    def map_distance_to_heading(self, distance_error):
        # Linear interpolation within the range -200 mm to 200 mm
        distance_error = -distance_error #if left
        if -800 <= distance_error <= 800:
            return -0.085 * distance_error  # Note the negative sign to flip the direction
       
        # Nonlinear response outside the boundary
        if distance_error > 200:
            return -20 * math.log10(1 + (distance_error - 200) / 200)
        elif distance_error < -200:
            return 20 * math.log10(1 + (200 - distance_error) / 200)

    def control_loop(self):
        # Run the control loop indefinitely
        while rclpy.ok():
            # Calculate the distance error in millimeters
            distance_error = self.target_distance - self.actual_distance
            
            # Map the distance error to a target heading angle using the corrected nonlinear function
            target_heading_angle = self.map_distance_to_heading(distance_error)
            
            # Calculate the control effort using PID based on heading angle difference
            control_effort = self.distance_pid(target_heading_angle - self.actual_heading_angle)
            
            # Calculate the servo angle, assuming 90 is straight, and adjust based on control effort
            servo_angle = 100 + control_effort
            
            # Clamp the servo angle within its physical limits (0 to 180 degrees)
            servo_angle = max(40, min(160, servo_angle))
            
            # Move the servo to the new angle
            self.servo.angle = servo_angle
            self.get_logger().info(f'Servo Angle: {servo_angle}, Distance Error: {distance_error}, Target Heading: {target_heading_angle}')
            
            # Sleep to maintain control loop rate
            time.sleep(0.01)

def main(args=None):
    rclpy.init(args=args)
    wall_following_car = WallFollowingCar()
    try:
        rclpy.spin(wall_following_car)
    except KeyboardInterrupt:
        pass
    finally:
        wall_following_car.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

