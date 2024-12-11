import math
import time
import busio
from board import SCL, SDA
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# Function to initialize the servo motor
def Servo_Motor_Initialization():
    i2c_bus = busio.I2C(SCL, SDA)
    pca = PCA9685(i2c_bus)
    pca.frequency = 100
    return pca

# Function to control motor speed (forward/reverse)
def Motor_Speed(pca, percent):
    # Converts a -1 to 1 value to 16-bit duty cycle
    speed = ((percent) * 3277) + 65535 * 0.15
    pca.channels[15].duty_cycle = math.floor(speed)
    print(f"Motor speed set to {percent * 100:.2f}%")

# Function to update the servo steering angle
def update_servo(servo, new_angle):
    angle = max(0, min(180, new_angle))  # Clamp the angle between 0 and 180
    servo.angle = angle
    print(f"Servo angle set to {angle}")

class MotorControllerNode(Node):
    def __init__(self):
        super().__init__('motor_controller_node')
        
        # Initialize PCA9685 and servo
        self.pca = Servo_Motor_Initialization()
        channel_num = 14
        self.servo7 = servo.Servo(self.pca.channels[channel_num])
        
        # Initial angle and speed
        self.angle = 110
        self.speed_percent = 0
        self.servo7.angle = self.angle
        Motor_Speed(self.pca, self.speed_percent)
        
        # Subscriber to listen for keyboard input
        self.subscription = self.create_subscription(String, 'keyboard_input', self.keyboard_input_callback, 10)
        self.get_logger().info("Motor controller started. Listening for keyboard input...")
    
    # Callback function to handle keyboard inputs
    def keyboard_input_callback(self, msg):
        key = msg.data
        
        if key == 'd':  # Turn right
            self.angle = max(0, self.angle - 5)  # Decrease the angle
            update_servo(self.servo7, self.angle)
        elif key == 'a':  # Turn left
            self.angle = min(180, self.angle + 5)  # Increase the angle
            update_servo(self.servo7, self.angle)
        elif key == 'w':  # Accelerate
            self.speed_percent = min(self.speed_percent + 0.01, 1)  # Increase speed
            Motor_Speed(self.pca, self.speed_percent)
        elif key == 's':  # Decelerate
            self.speed_percent = max(self.speed_percent - 0.01, -1)  # Decrease speed (reverse)
            Motor_Speed(self.pca, self.speed_percent)
        elif key == 'q':  # Quit the program
            self.get_logger().info("Returning motor speed to 0% and shutting down...")
            self.speed_percent = 0
            Motor_Speed(self.pca, self.speed_percent)  # Ensure speed is set to 0%
            rclpy.shutdown()

    def destroy(self):
        self.pca.deinit()  # Clean up the PCA9685 when done
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    motor_controller_node = MotorControllerNode()

    try:
        rclpy.spin(motor_controller_node)
    except KeyboardInterrupt:
        motor_controller_node.destroy()

if __name__ == '__main__':
    main()
