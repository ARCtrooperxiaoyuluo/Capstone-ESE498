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

class XInterceptController(Node):
    def __init__(self):
        super().__init__('x_intercept_controller')
        
        # Target x-intercept for the median line
        self.target_x_intercept = 256.0
        
        # PID controller initialization
        self.x_intercept_pid = PID(0.17, 0, 0.07, setpoint=0)  # Adjust PID values as needed
        
        # Variables to hold the actual slope and intercept from the median line
        self.slope = 0.0
        self.intercept = 0.0

        # Set up ROS subscriber
        self.data_subscriber = self.create_subscription(
            Float32MultiArray, 'slope_intercept', self.data_callback, 10)

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
        # Extract slope and intercept from the message
        if len(msg.data) >= 2:
            self.slope = msg.data[0]
            self.intercept = msg.data[1]
            self.get_logger().info(f'Received Slope = {self.slope}, Intercept = {self.intercept}')

    def calculate_x_intercept(self):
        # Calculate the x-intercept of the line (x = -intercept / slope)
        if self.slope != 0:
            return (288 - self.intercept) / self.slope 
        else:
            self.get_logger().info("Slope is zero, cannot calculate x-intercept.")
            return None

    def control_loop(self):
        # Run the control loop indefinitely
        while rclpy.ok():
            # Calculate x-intercept and error only if a valid slope and intercept are available
            x_intercept = self.calculate_x_intercept()
            if x_intercept is not None:
                # Calculate the error relative to the target x-intercept
                error = x_intercept - self.target_x_intercept
                self.get_logger().info(f'Calculated x-intercept: {x_intercept}, Error: {error}')

                # Calculate the control effort using the PID controller
                control_effort = self.x_intercept_pid(error)
                
                # Calculate the servo angle, with 90 as straight and adjusted by control effort
                servo_angle = 110 + control_effort
                
                # Clamp the servo angle within physical limits (0 to 180 degrees)
                servo_angle = max(40, min(180, servo_angle))
                
                # Move the servo to the calculated angle
                self.servo.angle = servo_angle
                self.get_logger().info(f'Servo Angle: {servo_angle}, Control Effort: {control_effort}')
            
            # Sleep briefly to maintain the control loop rate
            time.sleep(0.01)

def main(args=None):
    rclpy.init(args=args)
    x_intercept_controller = XInterceptController()
    try:
        rclpy.spin(x_intercept_controller)
    except KeyboardInterrupt:
        pass
    finally:
        x_intercept_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
