import os
import math
import time
import yaml
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685

# Function to load PID controller parameters from a YAML file
from ament_index_python.packages import get_package_share_directory

def load_config():
    package_share_directory = get_package_share_directory('driving')
    config_file_path = os.path.join(package_share_directory, 'config', 'config.yaml')
    
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Accessing the parameters under the pid_controller key
    kp = config['pid_controller']['kp']
    ki = config['pid_controller']['ki']
    kd = config['pid_controller']['kd']
    setpoint_speed = config['pid_controller']['setpoint_speed']
    
    return kp, ki, kd, setpoint_speed

# Load the configuration
kp, ki, kd, setpoint_speed = load_config()
print(kp, ki, kd)

actual_speed = 0.0  # Initialize actual speed
integral = 0.0  # Initialize integral term
previous_error = 0.0  # For D term
pca = None
start_time = None
initial_acceleration = False  # Flag for initial acceleration phase

# PID Controller function
def compute_pid_control(actual_speed, setpoint, kp, ki, kd, integral, previous_error, dt=0.005):
    error = setpoint - actual_speed
    p_term = kp * error
    integral += error * dt
    i_term = ki * integral
    d_term = kd * (error - previous_error) / dt
    control_signal = p_term + i_term + d_term
    return control_signal, integral, p_term, i_term, d_term, error

# Function to initialize the Servo motor
def Servo_Motor_Initialization():
    i2c_bus = busio.I2C(SCL, SDA)
    pca = PCA9685(i2c_bus)
    pca.frequency = 100
    return pca

# Function to set motor speed using PWM
def Motor_Speed(pca, percent):
    # Converts a -1 to 1 value to 16-bit duty cycle
    speed = ((percent) * 3277) + 65535 * 0.15
    pca.channels[15].duty_cycle = math.floor(speed)

# ROS2 Node class
class MotorControllerNode(Node):
    def __init__(self):
        super().__init__('motor_controller')
        global pca, start_time
        pca = Servo_Motor_Initialization()
        start_time = time.time()
        self.subscription = self.create_subscription(
            Float64,
            'wheel_speed',
            self.speed_callback,
            10
        )
        # Create a timer to ensure periodic updates
        self.timer = self.create_timer(0.005, self.timer_callback)
    
    def speed_callback(self, msg):
        global actual_speed
        actual_speed = msg.data
        #print(f'Actual speed: {actual_speed:.4f} km/h')

    def timer_callback(self):
        global integral, previous_error, start_time, initial_acceleration
        
        # Check if we are still in the initial acceleration phase
        if initial_acceleration:
            # If actual speed is less than target speed - 10 km/h, keep accelerating
            if actual_speed < setpoint_speed - 10.0:
                Motor_Speed(pca, 0.17)  # Accelerate at 50% power
                print("Accelerating...")
            else:
                # Switch to PID control once close to the target speed
                initial_acceleration = False
                print("Switching to PID control.")
        
        # Use PID control if not in the acceleration phase
        if not initial_acceleration:
            control_signal, integral, p_term, i_term, d_term, error = compute_pid_control(
                actual_speed, setpoint_speed, kp, ki, kd, integral, previous_error)
            control_signal = max(min(control_signal, 0.22), 0)  # Clamp the control signal between 0 and 0.35
            Motor_Speed(pca, control_signal)
            
            # Calculate the elapsed time
            elapsed_time = time.time() - start_time

            # Print the actual speed, P term, I term, D term, and elapsed time
            print(f"Time: {elapsed_time:.4f}, Speed: {actual_speed:.4f}, P: {p_term:.4f}, I: {i_term:.4f}, D: {d_term:.4f}")

            # Update previous error
            previous_error = error

        # Stop the motor after 20 seconds
        elapsed_time = time.time() - start_time
        if elapsed_time >= 20:
            print("20 seconds have passed. Stopping the motor.")
            Motor_Speed(pca, 0)  # Set motor speed to 0
            self.destroy_node()  # Shut down the node

def main(args=None):
    rclpy.init(args=args)
    motor_controller_node = MotorControllerNode()
    try:
        rclpy.spin(motor_controller_node)
    except KeyboardInterrupt:
        pass
    motor_controller_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

