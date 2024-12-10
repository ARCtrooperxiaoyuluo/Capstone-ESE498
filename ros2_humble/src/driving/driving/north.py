# 9DOF fusion  north
# publish node 
import time
import board
import adafruit_mpu6050
import math
import numpy as np
import adafruit_mmc56x3
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

# Sensor initialization
i2c = board.I2C()
mpu = adafruit_mpu6050.MPU6050(i2c)
magn = adafruit_mmc56x3.MMC5603(i2c, address=0x30)

# Calibration parameters
gyro_offsets = [-0.0049, 0.0186, -0.0045]
accel_coeffs = [
    [0.10146388, -0.01743124],
    [0.10132916, -0.01252953],
    [-0.09826508, -0.49321221]
]

# Kalman filter parameters
dt = 0.05
Q = np.array([[0.001, 0], [0, 0.001]])
R = np.array([[0.05, 0], [0, 0.05]])
P = np.array([[1.0, 0.0], [0.0, 1.0]])
state = np.array([0.0, 0.0])

# Filter parameters
alpha = 0.5  
filtered_pitch = 0.0
filtered_roll = 0.0
alpha_m = 0.2
filtered_x, filtered_y, filtered_z = 0.0, 0.0, 0.0

# Helper functions
def accel_fit(x_input, m_x, b):
    return (m_x * x_input) + b

def calculate_accel_angles(accel_x, accel_y, accel_z):
    roll = math.atan2(accel_y, math.sqrt(accel_x ** 2 + accel_z ** 2)) * (180 / math.pi)
    pitch = math.atan2(-accel_x, accel_z) * (180 / math.pi)
    return pitch, roll

def calculate_tilt_compensated_heading(mag_x, mag_y, mag_z, pitch, roll):
    mag_z = -mag_z
    mag_x = -mag_x
    pitch_rad = math.radians(pitch)
    roll_rad = math.radians(roll)

    mag_x_tilt_comp = mag_x * math.cos(pitch_rad) + mag_z * math.sin(pitch_rad) * math.sin(roll_rad)
    mag_y_tilt_comp = mag_y * math.cos(roll_rad) - mag_z * math.sin(pitch_rad) * math.cos(roll_rad)
    
    heading_rad = math.atan2(mag_y_tilt_comp, mag_x_tilt_comp)
    heading_deg = math.degrees(heading_rad) +80  # Adjust for magnetic declination
    heading_deg = heading_deg % 360
    return heading_deg 

# Node definition
class CompassPublisher(Node):
    def __init__(self):
        super().__init__('compass_publisher')
        self.publisher_ = self.create_publisher(Float32, 'compass_heading', 10)
        self.heading = 0.0

        # Create separate threads for computation and publishing
        self.compute_thread = threading.Thread(target=self.compute_heading)
        self.compute_thread.daemon = True
        self.compute_thread.start()

        self.publish_thread = threading.Thread(target=self.publish_heading)
        self.publish_thread.daemon = True
        self.publish_thread.start()

    def compute_heading(self):
        global filtered_pitch, filtered_roll, state, P, filtered_x, filtered_y, filtered_z
        while rclpy.ok():
            # Sensor readings
            accel_x, accel_y, accel_z = mpu.acceleration
            gyro_x, gyro_y, gyro_z = mpu.gyro
            mag_x, mag_y, mag_z = magn.magnetic

            # Gyro and accelerometer calibration
            calibrated_gyro_x = gyro_x - gyro_offsets[0]
            calibrated_gyro_y = gyro_y - gyro_offsets[1]
            calibrated_gyro_z = gyro_z - gyro_offsets[2]
            calibrated_accel_x = accel_fit(accel_x, *accel_coeffs[0])
            calibrated_accel_y = accel_fit(accel_y, *accel_coeffs[1])
            calibrated_accel_z = accel_fit(accel_z, *accel_coeffs[2])

            # Calculate accelerometer-based angles
            accel_pitch, accel_roll = calculate_accel_angles(calibrated_accel_x, calibrated_accel_y, calibrated_accel_z)
            
            # Predict state with gyro data
            gyro_pitch = state[0] + calibrated_gyro_y * dt
            gyro_roll = state[1] + calibrated_gyro_x * dt
            state_pred = np.array([gyro_pitch, gyro_roll])
            P += Q
            
            # Kalman filter update
            z = np.array([accel_pitch, accel_roll])
            S = P + R
            K = P @ np.linalg.inv(S)
            state = state_pred + K @ (z - state_pred)
            P = (np.eye(2) - K) @ P

            # Apply low-pass filter
            pitch, roll = state
            filtered_pitch = alpha * pitch + (1 - alpha) * filtered_pitch
            filtered_roll = alpha * roll + (1 - alpha) * filtered_roll

            # Magnetometer adjustments and filtering
            adjust_x = mag_x + 59.703125
            adjust_y = mag_y - 24.584375
            adjust_z = mag_z + 76.234375
            filtered_x = alpha_m * adjust_x + (1 - alpha_m) * filtered_x
            filtered_y = alpha_m * adjust_y + (1 - alpha_m) * filtered_y
            filtered_z = alpha_m * adjust_z + (1 - alpha_m) * filtered_z

            # Update heading
            self.heading = calculate_tilt_compensated_heading(filtered_x, filtered_y, filtered_z, filtered_pitch, filtered_roll)
            time.sleep(dt)

    def publish_heading(self):
        msg = Float32()
        while rclpy.ok():
            msg.data = self.heading
            self.publisher_.publish(msg)
            self.get_logger().info(f'published heading :{self.heading:.2f}')
            time.sleep(0.1)  # Publish at 10 Hz

def main(args=None):
    rclpy.init(args=args)
    compass_publisher = CompassPublisher()
    rclpy.spin(compass_publisher)
    compass_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


