import time
import board
import adafruit_mpu6050
import math
import numpy as np

# Initialize MPU6050 using I2C
i2c = board.I2C()  # uses board.SCL and board.SDA
mpu = adafruit_mpu6050.MPU6050(i2c)

# Set gyroscope offsets (example values, replace with your calibration values)
gyro_offsets = [-0.0049, 0.0186, -0.0045]  # Example offsets for gyro X, Y, Z

# Set accelerometer calibration coefficients (example values, replace with your calibration values)
accel_coeffs = [
    [0.10146388, -0.01743124],  # X-axis calibration [m_x, b]
    [0.10132916, -0.01252953],  # Y-axis calibration [m_y, b]
    [-0.09826508, -0.49321221]   # Z-axis calibration [m_z, b]
]

# Kalman filter parameters
dt = 0.1  # Time step for gyro integration (s)
Q = np.array([[0.001, 0], [0, 0.001]])  # Process noise covariance
R = np.array([[0.05, 0], [0, 0.05]])  # Measurement noise covariance
P = np.array([[1, 0], [0, 1]])  # Initial estimation error covariance
state = np.array([0.0, 0.0])  # Initial state [pitch, roll]

# First-order filter parameters
alpha = 0.5  # Smoothing factor for the first-order filter
filtered_state = np.array([0, 0])  # Initialize filtered pitch and roll

# Function to apply accelerometer calibration
def accel_fit(x_input, m_x, b):
    return (m_x * x_input) + b

# Function to calculate pitch and roll from accelerometer data
def calculate_accel_angles(accel_x, accel_y, accel_z):
    roll = math.atan2(accel_y, math.sqrt(accel_x ** 2 + accel_z ** 2)) * (180 / math.pi)
    pitch = math.atan2(-accel_x, accel_z) * (180 / math.pi)
    return pitch, roll

while True:
    # Raw sensor readings
    accel_x, accel_y, accel_z = mpu.acceleration
    gyro_x, gyro_y, gyro_z = mpu.gyro

    # Apply calibration offsets to gyro readings
    calibrated_gyro_x = (gyro_x - gyro_offsets[0])*(180/3.14)
    calibrated_gyro_y = (gyro_y - gyro_offsets[1])*(180/3.14)
    calibrated_gyro_z = (gyro_z - gyro_offsets[2])*(180/3.14)

    # Apply calibration coefficients to accelerometer readings
    calibrated_accel_x = accel_fit(accel_x, *accel_coeffs[0])
    calibrated_accel_y = accel_fit(accel_y, *accel_coeffs[1])
    calibrated_accel_z = accel_fit(accel_z, *accel_coeffs[2])

    # Calculate pitch and roll from accelerometer
    accel_pitch, accel_roll = calculate_accel_angles(calibrated_accel_x, calibrated_accel_y, calibrated_accel_z)

    # Prediction step
    gyro_pitch = state[0] + calibrated_gyro_y * dt  # pitch prediction
    gyro_roll = state[1] + calibrated_gyro_x * dt   # roll prediction
    state_pred = np.array([gyro_pitch, gyro_roll])

    # Update the error covariance matrix
    P = P + Q

    # Measurement update step
    # Measurement vector (from accelerometer)
    z = np.array([accel_pitch, accel_roll])

    # Calculate the Kalman gain
    S = P + R  # Innovation covariance
    K = P @ np.linalg.inv(S)  # Kalman gain

    # Update the estimate with the measurement
    state = state_pred + K @ (z - state_pred)

    # Update the error covariance matrix
    P = (np.eye(2) - K) @ P

    # Extract pitch and roll from the state
    pitch, roll = state

    # First-order filter application
    filtered_state[0] = alpha * pitch + (1 - alpha) * filtered_state[0]  # Filtered pitch
    filtered_state[1] = alpha * roll + (1 - alpha) * filtered_state[1]    # Filtered roll

    # Print calibrated sensor data and filtered pitch and roll
    print(f"Filtered Pitch: {filtered_state[0]:.2f}, Filtered Roll: {filtered_state[1]:.2f}")
    #print(f"Calibrated Acceleration: X:{calibrated_accel_x:.2f}, Y: {calibrated_accel_y:.2f}, Z: {calibrated_accel_z:.2f} m/s^2")
    #print(f"Calibrated Gyro: X:{calibrated_gyro_x:.2f}, Y: {calibrated_gyro_y:.2f}, Z: {calibrated_gyro_z:.2f} rad/s")
    print("")

    time.sleep(dt)

