# structure of 9DOF
import time
import board
import adafruit_mpu6050
import math
import numpy as np
import adafruit_mmc56x3

i2c = board.I2C()  
mpu = adafruit_mpu6050.MPU6050(i2c)
#i2c = busio.I2C(board.SCL, board.SDA)
magn = adafruit_mmc56x3.MMC5603(i2c, address=0x30)

gyro_offsets = [-0.0049, 0.0186, -0.0045]  # Example offsets for gyro X, Y, Z
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
state = np.array([0, 0])  # Initial state [pitch, roll]

# First-order low-pass filter parameters
alpha = 0.5  # Filter constant (0 < alpha < 1)
filtered_pitch = 0  # Initial filtered pitch
filtered_roll = 0  # Initial filtered roll


# magnetometer parameters
MAGNETIC_DECLINATION = -6.0  # example for a location with -5Â° declination
alpha_m = 0.2
filtered_x, filtered_y, filtered_z = 0.0, 0.0, 0.0


def calculate_heading(mag_x, mag_y, mag_z, pitch, roll):
    mag_z = -mag_z
    mag_x = -mag_x
    pitch = int(pitch)
    roll = int(roll)
    pitch_rad = math.radians(pitch)
    roll_rad = math.radians(roll)
    # Adjust magnetometer values for tilt
    mag_x_tilt_comp = mag_x * math.cos(pitch_rad) + mag_z * math.sin(pitch_rad) * math.sin(roll_rad)
    mag_y_tilt_comp = mag_y * math.cos(roll_rad) - mag_z * math.sin(pitch_rad) * math.cos(roll_rad)
    # Compute the heading in radians using atan2 to get the correct quadrant
    heading_rad = math.atan2(mag_y_tilt_comp, mag_x_tilt_comp)
    # Convert to degrees
    heading_deg = math.degrees(heading_rad)
    # Adjust for magnetic declination
    heading_deg += 80-4  # Example: adjusting for local magnetic declination
    # Normalize the heading to 0-360 degrees
    if heading_deg < 0:
        heading_deg += 360
    if heading_deg>360:
        heading_deg-=360
    # Additional adjustment based on your requirement
    return heading_deg

def accel_fit(x_input, m_x, b):
    return (m_x * x_input) + b

def calculate_accel_angles(accel_x, accel_y, accel_z):
    roll = math.atan2(accel_y, math.sqrt(accel_x ** 2 + accel_z ** 2)) * (180 / math.pi)
    pitch = math.atan2(-accel_x, accel_z) * (180 / math.pi)
    return pitch, roll

while True:
    # Raw sensor readings
    accel_x, accel_y, accel_z = mpu.acceleration
    gyro_x, gyro_y, gyro_z = mpu.gyro
    calibrated_gyro_x = gyro_x - gyro_offsets[0]
    calibrated_gyro_y = gyro_y - gyro_offsets[1]
    calibrated_gyro_z = gyro_z - gyro_offsets[2]
    calibrated_accel_x = accel_fit(accel_x, *accel_coeffs[0])
    calibrated_accel_y = accel_fit(accel_y, *accel_coeffs[1])
    calibrated_accel_z = accel_fit(accel_z, *accel_coeffs[2])

    accel_pitch, accel_roll = calculate_accel_angles(calibrated_accel_x, calibrated_accel_y, calibrated_accel_z)

    # Prediction step
    gyro_pitch = state[0] + calibrated_gyro_y * dt  # pitch prediction
    gyro_roll = state[1] + calibrated_gyro_x * dt   # roll prediction
    state_pred = np.array([gyro_pitch, gyro_roll])

    # Update the error covariance matrix
    P = P + Q
    # Measurement update step
    z = np.array([accel_pitch, accel_roll])  # Measurement vector (from accelerometer)
    # Calculate the Kalman gain
    S = P + R  # Innovation covariance
    K = P @ np.linalg.inv(S)  # Kalman gain
    # Update the estimate with the measurement
    state = state_pred + K @ (z - state_pred)
    # Update the error covariance matrix
    P = (np.eye(2) - K) @ P
    # Extract pitch and roll from the state
    
    pitch, roll = state
    # pitch and roll ouput smooth
    filtered_pitch = alpha * pitch + (1 - alpha) * filtered_pitch
    filtered_roll = alpha * roll + (1 - alpha) * filtered_roll
    filtered_pitch = int(filtered_pitch)
    filtered_roll = int(filtered_roll)
    # magnetometer, compass angle part
    mag_x, mag_y, mag_z = magn.magnetic
    # Apply calibration adjustments
    adjust_x = mag_x + 59.703125
    adjust_y = mag_y - 24.584375
    adjust_z = mag_z + 76.234375
    # Apply low-pass filter  remove the effect of interference
    filtered_x = alpha_m * adjust_x + (1 - alpha_m) * filtered_x
    filtered_y = alpha_m * adjust_y + (1 - alpha_m) * filtered_y
    filtered_z = alpha_m * adjust_z + (1 - alpha_m) * filtered_z
    # Calculate the heading using filtered values
    heading = calculate_heading(filtered_x, filtered_y,filtered_z,filtered_pitch,filtered_roll)
    
    
    # Print
    print(f"Filtered Pitch: {filtered_pitch:.2f}, Filtered Roll: {filtered_roll:.2f}")
    print("Heading: {0:6.2f}Â°".format(heading))
    print("")

    time.sleep(dt)

