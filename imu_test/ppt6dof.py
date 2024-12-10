import time
import board
import adafruit_mpu6050
import math
import numpy as np
import matplotlib.pyplot as plt

# Initialize MPU6050 using I2C
i2c = board.I2C()  # uses board.SCL and board.SDA
mpu = adafruit_mpu6050.MPU6050(i2c)

# Calibration coefficients and offsets
gyro_offsets = [-0.0049, 0.0186, -0.0045]  # Gyro offsets
accel_coeffs = [
    [0.10146388, -0.01743124],  # X-axis calibration [m_x, b]
    [0.10132916, -0.01252953],  # Y-axis calibration [m_y, b]
    [-0.09826508, -0.49321221]   # Z-axis calibration [m_z, b]
]

# Complementary filter constants
alpha_complementary = 0.93
dt = 0.05  # Sampling time (s)

# Kalman filter parameters
Q = np.array([[0.001, 0], [0, 0.001]])  # Process noise covariance
R = np.array([[0.05, 0], [0, 0.05]])    # Measurement noise covariance
P = np.array([[1, 0], [1, 0]])          # Initial estimation error covariance
state = np.array([0.0, 0.0])            # Initial state [pitch, roll]

# Initialize variables for complementary filter
previous_pitch = 0.0
previous_roll = 0.0

# Initialize variables for gyro-based pitch and roll
gyro_pitch = 0.0
gyro_roll = 0.0

# Data storage for plotting
timestamps = []
basic_pitch_roll = []
complementary_pitch_roll = []
kalman_pitch_roll = []
gyro_pitch_roll = []

# Function to apply accelerometer calibration
def accel_fit(x_input, m_x, b):
    return (m_x * x_input) + b

# Function to calculate pitch and roll from accelerometer data
def calculate_accel_angles(accel_x, accel_y, accel_z):
    roll = math.atan2(accel_y, math.sqrt(accel_x**2 + accel_z**2)) * (180 / math.pi)
    pitch = math.atan2(-accel_x, accel_z) * (180 / math.pi)
    return pitch, roll

# Start data collection
start_time = time.time()
while len(timestamps) < 500:  # Collect 100 data points for plotting
    # Raw sensor readings
    accel_x, accel_y, accel_z = mpu.acceleration
    gyro_x, gyro_y, gyro_z = mpu.gyro

    # Apply calibration
    calibrated_gyro_x = (gyro_x - gyro_offsets[0])
    calibrated_gyro_y = (gyro_y - gyro_offsets[1])
    calibrated_gyro_z = (gyro_z - gyro_offsets[2])
    calibrated_accel_x = accel_fit(accel_x, *accel_coeffs[0])
    calibrated_accel_y = accel_fit(accel_y, *accel_coeffs[1])
    calibrated_accel_z = accel_fit(accel_z, *accel_coeffs[2])

    # Basic accelerometer calculation
    accel_pitch, accel_roll = calculate_accel_angles(
        calibrated_accel_x, calibrated_accel_y, calibrated_accel_z
    )

    # Gyro integration for pitch and roll
    gyro_pitch += (calibrated_gyro_y*180/3.14) * dt
    gyro_roll += (calibrated_gyro_x*180/3.14) * dt
    #gyro_pitch = gyro_pitch*(180/3.14)
    #gyro_roll = gyro_roll*(180/3.14)
    # Complementary filter
    comp_pitch = alpha_complementary * (previous_pitch + calibrated_gyro_y * dt) + \
                 (1 - alpha_complementary) * accel_pitch
    comp_roll = alpha_complementary * (previous_roll + calibrated_gyro_x * dt) + \
                (1 - alpha_complementary) * accel_roll
    previous_pitch = comp_pitch
    previous_roll = comp_roll

    # Kalman filter
    gyro_pitch_pred = state[0] + calibrated_gyro_y * dt  # Pitch prediction
    gyro_roll_pred = state[1] + calibrated_gyro_x * dt   # Roll prediction
    state_pred = np.array([gyro_pitch_pred, gyro_roll_pred])
    P = P + Q
    z = np.array([accel_pitch, accel_roll])  # Measurement vector
    S = P + R
    K = P @ np.linalg.inv(S)
    state = state_pred + K @ (z - state_pred)
    P = (np.eye(2) - K) @ P
    kalman_pitch, kalman_roll = state

    # Store data for plotting
    current_time = time.time() - start_time
    timestamps.append(current_time)
    basic_pitch_roll.append((accel_pitch, accel_roll))
    complementary_pitch_roll.append((comp_pitch, comp_roll))
    kalman_pitch_roll.append((kalman_pitch, kalman_roll))
    gyro_pitch_roll.append((gyro_pitch, gyro_roll))

    # Print results
    print(f"Time: {current_time:.2f}s")
    print(f"Basic Pitch/Roll: {accel_pitch:.2f}, {accel_roll:.2f}")
    print(f"Gyro Pitch/Roll: {gyro_pitch:.2f}, {gyro_roll:.2f}")
    print(f"Complementary Pitch/Roll: {comp_pitch:.2f}, {comp_roll:.2f}")
    print(f"Kalman Pitch/Roll: {kalman_pitch:.2f}, {kalman_roll:.2f}")
    print("-" * 50)

    time.sleep(dt)

# Plot results
basic_pitch, basic_roll = zip(*basic_pitch_roll)
gyro_pitch_vals, gyro_roll_vals = zip(*gyro_pitch_roll)
comp_pitch, comp_roll = zip(*complementary_pitch_roll)
kal_pitch, kal_roll = zip(*kalman_pitch_roll)

plt.figure(figsize=(12, 8))

# Plot pitch
plt.subplot(2, 1, 1)
plt.plot(timestamps, basic_pitch, label="Basic (Accel)", linestyle="--")
plt.plot(timestamps, gyro_pitch_vals, label="Gyro (Integration)", linestyle="-.")
plt.plot(timestamps, comp_pitch, label="Complementary")
plt.plot(timestamps, kal_pitch, label="Kalman")
plt.title("Pitch")
plt.legend()
plt.grid()

# Plot roll
plt.subplot(2, 1, 2)
plt.plot(timestamps, basic_roll, label="Basic (Accel)", linestyle="--")
plt.plot(timestamps, gyro_roll_vals, label="Gyro (Integration)", linestyle="-.")
plt.plot(timestamps, comp_roll, label="Complementary")
plt.plot(timestamps, kal_roll, label="Kalman")
plt.title("Roll")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("pitch_roll_plot_with_gyro.png")

