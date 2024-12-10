import numpy as np
import time
import board
import adafruit_mpu6050

# Initialize MPU6050 using I2C
i2c = board.I2C()
mpu = adafruit_mpu6050.MPU6050(i2c)

# Set gyroscope offsets (replace with your actual calibration values)
gyro_offsets = [-0.0049, 0.0186, -0.0045]

# Set accelerometer calibration coefficients (replace with your actual calibration values)
accel_coeffs = [
    [0.10146388, -0.01743124],  # X-axis calibration [m_x, b]
    [0.10132916, -0.01252953],  # Y-axis calibration [m_y, b]
    [-0.09826508, -0.49321221]  # Z-axis calibration [m_z, b]
]

# Function to apply accelerometer calibration
def accel_fit(x_input, m_x, b):
    return (m_x * x_input) + b

# Collect data for calculating variance
num_samples = 500  # Adjust sample size as needed
accel_data = []
gyro_data = []

for _ in range(num_samples):
    # Raw sensor readings
    accel_x, accel_y, accel_z = mpu.acceleration
    gyro_x, gyro_y, gyro_z = mpu.gyro

    # Apply calibration to gyro readings
    calibrated_gyro_x = gyro_x - gyro_offsets[0]
    calibrated_gyro_y = gyro_y - gyro_offsets[1]
    calibrated_gyro_z = gyro_z - gyro_offsets[2]

    # Apply calibration to accelerometer readings
    calibrated_accel_x = accel_fit(accel_x, *accel_coeffs[0])
    calibrated_accel_y = accel_fit(accel_y, *accel_coeffs[1])
    calibrated_accel_z = accel_fit(accel_z, *accel_coeffs[2])

    # Append calibrated data to lists
    accel_data.append([calibrated_accel_x, calibrated_accel_y, calibrated_accel_z])
    gyro_data.append([calibrated_gyro_x, calibrated_gyro_y, calibrated_gyro_z])

    time.sleep(0.1)

# Convert to numpy arrays for easier processing
accel_data = np.array(accel_data)
gyro_data = np.array(gyro_data)

# Calculate variances
accel_variance = np.var(accel_data, axis=0)
gyro_variance = np.var(gyro_data, axis=0)

print("Accelerometer Variance (for R):", accel_variance)
print("Gyroscope Variance (for R):", gyro_variance)

# Use these variances as initial values for R's diagonal elements

