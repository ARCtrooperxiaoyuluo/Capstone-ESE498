# reading of imu with calibration 
# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT

import time
import board
import adafruit_mpu6050

# Initialize MPU6050 using I2C
i2c = board.I2C()  # uses board.SCL and board.SDA
mpu = adafruit_mpu6050.MPU6050(i2c)

# Set gyroscope offsets (example values, replace with your calibration values)
gyro_offsets = [-0.0049, 0.0186, -0.0045]  # Example offsets for gyro X, Y, Z

# Set accelerometer calibration coefficients (example values, replace with your calibration values)
# Each axis has [slope (m_x), intercept (b)] values from the calibration step
accel_coeffs = [
    [0.10146388, -0.01743124],  # X-axis calibration [m_x, b]
    [0.10132916, -0.01252953],  # Y-axis calibration [m_y, b]
    [-0.09826508, -0.49321221]   # Z-axis calibration [m_z, b]
]

# Function to apply accelerometer calibration
def accel_fit(x_input, m_x, b):
    return (m_x * x_input) + b

while True:
    # Raw sensor readings
    accel_x, accel_y, accel_z = mpu.acceleration
    gyro_x, gyro_y, gyro_z = mpu.gyro
    temperature = mpu.temperature

    # Apply calibration offsets to gyro readings
    calibrated_gyro_x = gyro_x - gyro_offsets[0]
    calibrated_gyro_y = gyro_y - gyro_offsets[1]
    calibrated_gyro_z = gyro_z - gyro_offsets[2]

    # Apply calibration coefficients to accelerometer readings
    calibrated_accel_x = accel_fit(accel_x, *accel_coeffs[0])
    calibrated_accel_y = accel_fit(accel_y, *accel_coeffs[1])
    calibrated_accel_z = accel_fit(accel_z, *accel_coeffs[2])

    # Print calibrated sensor data
    print("Calibrated Acceleration: X:%.2f, Y: %.2f, Z: %.2f m/s^2" % (calibrated_accel_x, calibrated_accel_y, calibrated_accel_z))
    print("Calibrated Gyro: X:%.2f, Y: %.2f, Z: %.2f rad/s" % (calibrated_gyro_x, calibrated_gyro_y, calibrated_gyro_z))
    print("")

    time.sleep(0.1)

