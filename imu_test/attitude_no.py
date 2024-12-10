import time
import board
import adafruit_mpu6050
import math

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

dt = 0.1  # Time step for gyro integration (s)
# Initialize previous pitch and roll for integration
previous_pitch = 0.0
previous_roll = 0.0

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
    calibrated_gyro_x = gyro_x - gyro_offsets[0]
    calibrated_gyro_y = gyro_y - gyro_offsets[1]
    calibrated_gyro_z = gyro_z - gyro_offsets[2]

    # Apply calibration coefficients to accelerometer readings
    calibrated_accel_x = accel_fit(accel_x, *accel_coeffs[0])
    calibrated_accel_y = accel_fit(accel_y, *accel_coeffs[1])
    calibrated_accel_z = accel_fit(accel_z, *accel_coeffs[2])

    # Calculate pitch and roll using accelerometer
    accel_pitch, accel_roll = calculate_accel_angles(calibrated_accel_x, calibrated_accel_y, calibrated_accel_z)

    # Calculate pitch and roll using gyroscope (integrate gyro rates)
    gyro_pitch = previous_pitch + calibrated_gyro_y * dt
    gyro_roll = previous_roll + calibrated_gyro_x * dt

    # Update previous angles for next iteration
    previous_pitch = gyro_pitch
    previous_roll = gyro_roll

    # Print accelerometer and gyroscope-based angles
    print(f"Accel Pitch: {accel_pitch:.2f}, Accel Roll: {accel_roll:.2f}, Gyro Pitch: {gyro_pitch:.2f}, Gyro Roll: {gyro_roll:.2f}")

    time.sleep(dt)

