# calibrate gyro with plot
# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT

import time
import board
import adafruit_mpu6050
import numpy as np
import matplotlib.pyplot as plt

# Initialize MPU6050 using I2C
i2c = board.I2C()  # uses board.SCL and board.SDA
mpu = adafruit_mpu6050.MPU6050(i2c)

def gyro_cal(cal_size=500):
    print("-" * 50)
    print('Gyro Calibrating - Keep the IMU Steady')
    
    # Clear buffer before calibration
    [mpu.gyro for _ in range(cal_size)]  # Dummy read to clear buffer
    mpu_array = []
    gyro_offsets = [0.0, 0.0, 0.0]

    while True:
        wx, wy, wz = mpu.gyro  # Get gyro values

        mpu_array.append([wx, wy, wz])  # Gyro vector append
        if len(mpu_array) == cal_size:
            for qq in range(3):
                gyro_offsets[qq] = np.mean(np.array(mpu_array)[:, qq])  # Calculate average
            break

    print('Gyro Calibration Complete')
    print(f"Gyro Offsets: wx: {gyro_offsets[0]:.4f}, wy: {gyro_offsets[1]:.4f}, wz: {gyro_offsets[2]:.4f}")
    return gyro_offsets

if __name__ == '__main__':
    cal_size = 500  # Points to use for calibration

    # Gyroscope Offset Calculation
    gyro_offsets = gyro_cal(cal_size)  # Calculate gyro offsets

    # Record new data
    data = np.array([mpu.gyro for _ in range(cal_size)])  # Get new gyro data

    # Plot with and without offsets
    plt.style.use('ggplot')
    fig, axs = plt.subplots(2, 1, figsize=(12, 9))
    for ii in range(3):
        axs[0].plot(data[:, ii], label=f'Gyro {ii+1} Uncalibrated')
        axs[1].plot(data[:, ii] - gyro_offsets[ii], label=f'Gyro {ii+1} Calibrated')
    
    axs[0].legend(fontsize=14)
    axs[1].legend(fontsize=14)
    axs[0].set_ylabel('Gyro [°/s]', fontsize=18)
    axs[1].set_ylabel('Gyro [°/s]', fontsize=18)
    axs[1].set_xlabel('Sample', fontsize=18)
    axs[0].set_ylim([-2, 2])
    axs[1].set_ylim([-2, 2])
    axs[0].set_title('Gyroscope Calibration Offset Correction', fontsize=22)

    # Save and show plot
    fig.savefig('gyro_calibration_output.png', dpi=300, bbox_inches='tight', facecolor='#FCFCFC')
    plt.show()

