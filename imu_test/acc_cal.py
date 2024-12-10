# calibrate acc with plot
# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT

import time
import board
import adafruit_mpu6050
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Initialize MPU6050 using I2C
i2c = board.I2C()  # uses board.SCL and board.SDA
mpu = adafruit_mpu6050.MPU6050(i2c)

def accel_fit(x_input, m_x, b):
    return (m_x * x_input) + b  # Fit equation for accel calibration

def accel_cal(cal_size=1000):
    print("-" * 50)
    print("Accelerometer Calibration")
    mpu_offsets = [[], [], []]  # Offset array to be printed
    axis_vec = ['z', 'y', 'x']  # Axis labels
    cal_directions = ["upward", "downward", "perpendicular to gravity"]  # Calibration directions
    cal_indices = [2, 1, 0]  # Axis indices

    for qq, ax_qq in enumerate(axis_vec):
        ax_offsets = [[], [], []]
        print("-" * 50)
        for direc_ii, direc in enumerate(cal_directions):
            input("-" * 8 + " Press Enter and Keep IMU Steady to Calibrate the Accelerometer with the -" +
                  ax_qq + "-axis pointed " + direc)
            [mpu.acceleration for _ in range(cal_size)]  # Clear buffer before reading
            mpu_array = []
            while len(mpu_array) < cal_size:
                ax, ay, az = mpu.acceleration  # Get accelerometer data
                mpu_array.append([ax, ay, az])  # Append to array

            ax_offsets[direc_ii] = np.array(mpu_array)[:, cal_indices[qq]]  # Offsets for direction

        # Use three calibrations (+1g, -1g, 0g) for linear fit
        popts, _ = curve_fit(accel_fit, np.append(np.append(ax_offsets[0], ax_offsets[1]), ax_offsets[2]),
                             np.append(np.append(1.0 * np.ones(np.shape(ax_offsets[0])),
                                                 -1.0 * np.ones(np.shape(ax_offsets[1]))),
                                       0.0 * np.ones(np.shape(ax_offsets[2]))),
                             maxfev=10000)
        mpu_offsets[cal_indices[qq]] = popts  # Place slope and intercept in offset array
    print('Accelerometer Calibration Complete')
    print(f"Calibration Coefficients: {mpu_offsets}")
    return mpu_offsets

if __name__ == '__main__':
    cal_size = 1000  # Number of points to use for calibration

    # Accelerometer Gravity Calibration
    accel_labels = ['a_x', 'a_y', 'a_z']  # Accelerometer labels for plots
    accel_coeffs = accel_cal(cal_size)  # Grab accelerometer coefficients

    # Record new data
    data = np.array([mpu.acceleration for _ in range(cal_size)])  # Get new accelerometer data

    # Plot with and without offsets
    plt.style.use('ggplot')
    fig, axs = plt.subplots(2, 1, figsize=(12, 9))
    for ii in range(3):
        axs[0].plot(data[:, ii], label=f'${accel_labels[ii]}$, Uncalibrated')
        axs[1].plot(accel_fit(data[:, ii], *accel_coeffs[ii]), label=f'${accel_labels[ii]}$, Calibrated')

    axs[0].legend(fontsize=14)
    axs[1].legend(fontsize=14)
    axs[0].set_ylabel('$a_{x, y, z}$ [g]', fontsize=18)
    axs[1].set_ylabel('$a_{x, y, z}$ [g]', fontsize=18)
    axs[1].set_xlabel('Sample', fontsize=18)
    axs[0].set_ylim([-2, 2])
    axs[1].set_ylim([-2, 2])
    axs[0].set_title('Accelerometer Calibration Correction', fontsize=18)

    # Save and show plot
    fig.savefig('accel_calibration_output.png', dpi=300, bbox_inches='tight', facecolor='#FCFCFC')
    plt.show()

