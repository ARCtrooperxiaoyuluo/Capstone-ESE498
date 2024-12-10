######################################################
# Copyright (c) 2021 Maker Portal LLC
# Author: Joshua Hrisko
# Adapted for MMC56x3 magnetometer using Adafruit Library
######################################################
#
# This code reads data from the MMC56x3 magnetometer
# using the Adafruit library and solves for the hard
# iron offset for a magnetometer using a calibration block.
#
######################################################

import time
import numpy as np
import matplotlib.pyplot as plt
import board
import busio
import adafruit_mmc56x3

# Initialize I2C bus and create sensor object
i2c = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_mmc56x3.MMC5603(i2c, address=0x30)

# Function to read magnetic data from the MMC56x3 sensor
def read_mmc56x3():
    # Use the Adafruit library to get magnetic data from the sensor
    mag_data = sensor.magnetic  # returns a tuple (x, y, z) in microteslas
    
    # Unpack the data
    x, y, z = mag_data
    print(f"Raw MMC56x3 Data: X={x}, Y={y}, Z={z}")
    return x, y, z

# Function to remove outliers in the calibration data
def outlier_removal(x_ii, y_ii):
    x_diff = np.append(0.0, np.diff(x_ii))  # looking for outliers
    stdev_amt = 5.0  # standard deviation multiplier
    x_outliers = np.where(np.abs(x_diff) > np.abs(np.mean(x_diff)) + (stdev_amt * np.std(x_diff)))
    y_diff = np.append(0.0, np.diff(y_ii))  # looking for outliers
    y_outliers = np.where(np.abs(y_diff) > np.abs(np.mean(y_diff)) + (stdev_amt * np.std(y_diff)))
    
    if len(x_outliers) != 0:
        x_ii[x_outliers] = np.nan  # null outlier
        y_ii[x_outliers] = np.nan  # null outlier
    if len(y_outliers) != 0:
        y_ii[y_outliers] = np.nan  # null outlier
        x_ii[y_outliers] = np.nan  # null outlier
    
    return x_ii, y_ii

# Magnetometer calibration function
def mag_cal():
    print("-" * 50)
    print("Magnetometer Calibration")
    mag_cal_rotation_vec = []  # variable for calibration calculations
    
    for qq, ax_qq in enumerate(mag_cal_axes):
        input("-" * 8 + " Press Enter and Start Rotating the IMU Around the " + ax_qq + "-axis")
        print("\t When Finished, Press CTRL+C")
        mag_array = []
        t0 = time.time()
        
        while True:
            try:
                # Read MMC56x3 data using Adafruit library
                mx, my, mz = read_mmc56x3()  # Read MMC56x3 data
            except KeyboardInterrupt:
                break
            except:
                continue
            
            mag_array.append([mx, my, mz])  # Append magnetic field data
            
        mag_array = mag_array[20:]  # throw away first few points (buffer clearing)
        mag_cal_rotation_vec.append(mag_array)  # calibration array
        print("Sample Rate: {0:2.0f} Hz".format(len(mag_array) / (time.time() - t0)))
    
    mag_cal_rotation_vec = np.array(mag_cal_rotation_vec)  # make numpy array
    ak_fit_coeffs = []
    indices_to_save = [0, 0, 1]  # indices to save as offsets
    
    for mag_ii, mags in enumerate(mag_cal_rotation_vec):
        mags = np.array(mags)  # mag numpy array
        x, y = mags[:, cal_rot_indices[mag_ii][0]], mags[:, cal_rot_indices[mag_ii][1]]  # sensors to analyze
        x, y = outlier_removal(x, y)  # outlier removal
        y_0 = (np.nanmax(y) + np.nanmin(y)) / 2.0  # y-offset
        x_0 = (np.nanmax(x) + np.nanmin(x)) / 2.0  # x-offset
        ak_fit_coeffs.append([x_0, y_0][indices_to_save[mag_ii]])  # append to offset
        
    return ak_fit_coeffs, mag_cal_rotation_vec

# Plotting function to see the calibration impact
def mag_cal_plot():
    plt.style.use('ggplot')  # start figure
    fig, axs = plt.subplots(1, 2, figsize=(12, 7))  # start figure
    
    for mag_ii, mags in enumerate(mag_cal_rotation_vec):
        mags = np.array(mags)  # magnetometer numpy array
        x, y = mags[:, cal_rot_indices[mag_ii][0]], mags[:, cal_rot_indices[mag_ii][1]]
        x, y = outlier_removal(x, y)  # outlier removal
        axs[0].scatter(x, y, label='Rotation Around ${0}$-axis (${1},{2}$)'.format(
            mag_cal_axes[mag_ii], mag_labels[cal_rot_indices[mag_ii][0]], mag_labels[cal_rot_indices[mag_ii][1]]))
        axs[1].scatter(x - mag_coeffs[cal_rot_indices[mag_ii][0]], y - mag_coeffs[cal_rot_indices[mag_ii][1]], 
                       label='Rotation Around ${0}$-axis (${1},{2}$)'.format(
                           mag_cal_axes[mag_ii], mag_labels[cal_rot_indices[mag_ii][0]], mag_labels[cal_rot_indices[mag_ii][1]]))
    
    axs[0].set_title('Before Hard Iron Offset')  # plot title
    axs[1].set_title('After Hard Iron Offset')  # plot title
    
    mag_lims = [np.nanmin(np.nanmin(mag_cal_rotation_vec)), np.nanmax(np.nanmax(mag_cal_rotation_vec))]  # array limits
    mag_lims = [-1.1 * np.max(mag_lims), 1.1 * np.max(mag_lims)]  # axes limits
    
    for jj in range(0, 2):
        axs[jj].set_ylim(mag_lims)  # set limits
        axs[jj].set_xlim(mag_lims)  # set limits
        axs[jj].legend()  # legend
        axs[jj].set_aspect('equal', adjustable='box')  # square axes
    
    fig.savefig('mag_cal_hard_offset_white.png', dpi=300, bbox_inches='tight', facecolor='#FFFFFF')  # save figure
    plt.show()  # show plot

# Main function to perform magnetometer calibration and plotting
if __name__ == '__main__':
    # Labels and setup
    mag_labels = ['m_x', 'm_y', 'm_z']  # mag labels for plots
    mag_cal_axes = ['z', 'y', 'x']  # axis order being rotated
    cal_rot_indices = [[0, 1], [1, 2], [0, 2]]  # indices of heading for each axis

    mag_coeffs, mag_cal_rotation_vec = mag_cal()  # grab mag coefficients
    print(mag_coeffs)
    # Plot with and without offsets
    mag_cal_plot()  # plot un-calibrated and calibrated results