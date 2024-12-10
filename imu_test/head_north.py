import time
import math
import board
import busio
import adafruit_mmc56x3

# Initialize I2C communication and magnetometer
i2c = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_mmc56x3.MMC5603(i2c, address=0x30)

# Set magnetic declination in degrees (based on your geographic location)
MAGNETIC_DECLINATION = -6.0  # example for a location with -5° declination

# Low-pass filter smoothing factor for 1 Hz cutoff frequency
alpha = 0.2

# Initialize previous filtered values
filtered_x, filtered_y, filtered_z = 0.0, 0.0, 0.0

def calculate_heading(mag_x, mag_y):
    """
    Calculate heading (in degrees) from magnetometer X and Y values.
    Adjust with magnetic declination for true north.
    """
    # Compute the heading in radians using atan2 to get the correct quadrant
    
    
    heading_deg = math.atan2(mag_y, -mag_x) * 180 / math.pi
    heading_deg = heading_deg+80
    # Normalize the heading to 0-360 degrees
    if heading_deg < 0:
        heading_deg += 360
    if heading_deg>360:
        heading_deg-=360
    return heading_deg

while True:
    # Read magnetic field values in microteslas (uT)
    mag_x, mag_y, mag_z = sensor.magnetic

    # Apply calibration adjustments
    adjust_x = mag_x + 59.703125
    adjust_y = mag_y - 24.584375
    adjust_z = mag_z + 76.234375

    # Apply low-pass filter
    filtered_x = alpha * adjust_x + (1 - alpha) * filtered_x
    filtered_y = alpha * adjust_y + (1 - alpha) * filtered_y
    filtered_z = alpha * adjust_z + (1 - alpha) * filtered_z

    # Calculate the heading using filtered values
    heading = calculate_heading(filtered_x, filtered_y)

    # Print magnetic field values and calculated heading
    print("Filtered Magnetic Field: X:{0:10.2f} uT, Y:{1:10.2f} uT, Z:{2:10.2f} uT".format(filtered_x, filtered_y, filtered_z))
    print("Heading: {0:6.2f}°".format(heading))
    print("")

    # Pause before next reading
    time.sleep(1.0)

