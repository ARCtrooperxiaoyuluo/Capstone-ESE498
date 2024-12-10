# calculate compass angle 
import time
import math
import board
import busio
import adafruit_mmc56x3

# Initialize I2C communication and magnetometer
i2c = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_mmc56x3.MMC5603(i2c, address=0x30)

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
    heading_deg = heading_deg+72
    # Normalize the heading to 0-360 degrees
    if heading_deg < 0:
        heading_deg += 360
    if heading_deg>360:
        heading_deg-=360
    return heading_deg

def calculate_adjusted_heading(heading):
    # Define intervals and the corresponding increments for linear interpolation
    reference_points = [
        (34, 6), (52, 8), (71, 9), (88, 12), 
        (105, 15), (120, 23), (150, 35), (180, 40),(210,40), 
        (253, 30), (279, 0), (296, 0), (310, 0),(328,0)
    ]
    
    # Check if the heading is outside the specified range
    if heading < reference_points[0][0]:
        return heading
    elif heading > reference_points[-1][0]:
        return heading

    # Apply linear interpolation within each interval
    for (low_heading, low_addition), (high_heading, high_addition) in zip(reference_points, reference_points[1:]):
        if low_heading <= heading <= high_heading:
            # Linear interpolation for the addition based on position within the interval
            addition = low_addition + (heading - low_heading) * (high_addition - low_addition) / (high_heading - low_heading)
            adjusted_heading = heading + addition
            return adjusted_heading

    # If no intervals match (should not happen in this case), return original heading
    return heading

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
    heading1 = calculate_adjusted_heading(heading)
    # Print magnetic field values and calculated heading
    print("Filtered Magnetic Field: X:{0:10.2f} uT, Y:{1:10.2f} uT, Z:{2:10.2f} uT".format(filtered_x, filtered_y, filtered_z))
    print("Heading: {0:6.2f}Â°".format(heading))
    print(heading1)
    print("")

    # Pause before next reading
    time.sleep(1.0)

