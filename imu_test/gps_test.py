import time
import board
import busio
import adafruit_gps

# Set up I2C communication on the default pins (SCL and SDA)
i2c = busio.I2C(board.SCL, board.SDA)

# Create the GPS module instance using I2C
gps = adafruit_gps.GPS_GtopI2C(i2c,address=0x68, debug=False)  # Use I2C interface

# Initialize the GPS with NMEA data output and update rate configurations
# Basic GGA and RMC info, which provides location and timestamp data
gps.send_command(b"PMTK314,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0")

# Set update rate to once per second (1Hz)
gps.send_command(b"PMTK220,1000")

# Start main loop to capture and print GPS data
last_print = time.monotonic()
while True:
    # Ensure gps.update() is called every loop iteration
    gps.update()
    
    # Print GPS data every second if a fix is acquired
    current = time.monotonic()
    if current - last_print >= 1.0:
        last_print = current
        if not gps.has_fix:
            print("Waiting for GPS fix...")
            continue

        # GPS fix acquired, print location and other GPS info
        print("=" * 40)
        print(
            "Fix timestamp: {}/{}/{} {:02}:{:02}:{:02}".format(
                gps.timestamp_utc.tm_mon,
                gps.timestamp_utc.tm_mday,
                gps.timestamp_utc.tm_year,
                gps.timestamp_utc.tm_hour,
                gps.timestamp_utc.tm_min,
                gps.timestamp_utc.tm_sec,
            )
        )
        print("Latitude: {0:.6f} degrees".format(gps.latitude))
        print("Longitude: {0:.6f} degrees".format(gps.longitude))
        print("Fix quality: {}".format(gps.fix_quality))

        # Optional data (check for None to ensure presence)
        if gps.satellites is not None:
            print("Number of satellites: {}".format(gps.satellites))
        if gps.altitude_m is not None:
            print("Altitude: {} meters".format(gps.altitude_m))
        if gps.speed_knots is not None:
            print("Speed: {} knots".format(gps.speed_knots))
        if gps.speed_kmh is not None:
            print("Speed: {} km/h".format(gps.speed_kmh))
        if gps.track_angle_deg is not None:
            print("Track angle: {} degrees".format(gps.track_angle_deg))
        if gps.horizontal_dilution is not None:
            print("Horizontal dilution: {}".format(gps.horizontal_dilution))
        if gps.height_geoid is not None:
            print("Height above geoid: {} meters".format(gps.height_geoid))

