import RPi.GPIO as IO
import time
import sys
import argparse
import busio
import smbus
from time import sleep
import adafruit_mcp3xxx.mcp3008 as MCP
from adafruit_mcp3xxx.analog_in import AnalogIn
import math

# Setup GPIO
IO.setwarnings(False)
IO.setmode(IO.BCM)

GPIO_num = 16
IO.setup(GPIO_num, IO.IN, IO.PUD_UP)

# Parameters for sampling rate
sampling_rate_hz = 500  # Desired sampling rate in Hz
sample_interval = 1.0 / sampling_rate_hz

# Wheel diameter in meters (for example, 0.1 meters)
wheel_diameter = 0.1
circumference = math.pi * wheel_diameter  # Circumference in meters

prev_pin_val = IO.input(GPIO_num)  # Initial state of the GPIO pin
last_time = time.time()  # Initialize the last detection time

# Open a text file in append mode
with open("wheel_speed_nofilter.txt", "a") as file:
    while True:
        start_time = time.time()
        
        # Read the GPIO pin value
        curr_pin_val = IO.input(GPIO_num)
        
        # Check for 0 to 1 transition (rising edge)
        if curr_pin_val == 1 and prev_pin_val == 0:
            current_time = time.time()
            delta_t = current_time - last_time  # Calculate time interval between rising edges
            
            if delta_t > 0:
                # Calculate wheel speed in km/h
                rps = 1 / delta_t  # Revolutions per second
                speed_kmh = rps * circumference * 3.6  # Convert to km/h
                
                # Print the wheel speed
                print(f"Wheel Speed: {speed_kmh:.4f} km/h")
                
                # Write the wheel speed to the file with a timestamp
                file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Wheel Speed: {speed_kmh:.2f} km/h\n")
                file.flush()  # Ensure the data is written to the file immediately
            
            # Update the last detection time
            last_time = current_time
        
        # Update the previous pin value
        prev_pin_val = curr_pin_val
        
        # Sleep to maintain the desired sampling rate
        elapsed_time = time.time() - start_time
        if elapsed_time < sample_interval:
            time.sleep(sample_interval - elapsed_time)
