# cal speed with filter
import RPi.GPIO as IO
import time
import math

IO.setwarnings(False)
IO.setmode(IO.BCM)
GPIO_num = 16
IO.setup(GPIO_num, IO.IN, IO.PUD_UP)

sampling_rate_hz = 500  # Hz
sample_interval = 1.0 / sampling_rate_hz

wheel_diameter = 0.1
circumference = math.pi * wheel_diameter

prev_pin_val = IO.input(GPIO_num)  
last_time = time.time()  

# First-order filter parameters
alpha = 0.2

filtered_speed = 0.0  # Initial filtered speed

# Open file to write filtered speed data
with open("wheel_speed_filter.txt", "a") as file:
    while True:
        start_time = time.time()
        curr_pin_val = IO.input(GPIO_num)
        
        if curr_pin_val == 1 and prev_pin_val == 0:
            current_time = time.time()
            delta_t = current_time - last_time 
            
            if delta_t > 0:
                rps = 1 / delta_t 
                speed_kmh = rps * circumference * 3.6  # to km/h
                
                # Apply first-order filter
                filtered_speed = alpha * speed_kmh + (1 - alpha) * filtered_speed
                
                # Write to file
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
                file.write(f"{timestamp}, {filtered_speed:.3f} km/h\n")
                file.flush()  # Ensure data is written to the file immediately
                
                print(f"Filtered Wheel Speed: {filtered_speed:.3f} km/h")
            
            last_time = current_time

        prev_pin_val = curr_pin_val

        # Sampling rate control
        elapsed_time = time.time() - start_time
        if elapsed_time < sample_interval:
            time.sleep(sample_interval - elapsed_time)

