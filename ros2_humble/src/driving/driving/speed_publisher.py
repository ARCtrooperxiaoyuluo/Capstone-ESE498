# continuous calculate and publish at rate, 2 threads
# first order filter
# need test 
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import RPi.GPIO as IO
import time
import threading
import math

class SpeedCalculator(Node):
    def __init__(self):
        super().__init__('speed_calculator')

        # GPIO setup
        self.GPIO_num = 16
        IO.setwarnings(False)
        IO.setmode(IO.BCM)
        IO.setup(self.GPIO_num, IO.IN, IO.PUD_UP)

        # Parameters
        self.sampling_rate_hz = 500  # Hz
        self.sample_interval = 1.0 / self.sampling_rate_hz
        self.wheel_diameter = 0.1  # meters
        self.circumference = math.pi * self.wheel_diameter
        self.alpha = 0.2  # Smoothing factor for low-pass filter (0 < alpha < 1)

        # Variables for speed calculation
        self.prev_pin_val = IO.input(self.GPIO_num)
        self.last_time = time.time()
        self.current_speed_kmh = 0.0
        self.filtered_speed_kmh = 0.0  # For storing filtered speed

        # ROS2 publisher
        self.speed_publisher = self.create_publisher(Float64, 'data', 10)

        # ROS2 timer to publish speed at a fixed rate (e.g., 10 Hz)
        self.timer = self.create_timer(0.1, self.publish_speed)

        # Thread for continuous speed calculation
        self.calculation_thread = threading.Thread(target=self.calculate_speed_continuously)
        self.calculation_thread.daemon = True
        self.calculation_thread.start()

    def calculate_speed_continuously(self):
        while rclpy.ok():  # Continue running as long as ROS2 is active
            start_time = time.time()
            curr_pin_val = IO.input(self.GPIO_num)

            # Detect rising edge
            if curr_pin_val == 1 and self.prev_pin_val == 0:
                current_time = time.time()
                delta_t = current_time - self.last_time
                if delta_t > 0:
                    rps = 1 / delta_t
                    self.current_speed_kmh = rps * self.circumference * 3.6  # Convert to km/h
                    self.last_time = current_time

                    # Apply the low-pass filter
                    self.filtered_speed_kmh = (self.alpha * self.current_speed_kmh) + \
                                              ((1 - self.alpha) * self.filtered_speed_kmh)

            self.prev_pin_val = curr_pin_val

            # Sampling rate control
            elapsed_time = time.time() - start_time
            if elapsed_time < self.sample_interval:
                time.sleep(self.sample_interval - elapsed_time)

    def publish_speed(self):
        msg = Float64()
        msg.data = self.filtered_speed_kmh  # Publish the filtered speed
        self.speed_publisher.publish(msg)
        self.get_logger().info(f'Published speed: {self.filtered_speed_kmh:.4f} km/h')

def main(args=None):
    rclpy.init(args=args)
    speed_calculator = SpeedCalculator()

    try:
        rclpy.spin(speed_calculator)
    except KeyboardInterrupt:
        pass
    finally:
        speed_calculator.destroy_node()
        rclpy.shutdown()
        IO.cleanup()

if __name__ == '__main__':
    main()

