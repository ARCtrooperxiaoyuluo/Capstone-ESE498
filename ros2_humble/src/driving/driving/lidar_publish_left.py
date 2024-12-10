# left wall

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
from adafruit_rplidar import RPLidar
from math import cos, sin, pi, atan2, degrees
import threading
import time

class LidarProcessor(Node):
    def __init__(self):
        super().__init__('lidar_processor')
        self.PORT_NAME = '/dev/ttyUSB0'
        self.lidar = RPLidar(None, self.PORT_NAME, timeout=3)
        self.MAX_LIDAR_DISTANCE = 12000  # 12 meters in millimeters
        self.angle_limit = (70, 110)  # Center = 270
        self.current_yaw = 0.0
        self.current_distance = 0.0
        self.filtered_yaw = 0.0
        self.filtered_distance = 0.0

        self.alpha = 0.8  # Smoothing factor, adjust between 0 and 1

     
        self.lidar_publisher = self.create_publisher(Float32MultiArray, 'lidar_70_110_data', 10)
        self.timer = self.create_timer(0.01, self.publish_lidar_data)
        self.lidar_thread = threading.Thread(target=self.process_lidar_data)
        self.lidar_thread.daemon = True
        self.lidar_thread.start()
    def process_lidar_data(self):
        try:
            for scan in self.lidar.iter_scans():
                points = []
                for (_, angle, distance) in scan:
                    if self.angle_limit[0] <= angle < self.angle_limit[1]:
                        radians = -(angle - 270) * pi / 180.0
                        x = distance * cos(radians)
                        y = distance * sin(radians)
                        points.append((x, y))

                if points:
                    np_points = np.array(points, dtype=np.float32)
                    x_data = np_points[:, 0]
                    y_data = np_points[:, 1]

                    # Step 1: Calculate Z-scores for x_data to detect outliers
                    z_scores = np.abs((x_data - np.mean(x_data)) / np.std(x_data))
                    threshold = 1.5  # Define a Z-score threshold to remove outliers

                    # Step 2: Filter out outliers
                    inliers = z_scores < threshold  # Boolean array to filter inliers
                    x_cleaned = x_data[inliers]
                    y_cleaned = y_data[inliers]

                    if len(x_cleaned) > 1:  # Ensure there's enough data left for fitting
                        # Step 3: Fit a line to the cleaned data
                        A = np.vstack([x_cleaned, np.ones_like(x_cleaned)]).T
                        m, c = np.linalg.lstsq(A, y_cleaned, rcond=None)[0]

                        # Calculate the angle of the line with respect to the x-axis
                        angle_radians = atan2(m, 1)  # arctan(slope)
                        angle_degrees = degrees(angle_radians)
                        yaw = 90 - angle_degrees
                        if yaw > 90:
                            yaw -= 180
                        yaw = -yaw
                        distance = abs(c) / (np.sqrt(m**2 + 1))
                        self.filtered_yaw = self.alpha * yaw + (1 - self.alpha) * self.filtered_yaw
                        self.filtered_distance = self.alpha * distance + (1 - self.alpha) * self.filtered_distance
        except KeyboardInterrupt:
            self.get_logger().info('Stopping LiDAR.')
            self.lidar.stop()
            self.lidar.disconnect()

    def publish_lidar_data(self):
        # Create and populate the Float32MultiArray message with filtered data
        msg = Float32MultiArray()
        msg.data = [self.filtered_yaw, self.filtered_distance]

        self.lidar_publisher.publish(msg)
        self.get_logger().info(f'Published yaw: {self.filtered_yaw:.2f} degrees, distance: {self.filtered_distance:.2f} mm')
def main(args=None):
    rclpy.init(args=args)
    lidar_processor = LidarProcessor()

    try:
        rclpy.spin(lidar_processor)
    except KeyboardInterrupt:
        pass
    finally:
        lidar_processor.lidar.stop()
        lidar_processor.lidar.disconnect()
        lidar_processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
