# new publish method for yaw and dist
# calculation and publish are in two threads
# tested and in use

# heading left, 0 to negative angle output
# heading right 0 to posistive angle output
# follow right side wall


# heading left, 0 to negative angle output
# heading right 0 to posistive angle output
# follow left side wall


# z_score method to remove effect of handrills
#  first order filter on yaw angle output

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
        
        # Initialize RPLidar
        self.PORT_NAME = '/dev/ttyUSB0'
        self.lidar = RPLidar(None, self.PORT_NAME, timeout=3)

        # LiDAR processing parameters
        self.MAX_LIDAR_DISTANCE = 12000  # 12 meters in millimeters
        self.angle_limits = {
            '250_290': (250, 290),  # For processing 250Â°â€“290Â° range (right side)
            '70_110': (70, 110)  # For processing 70Â°â€“110Â° range (left side)
        }

        # Variables for yaw and distance for each range
        self.yaw_250_290 = 0.0
        self.previous_yaw_250_290 = 0.0
        self.previous_dist_250_290 = 0.0
        self.distance_250_290 = 0.0

        self.yaw_70_110 = 0.0
        self.previous_yaw_70_110 = 0.0
        self.previous_dist_70_110 = 0.0;
        self.distance_70_110 = 0.0

        # Filter alpha value (for first-order filter smoothing)
        self.alpha = 0.75

        # ROS2 publishers using Float32MultiArray, each for a different range
        self.publisher_250_290 = self.create_publisher(Float32MultiArray, 'lidar_250_290_data', 10)
        self.publisher_70_110 = self.create_publisher(Float32MultiArray, 'lidar_70_110_data', 10)

        # ROS2 timers to publish data at a fixed rate (e.g., 100 Hz)
        self.timer_250_290 = self.create_timer(0.01, self.publish_lidar_250_290_data)
        self.timer_70_110 = self.create_timer(0.01, self.publish_lidar_70_110_data)

        # Thread for continuous LiDAR data processing
        self.lidar_thread = threading.Thread(target=self.process_lidar_data)
        self.lidar_thread.daemon = True
        self.lidar_thread.start()

    def process_lidar_data(self):
        """Processes LiDAR data to compute yaw and distance for the specified angle ranges."""
        try:
            for scan in self.lidar.iter_scans():
                points_250_290 = []
                points_70_110 = []

                # Process each scan reading
                for (_, angle, distance) in scan:
                    # Collect points for the 250Â°â€“290Â° range (right side)
                    if self.angle_limits['250_290'][0] <= angle < self.angle_limits['250_290'][1]:
                        radians = -(angle - 270) * pi / 180.0
                        x = distance * cos(radians)
                        y = distance * sin(radians)
                        points_250_290.append((x, y))

                    # Collect points for the 70Â°â€“110Â° range (left side)
                    if self.angle_limits['70_110'][0] <= angle < self.angle_limits['70_110'][1]:
                        radians = -(angle - 270) * pi / 180.0  # Adjust for 90 degrees center
                        x = distance * cos(radians)
                        y = distance * sin(radians)
                        points_70_110.append((x, y))

                # Process yaw and distance for 250Â°â€“290Â° range (right side)
                if points_250_290:
                    yaw, distance = self.calculate_yaw_and_distance(points_250_290, 90)
                    self.yaw_250_290 = self.first_order_filter(self.previous_yaw_250_290, yaw)
                    self.distance_250_290 = self.first_order_filter(self.previous_dist_250_290, distance)
                    self.previous_yaw_250_290 = self.yaw_250_290
                    self.previous_dist_250_290 = self.distance_250_290

                # Process yaw and distance for 70Â°â€“110Â° range (left side)
                if points_70_110:
                    yaw, distance = self.calculate_yaw_and_distance(points_70_110, 90)
                    self.yaw_70_110 = self.first_order_filter(self.previous_yaw_70_110, yaw)
                    self.distance_70_110 = self.first_order_filter(self.previous_dist_70_110, distance)
                    self.previous_yaw_70_110 = self.yaw_70_110
                    self.previous_dist_70_110 = self.distance_70_110

                # Small delay to prevent excessive CPU usage
                time.sleep(0.005)

        except KeyboardInterrupt:
            self.get_logger().info('Stopping LiDAR.')
            self.lidar.stop()
            self.lidar.disconnect()

    def calculate_yaw_and_distance(self, points, angle_correction):
        """Helper function to calculate yaw and distance from a set of LiDAR points."""
        np_points = np.array(points, dtype=np.float32)
        x_data = np_points[:, 0]
        y_data = np_points[:, 1]

        # Calculate Z-scores for x_data to detect and remove outliers
        z_scores = np.abs((x_data - np.mean(x_data)) / np.std(x_data))
        threshold = 1.5  # Define a Z-score threshold for outlier removal

        # Filter out outliers
        inliers = z_scores < threshold  # Boolean array to filter inliers
        x_cleaned = x_data[inliers]
        y_cleaned = y_data[inliers]

        if len(x_cleaned) > 1:  # Ensure enough data for fitting
            # Fit a line to the cleaned data
            A = np.vstack([x_cleaned, np.ones_like(x_cleaned)]).T
            m, c = np.linalg.lstsq(A, y_cleaned, rcond=None)[0]

            # Calculate the angle of the line with respect to the x-axis
            angle_radians = atan2(m, 1)  # arctan(slope)
            angle_degrees = degrees(angle_radians)

            # Calculate yaw: angle_correction - angle_degrees (angle_correction adjusts for range center)
            yaw = angle_correction - angle_degrees

            # Adjust yaw for outward angles:
            if yaw > 90:
                # If yaw > 90, reduce it to be in the negative range (convert outward to negative)
                yaw -= 180
            yaw = -yaw

            # Calculate the perpendicular distance from the origin to the line
            distance = abs(c) / (np.sqrt(m**2 + 1))

            return yaw, distance
        else:
            return 0.0, 0.0  # Return zero if not enough points

    def first_order_filter(self, previous_value, current_value, alpha=None):
        """Applies a first-order filter to smooth out the yaw angle."""
        if alpha is None:
            alpha = self.alpha  # Use the class-level alpha if not provided
        return alpha * previous_value + (1 - alpha) * current_value

    def publish_lidar_250_290_data(self):
        """Publishes the yaw and distance data for the 250Â°â€“290Â° range (right side)."""
        # Create and populate the Float32MultiArray message
        msg = Float32MultiArray()
        msg.data = [self.yaw_250_290, self.distance_250_290]

        # Publish the message
        self.publisher_250_290.publish(msg)
        self.get_logger().info(f'Published yaw 250Â°-290Â°: {self.yaw_250_290:.2f}Â°, distance: {self.distance_250_290:.2f} mm')

    def publish_lidar_70_110_data(self):
        """Publishes the yaw and distance data for the 70Â°â€“110Â° range (left side)."""
        # Create and populate the Float32MultiArray message
        msg = Float32MultiArray()
        msg.data = [self.yaw_70_110, self.distance_70_110]

        # Publish the message
        self.publisher_70_110.publish(msg)
        self.get_logger().info(f'Published yaw 70Â°-110Â°: {self.yaw_70_110:.2f}Â°, distance: {self.distance_70_110:.2f} mm')

def main(args=None):
    """Main function to initialize and run the LiDAR processor node."""
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
