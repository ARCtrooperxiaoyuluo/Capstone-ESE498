import cv2
import time
import requests
import io
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray  # For publishing slope and intercept values

# Global variable for storing the latest frame
latest_frame = None

class VideoStream:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src, cv2.CAP_V4L)
        if not self.capture.isOpened():
            raise ValueError(f"Unable to open video source {src}")
        
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.latest_frame = None
        self.stopped = False
        self.lock = threading.Lock()
        
        # Start the thread to read frames
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
    
    def update(self):
        while not self.stopped:
            ret, frame = self.capture.read()
            if not ret:
                print("Failed to grab frame")
                self.stop()
                break
            with self.lock:
                self.latest_frame = frame
            time.sleep(0.001)
    
    def read(self):
        with self.lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None
        return frame
    
    def stop(self):
        self.stopped = True
        self.thread.join()
        self.capture.release()

def upload_image(image_bytes, filename, tally):
    url = 'http://107.203.186.66:80/upload'  # Replace with the server's IP address
    files = {'file': (filename, image_bytes, 'image/jpeg')}
    data = {'tally': str(tally)}
    try:
        response = requests.post(url, files=files, data=data, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        print("Response is not in JSON format. Here’s the raw response:")
        print(response.text)
        return None

class SlopeInterceptPublisher(Node):
    def __init__(self):
        super().__init__('slope_intercept_publisher')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'slope_intercept', 10)
        self.slope = None
        self.intercept = None
        self.lock = threading.Lock()
        
        # Start the publishing thread
        self.publisher_thread = threading.Thread(target=self.publish_slope_intercept, daemon=True)
        self.publisher_thread.start()
    
    def set_slope_intercept(self, slope, intercept):
        with self.lock:
            self.slope = slope
            self.intercept = intercept
    
    def publish_slope_intercept(self):
        while rclpy.ok():
            with self.lock:
                if self.slope is not None and self.intercept is not None:
                    msg = Float32MultiArray()
                    msg.data = [self.slope, self.intercept]
                    self.publisher_.publish(msg)
                    self.get_logger().info(f'Published: Slope = {self.slope}, Intercept = {self.intercept}')
            time.sleep(0.1)

def main():
    rclpy.init()
    publisher_node = SlopeInterceptPublisher()
    tally = 1
    min_interval = 0.1  # Minimum time between requests in seconds

    try:
        video_stream = VideoStream(src=0)
    except ValueError as e:
        print(e)
        rclpy.shutdown()
        return

    try:
        while rclpy.ok():
            start_time = time.time()

            frame = video_stream.read()
            if frame is not None:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                success, encoded_image = cv2.imencode('.jpg', frame)
                if not success:
                    print("Failed to encode frame.")
                    continue
                
                image_bytes = encoded_image.tobytes()
                filename = f'frame_{int(time.time() * 1000)}.jpg'
                #print(f"Sending frame {filename} with tally {tally}")
                
                response = upload_image(image_bytes, filename, tally)
                finish_time = time.time()
                
                if response and response.get('tally') == str(tally):
                    slope = response.get('slope')
                    intercept = response.get('intercept')
                    if slope is not None and intercept is not None:
                        publisher_node.set_slope_intercept(slope, intercept)
                    #print(f'Slope & Intercept: {slope}, {intercept}')
                    tally += 1
                else:
                    print("Server returned an incorrect or empty response. Retrying...")
            else:
                print("No frame available.")
            
            elapsed_time = time.time() - start_time
            if elapsed_time < min_interval:
                time.sleep(min_interval - elapsed_time)
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")
    finally:
        video_stream.stop()
        rclpy.shutdown()
        print("Video stream stopped.")

if __name__ == '__main__':
    main()

