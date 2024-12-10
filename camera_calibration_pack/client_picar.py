import cv2
import time
import requests
import io
# Global variable for storing the latest frame
latest_frame = None
import threading

class VideoStream:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src, cv2.CAP_V4L)
        if not self.capture.isOpened():
            raise ValueError(f"Unable to open video source {src}")
        
        # Set camera properties if needed
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Initialize the latest frame
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
            # Sleep briefly to reduce CPU usage
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
    # Prepare the files and data payload
    files = {'file': (filename, image_bytes, 'image/jpeg')}  # Specify the correct MIME type
    data = {'tally': str(tally)}
    try:
        response = requests.post(url, files=files, data=data, timeout=10)  # Set a timeout
        response.raise_for_status()  # Raise an exception for HTTP errors
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    
    try:
        # Attempt to parse JSON response
        return response.json()
    except requests.exceptions.JSONDecodeError:
        # If JSON decoding fails, return the raw text response
        print("Response is not in JSON format. Here’s the raw response:")
        print(response.text)
        return None

def main():
    tally = 1
    min_interval = 0.1  # Minimum time between requests in seconds (10 requests per second)
    
    # Initialize the video stream
    try:
        video_stream = VideoStream(src=0)  # Change src if needed
    except ValueError as e:
        print(e)
        return
    
    print("Video stream started.")
    
    try:
        while True:
            start_time = time.time()  # Record the start time for each request
            
            # Get the latest frame
            frame = video_stream.read()
            
            if frame is not None:
                # Encode the frame to JPEG format in memory
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                success, encoded_image = cv2.imencode('.jpg', frame)
                if not success:
                    print("Failed to encode frame.")
                    continue  # Skip sending this frame
                
                # Convert the encoded image to bytes
                image_bytes = encoded_image.tobytes()
                
                # Generate a unique filename using a timestamp
                filename = f'frame_{int(time.time() * 1000)}.jpg'
                
                print(f"Sending frame {filename} with tally {tally}")
                
                # Send the image with the current tally
                response = upload_image(image_bytes, filename, tally)
                finish_time = time.time()
                
                if response and response.get('tally') == str(tally):
                    print(f"Server processed tally {tally}. Ready to send next image.")
                    print('Slope& Intercept: '+str(response.get('slope'))+str(response.get('intercept')))
                    tally += 1  # Increment tally for the next image
                else:
                    print("Server returned an incorrect or empty response. Retrying...")
            else:
                print("No frame available.")
            
            # Enforce a delay of at least `min_interval` seconds between uploads
            elapsed_time = time.time() - start_time
            if elapsed_time < min_interval:
                time.sleep(min_interval - elapsed_time)
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting...")
    finally:
        video_stream.stop()
        print("Video stream stopped.")

if __name__ == '__main__':
    main()
