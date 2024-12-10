import cv2
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

# Global variables
latest_frame = None
capture_image = False  # Global flag to trigger image capture
image_counter = 0  # Counter to generate unique image names

def capture_frames():
    global latest_frame, capture_image, image_counter
    
    # Open camera
    cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L)
    
    # Set dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            latest_frame = frame
            
            # If capture_image flag is set, save the image with a unique name
            if capture_image:
                image_name = f'captured_image_{image_counter}.jpg'
                cv2.imwrite(image_name, frame)
                print(f"Image captured and saved as '{image_name}'")
                # Increment the counter for the next image
                image_counter += 1
                # Reset the flag
                capture_image = False

        time.sleep(0.02)  # Reduced sleep time for faster frame capture

    # Release the camera when done
    cap.release()

class FrameHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()

            while True:
                if latest_frame is not None:
                    # Encode the latest frame as JPEG
                    ret, jpeg = cv2.imencode('.jpg', latest_frame)
                    if ret:
                        # Write the boundary, content type, and length headers
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', len(jpeg))
                        self.end_headers()
                        # Write the actual image data
                        self.wfile.write(jpeg.tobytes())
                        self.wfile.write(b'\r\n')
                
                # Reduced sleep time in the HTTP server loop
                time.sleep(0.02)

def capture_image_trigger():
    global capture_image
    while True:
        input("Press Enter to capture an image...")
        capture_image = True  # Set the flag to capture an image

# Start the frame capturing in a separate thread
capture_thread = Thread(target=capture_frames)
capture_thread.daemon = True  # Ensures the thread will close when the main program exits
capture_thread.start()

# Start the image capture trigger thread
capture_trigger_thread = Thread(target=capture_image_trigger)
capture_trigger_thread.daemon = True
capture_trigger_thread.start()

# Start the HTTP server
server_address = ('', 8000)
httpd = HTTPServer(server_address, FrameHandler)
print("Serving on port 8000...")
httpd.serve_forever()
