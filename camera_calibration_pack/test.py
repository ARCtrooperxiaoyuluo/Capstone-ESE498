import cv2
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

# Global variable for storing the latest frame
latest_frame = None

def capture_frames():
    global latest_frame
    
    # Open camera
    cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L)
    
    # Set dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Define the codec and create VideoWriter object
    # 'XVID' is a popular codec, output file will be saved as an AVI file
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_offset_1.avi', fourcc, 50.0, (1280, 720))  # 50.0 is the frame rate

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            latest_frame = frame

            # Write the frame to the video file
            out.write(frame)

        time.sleep(0.02)  # Reduced sleep time for faster frame capture

    # Release everything when done
    cap.release()
    out.release()

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

# Start the frame capturing in a separate thread
capture_thread = Thread(target=capture_frames)
capture_thread.daemon = True  # Ensures the thread will close when the main program exits
capture_thread.start()

# Start the HTTP server
server_address = ('', 8000)
httpd = HTTPServer(server_address, FrameHandler)
print("Serving on port 8000...")
httpd.serve_forever()
