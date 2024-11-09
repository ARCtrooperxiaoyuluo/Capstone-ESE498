from flask import Flask, request
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create upload folder if it doesn't exist

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or 'tally' not in request.form:
        return 'File or tally part missing', 400
    
    file = request.files['file']
    tally = request.form['tally']  # Get tally from the form data
    
    if file.filename == '':
        return 'No selected file', 400

    # Save the file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    # Return the tally number back to the client
    return {'status': 'File saved', 'tally': tally}, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)  # 8000 for local test, 80 for publishing server.

