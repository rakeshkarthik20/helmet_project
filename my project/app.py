from flask import Flask, render_template, request, send_from_directory
import cv2
import torch
import numpy as np
import os
import time

app = Flask(__name__)

# Ensure absolute path for 'uploads' folder
UPLOAD_FOLDER = os.path.abspath("uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Ensure absolute path for 'results' folder to store processed videos
RESULTS_FOLDER = os.path.abspath("static/results")
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

# Load YOLOv5 model
model_path = "best.pt"  # Changed to use relative path
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if model is None:
        return "Model failed to load. Check console for details.", 500
        
    if "file" not in request.files:
        return "No file uploaded", 400
    
    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400
    
    # Generate unique filename to prevent overwriting
    timestamp = int(time.time())
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Process the video and get the output path
    try:
        output_path = process_video(file_path, filename)
        return render_template("results.html", video_path=output_path)
    except Exception as e:
        return f"Error processing video: {str(e)}", 500

def process_video(video_path, filename):
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create output video file
    output_filename = f"processed_{filename}"
    output_path = os.path.join(RESULTS_FOLDER, output_filename)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    max_frames = 300  # Limit processing to 300 frames
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect helmets
        results = model(frame)
        
        # Render the results on the frame
        rendered_frame = np.squeeze(results.render())
        
        # Write the frame to output video
        out.write(rendered_frame)
        
        frame_count += 1
    
    # Release resources
    cap.release()
    out.release()
    
    # Return relative path to the video that can be accessed from the web
    return f"results/{output_filename}"

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == "__main__":
    app.run(debug=True)