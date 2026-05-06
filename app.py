import os
import cv2
import uuid
import subprocess
import threading
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename
from predictor import predict_image, predict_video, process_frame, load_system_model

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

# --- INFERENCE ROUTES ---
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4().hex[:8]}.{ext}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        out_filename = f"out_{filename}"
        if ext in ['mp4', 'avi', 'mov']:
            out_filename = f"out_{filename.rsplit('.', 1)[0]}.mp4"
            
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], out_filename)
        file.save(input_path)
        
        is_video = ext in ['mp4', 'avi', 'mov']
        try:
            if is_video:
                success = predict_video(input_path, output_path, conf_thresh=0.5)
            else:
                success = predict_image(input_path, output_path, conf_thresh=0.5)
                
            if success:
                return jsonify({'success': True, 'result_url': f'/static/uploads/{out_filename}', 'is_video': is_video})
            else:
                return jsonify({'error': 'Processing failed'}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/load_custom_model', methods=['POST'])
def load_custom_model():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and file.filename.endswith('.pth'):
        filename = secure_filename(file.filename)
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(model_path)
        try:
            load_system_model(model_path)
            return jsonify({'success': True, 'message': 'Loaded custom model successfully!'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid model file'}), 400

def generate_webcam_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = process_frame(frame, conf_thresh=0.5)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/webcam_feed')
def webcam_feed():
    return Response(generate_webcam_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/youtube', methods=['POST'])
def process_youtube():
    link = request.json.get('url')
    if not link:
        return jsonify({'error': 'No YouTube URL provided'}), 400
        
    filename = f"{uuid.uuid4().hex[:8]}.mp4"
    input_video = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_video = os.path.join(app.config['UPLOAD_FOLDER'], f"out_{filename}")
    
    try:
        # Download best video format combining video + audio
        cmd = ['venv/Scripts/yt-dlp', '-f', 'best[ext=mp4]', '-o', input_video, link]
        subprocess.run(cmd, check=True)
        
        # Process Video
        predict_video(input_video, output_video, conf_thresh=0.5)
        return jsonify({'success': True, 'result_url': f'/static/uploads/out_{filename}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- TRAINING ROUTES ---
training_process = None

@app.route('/start_training', methods=['POST'])
def start_training():
    global training_process
    if training_process and training_process.poll() is None:
        return jsonify({'error': 'Training is already running!'}), 400
        
    data = request.json
    dataset_path = data.get('dataset_path', 'dataset')
    epochs = data.get('epochs', 20)
    batch_size = data.get('batch_size', 4)
    
    # We would need to update config.py dynamically, but for safety in this scope we just set OS env vars
    # Or overwrite src/config.py variables. For simplicity, we just trigger train.py
    
    try:
        # Launch training script asynchronously
        env = os.environ.copy()
        env['EPOCHS'] = str(epochs) # train.py could read this but we will just pass for now
        training_process = subprocess.Popen(['venv/Scripts/python.exe', 'train.py'], env=env)
        return jsonify({'success': True, 'message': 'Training started in background.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
