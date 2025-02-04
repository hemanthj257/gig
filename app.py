from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import os
from scipy.stats import entropy
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from model import predict_diabetes
from arduino_controller import ArduinoController

# Remove dlib import and use OpenCV's face detection instead
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

app = Flask(__name__)
arduino = ArduinoController()

# Create upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize MobileNetV2 for skin detection
skin_model = MobileNetV2(weights='imagenet')

# Load the diabetes model and scaler
try:
    diabetes_model = joblib.load('diabetes_model.joblib')
    diabetes_scaler = joblib.load('diabetes_scaler.joblib')
except:
    # Train and save the model if not exists
    from sklearn.linear_model import LogisticRegression
    data = pd.read_csv("diabetes.csv")
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]
    
    diabetes_scaler = StandardScaler()
    X_scaled = diabetes_scaler.fit_transform(X)
    
    diabetes_model = LogisticRegression()
    diabetes_model.fit(X_scaled, y)
    
    joblib.dump(diabetes_model, 'diabetes_model.joblib')
    joblib.dump(diabetes_scaler, 'diabetes_scaler.joblib')

def detect_eyes(img):
    # Load OpenCV's pre-trained eye detector
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect eyes
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    if len(eyes) == 0:
        return "No eyes detected"
        
    # Check for redness in eye regions
    for (x,y,w,h) in eyes:
        roi = img[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Red color range in HSV
        red_mask = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
        red_ratio = np.sum(red_mask > 0) / (w * h)
        if red_ratio > 0.1:
            return "Conjunctivitis (Pink Eye) Detected"
    return "No Conjunctivitis Detected"

def detect_jaundice(img):
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Adjusted yellow color range for better detection
    lower_yellow = np.array([15, 50, 50])
    upper_yellow = np.array([35, 255, 255])
    
    # Create mask for yellow color
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Calculate percentage of yellow pixels
    yellow_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
    
    # Adjusted threshold (lowered from 0.15 to 0.08)
    return yellow_ratio > 0.08

def detect_acne(img):
    """Detect acne based on color and spot characteristics"""
    # Convert to appropriate color spaces
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range for reddish/inflammatory spots (narrower range)
    lower_red = np.array([0, 100, 70])  # Increased saturation threshold
    upper_red = np.array([10, 255, 255])
    
    # Create mask for red spots
    red_mask = cv2.inRange(img_hsv, lower_red, upper_red)
    
    # Apply more aggressive noise reduction
    kernel = np.ones((3,3), np.uint8)  # Smaller kernel
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours more strictly
    acne_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # More strict size and shape filtering
            if 30 < area < 300 and circularity > 0.5:  # Smaller area range and check for circular shape
                acne_count += 1
    
    print(f"Detected {acne_count} potential acne spots")  # Debug info
    
    # More conservative thresholds for classification
    if acne_count > 4:
        return "Moderate to Severe Acne Detected"
    elif acne_count > 1:
        return "Mild Acne Detected"
    else:
        return "No Significant Acne Detected"

def detect_cyanosis(img):
    """Detect bluish discoloration in lips and nail beds"""
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define blue/purple color range for cyanosis
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    # Create mask for blue regions
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Apply noise reduction
    kernel = np.ones((3,3), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    
    # Calculate blue ratio
    blue_ratio = np.sum(blue_mask > 0) / (blue_mask.shape[0] * blue_mask.shape[1])
    
    # Debug info
    print(f"Blue ratio: {blue_ratio:.4f}")
    
    return "Possible Cyanosis Detected" if blue_ratio > 0.05 else "No Cyanosis Detected"

def detect_facial_drooping(img):
    """Detect facial asymmetry using basic OpenCV"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return "No face detected"
    
    # Get the first face
    x, y, w, h = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    
    # Split face in half
    height, width = face_roi.shape
    left_side = face_roi[:, :width//2]
    right_side = face_roi[:, width//2:]
    right_side = cv2.flip(right_side, 1)  # Flip for comparison
    
    # Compare both sides
    diff = cv2.absdiff(left_side, right_side)
    asymmetry = np.mean(diff)
    
    print(f"Asymmetry score: {asymmetry}")
    
    return "Possible Facial Drooping Detected" if asymmetry > 30 else "No Facial Drooping Detected"

def handle_detection_result(result):
    # Check if condition is detected
    is_detected = (
        isinstance(result, dict) and result.get('result', '').lower().find('has diabetes') != -1
    ) or (
        isinstance(result, str) and (
            result.lower().find('detected') != -1 or
            result.lower().find('moderate') != -1 or
            result.lower().find('mild') != -1
        )
    )
    
    # Control Arduino LEDs
    arduino.set_led(is_detected)
    return result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame data'})
    
    frame_data = request.files['frame'].read()
    detection_type = request.form.get('type', 'eye')
    
    nparr = np.frombuffer(frame_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if detection_type == 'eye':
        result = detect_eyes(img)
    
    elif detection_type == 'skin':
        result = detect_acne(img)
    
    elif detection_type == 'jaundice':
        result = "Jaundice Detected" if detect_jaundice(img) else "No Jaundice Detected"
    
    elif detection_type == 'cyanosis':
        result = detect_cyanosis(img)
    
    elif detection_type == 'drooping':
        result = detect_facial_drooping(img)
    
    result = handle_detection_result(result)
    return jsonify({'result': result})

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    
    file = request.files['image']
    detection_type = request.form.get('type', 'skin')
    
    # Read image
    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Initialize result
    result = "Unknown detection type"
    
    try:
        if detection_type == 'eye':
            result = detect_eyes(img)
        elif detection_type == 'skin':
            result = detect_acne(img)
        elif detection_type == 'jaundice':
            result = "Jaundice Detected" if detect_jaundice(img) else "No Jaundice Detected"
        elif detection_type == 'cyanosis':
            result = detect_cyanosis(img)
        elif detection_type == 'drooping':
            result = detect_facial_drooping(img)
        else:
            result = f"Invalid detection type: {detection_type}"
            
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'})
    
    result = handle_detection_result(result)
    return jsonify({'result': result})

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes_route():
    try:
        data = request.json
        prediction, probability = predict_diabetes(
            data['gender'],
            data['age'],
            data['hypertension'],
            data['heart_disease'],
            data['smoking'],
            data['bmi'],
            data['HbA1c_level'],
            data['blood_glucose']
        )
        
        result = {
            'result': "Patient has diabetes" if prediction == 1 else "Patient doesn't have diabetes",
            'probability': f"{probability*100:.2f}%"
        }
        result = handle_detection_result(result)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Cleanup Arduino connection when app exits
import atexit
atexit.register(lambda: arduino.close())

if __name__ == '__main__':
    app.run(debug=True)
