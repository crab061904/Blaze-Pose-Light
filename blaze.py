import cv2
import mediapipe as mp
import numpy as np
import time
import urllib.request
import os

# 1. Automatically download the BlazePimport cv2
import mediapipe as mp
import numpy as np
import time
import urllib.request
import os

# 1. Automatically download the BlazePose-Lite model file if you don't have it
model_path = 'pose_landmarker_lite.task'
if not os.path.exists(model_path):
    print("Downloading BlazePose-Lite model (this only happens once)...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")

# 2. Setup the new MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def calculate_angle(a, b, c):
    """Calculates the angle at vertex b formed by points a, b, c."""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

# We need a global variable to store the real-time landmark coordinates
latest_result = None

# This function updates our global variable every time the AI detects a skeleton
def update_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

# Configure the options for BlazePose-Lite
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=update_result
)

# 3. Create the landmarker instance
landmarker = PoseLandmarker.create_from_options(options)

# 4. Map out how the joints connect — no face (0-10) or hand fingers (17-22)
POSE_CONNECTIONS = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Arms (shoulder → elbow → wrist only)
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    # Legs
    (23, 25), (24, 26), (25, 27), (26, 28),
    # Feet
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

# Landmarks to skip when drawing dots: face (0-10) and hand fingers (17-22)
SKIP_LANDMARKS = set(range(0, 11)) | set(range(17, 23))

# 5. Start Video Capture
cap = cv2.VideoCapture(0)
print("Starting BlazePose-Lite Stream. Press 'q' to exit.")

start_time = time.time()
last_timestamp_ms = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Flip the image horizontally for a selfie-view display
    frame = cv2.flip(frame, 1)
    
    # Convert colors for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # The Tasks API requires a strict timeline of timestamps
    timestamp_ms = int((time.time() - start_time) * 1000)
    if timestamp_ms <= last_timestamp_ms:
        timestamp_ms = last_timestamp_ms + 1
    last_timestamp_ms = timestamp_ms
    
    # Send the frame to the AI
    landmarker.detect_async(mp_image, timestamp_ms)
    
    # 6. Draw the skeleton using OpenCV if a body is detected
    if latest_result and latest_result.pose_landmarks:
        h, w, _ = frame.shape
        for pose_landmarks in latest_result.pose_landmarks:
            
            # Draw the connection lines (bones)
            for connection in POSE_CONNECTIONS:
                lm_start = pose_landmarks[connection[0]]
                lm_end = pose_landmarks[connection[1]]
                
                # Convert normalized AI coordinates (0.0 to 1.0) to actual pixel locations
                x_start, y_start = int(lm_start.x * w), int(lm_start.y * h)
                x_end, y_end = int(lm_end.x * w), int(lm_end.y * h)
                
                cv2.line(frame, (x_start, y_start), (x_end, y_end), (245, 66, 230), 2)
                
            # Draw the landmark dots (joints), skipping face and hand fingers
            for idx, landmark in enumerate(pose_landmarks):
                if idx in SKIP_LANDMARKS:
                    continue
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 4, (245, 117, 66), -1)

            # Arm angle overlay for both arms
            for shoulder_idx, elbow_idx, wrist_idx in [(11, 13, 15), (12, 14, 16)]:
                shoulder = pose_landmarks[shoulder_idx]
                elbow    = pose_landmarks[elbow_idx]
                wrist    = pose_landmarks[wrist_idx]

                shoulder_px = (int(shoulder.x * w), int(shoulder.y * h))
                elbow_px    = (int(elbow.x * w),    int(elbow.y * h))
                wrist_px    = (int(wrist.x * w),    int(wrist.y * h))

                angle = calculate_angle(shoulder, elbow, wrist)

                cv2.line(frame, shoulder_px, elbow_px, (255, 255, 255), 4)
                cv2.line(frame, elbow_px, wrist_px, (255, 255, 255), 4)

                cv2.circle(frame, shoulder_px, 8, (0, 255, 0), -1)
                cv2.circle(frame, elbow_px,    8, (0, 255, 0), -1)
                cv2.circle(frame, wrist_px,    8, (0, 255, 0), -1)

                # Place text inside the angle using the bisector direction
                e = np.array(elbow_px, dtype=float)
                s_dir = np.array(shoulder_px, dtype=float) - e
                w_dir = np.array(wrist_px, dtype=float) - e
                s_norm = s_dir / (np.linalg.norm(s_dir) + 1e-6)
                w_norm = w_dir / (np.linalg.norm(w_dir) + 1e-6)
                bisector = s_norm + w_norm
                bisector = bisector / (np.linalg.norm(bisector) + 1e-6)
                text_pos = (int(e[0] + bisector[0] * 40), int(e[1] + bisector[1] * 40))

                cv2.putText(frame, str(int(angle)),
                            text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('BlazePose-Lite Tracker (Press Q to Quit)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
landmarker.close()ose-Lite model file if you don't have it
model_path = 'pose_landmarker_lite.task'
if not os.path.exists(model_path):
    print("Downloading BlazePose-Lite model (this only happens once)...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")

# 2. Setup the new MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def calculate_angle(a, b, c):
    """Calculates the angle at vertex b formed by points a, b, c."""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

# We need a global variable to store the real-time landmark coordinates
latest_result = None

# This function updates our global variable every time the AI detects a skeleton
def update_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

# Configure the options for BlazePose-Lite
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=update_result
)

# 3. Create the landmarker instance
landmarker = PoseLandmarker.create_from_options(options)

# 4. Map out how the joints connect — no face (0-10) or hand fingers (17-22)
POSE_CONNECTIONS = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Arms (shoulder → elbow → wrist only)
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    # Legs
    (23, 25), (24, 26), (25, 27), (26, 28),
    # Feet
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

# Landmarks to skip when drawing dots: face (0-10) and hand fingers (17-22)
SKIP_LANDMARKS = set(range(0, 11)) | set(range(17, 23))

# 5. Start Video Capture
cap = cv2.VideoCapture(0)
print("Starting BlazePose-Lite Stream. Press 'q' to exit.")

start_time = time.time()
last_timestamp_ms = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Flip the image horizontally for a selfie-view display
    frame = cv2.flip(frame, 1)
    
    # Convert colors for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # The Tasks API requires a strict timeline of timestamps
    timestamp_ms = int((time.time() - start_time) * 1000)
    if timestamp_ms <= last_timestamp_ms:
        timestamp_ms = last_timestamp_ms + 1
    last_timestamp_ms = timestamp_ms
    
    # Send the frame to the AI
    landmarker.detect_async(mp_image, timestamp_ms)
    
    # 6. Draw the skeleton using OpenCV if a body is detected
    if latest_result and latest_result.pose_landmarks:
        h, w, _ = frame.shape
        for pose_landmarks in latest_result.pose_landmarks:
            
            # Draw the connection lines (bones)
            for connection in POSE_CONNECTIONS:
                lm_start = pose_landmarks[connection[0]]
                lm_end = pose_landmarks[connection[1]]
                
                # Convert normalized AI coordinates (0.0 to 1.0) to actual pixel locations
                x_start, y_start = int(lm_start.x * w), int(lm_start.y * h)
                x_end, y_end = int(lm_end.x * w), int(lm_end.y * h)
                
                cv2.line(frame, (x_start, y_start), (x_end, y_end), (245, 66, 230), 2)
                
            # Draw the landmark dots (joints), skipping face and hand fingers
            for idx, landmark in enumerate(pose_landmarks):
                if idx in SKIP_LANDMARKS:
                    continue
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 4, (245, 117, 66), -1)

            # Arm angle overlay for both arms
            for shoulder_idx, elbow_idx, wrist_idx in [(11, 13, 15), (12, 14, 16)]:
                shoulder = pose_landmarks[shoulder_idx]
                elbow    = pose_landmarks[elbow_idx]
                wrist    = pose_landmarks[wrist_idx]

                shoulder_px = (int(shoulder.x * w), int(shoulder.y * h))
                elbow_px    = (int(elbow.x * w),    int(elbow.y * h))
                wrist_px    = (int(wrist.x * w),    int(wrist.y * h))

                angle = calculate_angle(shoulder, elbow, wrist)

                cv2.line(frame, shoulder_px, elbow_px, (255, 255, 255), 4)
                cv2.line(frame, elbow_px, wrist_px, (255, 255, 255), 4)

                cv2.circle(frame, shoulder_px, 8, (0, 255, 0), -1)
                cv2.circle(frame, elbow_px,    8, (0, 255, 0), -1)
                cv2.circle(frame, wrist_px,    8, (0, 255, 0), -1)

                cv2.putText(frame, str(int(angle)),
                            (elbow_px[0] + 20, elbow_px[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('BlazePose-Lite Tracker (Press Q to Quit)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
landmarker.close()
