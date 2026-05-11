import cv2
import mediapipe as mp
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

# 4. Map out how the 33 joints connect to draw the skeleton
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), 
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), 
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23), 
    (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), 
    (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]

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
                
            # Draw the landmark dots (joints)
            for landmark in pose_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 4, (245, 117, 66), -1)

    cv2.imshow('BlazePose-Lite Tracker (Modern API)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
landmarker.close()