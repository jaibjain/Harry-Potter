import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
trajectory = []
recording = False
last_move_time = 0
movement_threshold = 60       # pixels
stop_delay = 1.0             # seconds before casting
target_length = 50
# -----------------------------
# Load trained model
# -----------------------------
clf = joblib.load("spell_classifier.joblib")

# -----------------------------
# Normalize function (MUST match training)
# -----------------------------
def normalize_trajectory(traj, target_length=50):

    if len(traj) < 10:
        return None

    traj = np.array(traj)

    # Center
    traj = traj - np.mean(traj, axis=0)

    # Scale
    max_val = np.max(np.abs(traj))
    if max_val != 0:
        traj = traj / max_val

    # Resize to fixed length
    indices = np.linspace(0, len(traj)-1, target_length).astype(int)
    traj = traj[indices]

    return traj.flatten()

# -----------------------------
# MediaPipe Setup (New API)
# -----------------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

trajectory = []
recording = True
predicted_spell = ""

print("Press R to record | S to stop and predict | Q to quit")

# -----------------------------
# Main Loop
# -----------------------------
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    current_time = time.time()

    result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            h, w, _ = frame.shape
            x = int(hand_landmarks[8].x * w)
            y = int(hand_landmarks[8].y * h)

            if recording:
                trajectory.append((x, y))
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    if len(trajectory) > 0:
            prev_x, prev_y = trajectory[-1]
            movement = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
    else:
            movement = 0

    # Show predicted spell
    cv2.putText(frame, f"Spell: {predicted_spell}",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 255), 2)

    cv2.putText(frame, "R=Record | S=Predict | Q=Quit",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    cv2.imshow("Spell Detector", frame)

    key = cv2.waitKey(1)

    if movement > movement_threshold:
        last_move_time = time.time()
        trajectory = []
        recording = True
        predicted_spell = ""
        print("Recording...")

    if (current_time - last_move_time > stop_delay) and recording:
        recording = False
        print("Stopped. Predicting...")

        processed = normalize_trajectory(trajectory)

        if processed is not None:
            prediction = clf.predict([processed])[0]
            predicted_spell = prediction
            print("Predicted:", prediction)
        else:
            predicted_spell = "Too Short"
            print("Gesture too short")

    if key == ord('q'):
        break

    time.sleep(0.01)

cap.release()
cv2.destroyAllWindows()