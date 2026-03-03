import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands using new API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
frame_id = 0

trajectory = []
recording = False

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect_for_video(mp_image, frame_id)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            h, w, _ = frame.shape
            x = int(hand_landmarks[8].x * w)  # index fingertip
            y = int(hand_landmarks[8].y * h)

            if recording:
                trajectory.append((x, y))
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.putText(frame, "Press R to Record | S to Stop", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Spell Detector", frame)

    key = cv2.waitKey(1)

    if key == ord('r'):
        trajectory = []
        recording = True
        print("Recording started")

    if key == ord('s'):
        recording = False
        print("Recording stopped")
        print("Trajectory length:", len(trajectory))

    if key == ord('q'):
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()