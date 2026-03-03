import cv2
import mediapipe as mp
import numpy as np
import time

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
recording = False


def normalize_trajectory(traj, target_length=50):
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


while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))  # resize
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    try:
        result = landmarker.detect(mp_image)
    except Exception as e:
        print("MediaPipe error:", e)
        continue

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            h, w, _ = frame.shape
            x = int(hand_landmarks[8].x * w)
            y = int(hand_landmarks[8].y * h)

            if recording:
                trajectory.append((x, y))
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.putText(frame, "Press R to Record | S to Stop | Q to Quit",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Spell Detector", frame)
    key = cv2.waitKey(1)
    time.sleep(0.01)  # tiny delay

    if key == ord('r'):
        trajectory = []
        recording = True
        print("Recording started")

    if key == ord('s'):
        recording = False
        print("Recording stopped")

        if len(trajectory) > 10:
            processed = normalize_trajectory(trajectory)
            spell_name = input("Enter spell name: ")

            np.save(f"{spell_name}_{len(trajectory)}.npy", processed)
            print("Saved!")
        else:
            print("Too short, try again.")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()