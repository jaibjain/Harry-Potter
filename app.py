from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager
import numpy as np
import joblib
import mediapipe as mp
import cv2
import time
import io
import base64
from typing import Optional

# -----------------------------
# Global State
# -----------------------------
state = {
    "trajectory": [],
    "recording": False,
    "last_move_time": 0.0,
    "predicted_spell": "",
}

MOVEMENT_THRESHOLD = 10     # pixels
STOP_DELAY = 1.0            # seconds
TARGET_LENGTH = 50

clf = None
landmarker = None


# -----------------------------
# Lifespan: load model + mediapipe
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global clf, landmarker

    # Load classifier
    try:
        clf = joblib.load("spell_classifier.joblib")
        print("✅ Spell classifier loaded.")
    except FileNotFoundError:
        print("⚠️  spell_classifier.joblib not found. /predict will return errors.")

    # Load MediaPipe hand landmarker
    try:
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=1,
        )
        landmarker = HandLandmarker.create_from_options(options)
        print("✅ MediaPipe hand landmarker loaded.")
    except Exception as e:
        print(f"⚠️  Could not load MediaPipe landmarker: {e}")

    yield

    # Cleanup
    if landmarker:
        landmarker.close()


app = FastAPI(
    title="Spell Gesture API",
    description="Detects hand gesture spells from webcam frames using MediaPipe + ML.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static", html=True), name="static")


# -----------------------------
# Schemas
# -----------------------------
class TrajectoryPoint(BaseModel):
    x: float
    y: float


class TrajectoryPayload(BaseModel):
    points: list[TrajectoryPoint]


class PredictResponse(BaseModel):
    spell: str
    status: str          # "predicted" | "too_short" | "no_model"


class FrameResponse(BaseModel):
    spell: str
    status: str
    recording: bool
    trajectory_length: int
    index_finger_x: Optional[int] = None
    index_finger_y: Optional[int] = None


class StateResponse(BaseModel):
    recording: bool
    trajectory_length: int
    last_move_time: float
    predicted_spell: str


# -----------------------------
# Utilities
# -----------------------------
def normalize_trajectory(traj: list, target_length: int = TARGET_LENGTH) -> Optional[np.ndarray]:
    """Normalize a raw trajectory to a fixed-length feature vector."""
    if len(traj) < 10:
        return None

    traj = np.array(traj)
    traj = traj - np.mean(traj, axis=0)

    max_val = np.max(np.abs(traj))
    if max_val != 0:
        traj = traj / max_val

    indices = np.linspace(0, len(traj) - 1, target_length).astype(int)
    traj = traj[indices]

    return traj.flatten()


def run_prediction(trajectory: list) -> PredictResponse:
    """Run spell prediction on a raw trajectory."""
    if clf is None:
        return PredictResponse(spell="", status="no_model")

    processed = normalize_trajectory(trajectory)
    if processed is None:
        return PredictResponse(spell="Too Short", status="too_short")

    prediction = clf.predict([processed])[0]
    return PredictResponse(spell=str(prediction), status="predicted")


# -----------------------------
# Routes
# -----------------------------
@app.get("/", tags=["Health"])
def root():
    return {"message": "Spell Gesture API is running 🪄"}


@app.get("/health", tags=["Health"])
def health():
    return {
        "classifier_loaded": clf is not None,
        "landmarker_loaded": landmarker is not None,
    }


# ------------------------------------------------------------------
# POST /process-frame
# Accepts a raw image (JPEG/PNG) as multipart upload.
# Detects the index-finger tip, updates trajectory state, and
# returns the current spell prediction if a gesture just completed.
# ------------------------------------------------------------------
@app.post("/process-frame", response_model=FrameResponse, tags=["Detection"])
async def process_frame(file: UploadFile = File(...)):
    """
    Upload a single webcam frame (JPEG/PNG).
    The server tracks the index-finger tip trajectory and auto-predicts
    when the hand stops moving.
    """
    if landmarker is None:
        raise HTTPException(status_code=503, detail="MediaPipe landmarker not loaded.")

    # Decode image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    frame = cv2.resize(frame, (640, 480))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    current_time = time.time()
    result = landmarker.detect(mp_image)

    finger_x, finger_y = None, None
    movement = 0.0

    hand_detected = bool(result.hand_landmarks)
    print(f"[FRAME] hand={hand_detected}", end="")

    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]
        h, w = frame.shape[:2]
        finger_x = int(hand_landmarks[8].x * w)
        finger_y = int(hand_landmarks[8].y * h)

        if len(state["trajectory"]) > 0:
            prev_x, prev_y = state["trajectory"][-1]
            movement = float(np.sqrt((finger_x - prev_x) ** 2 + (finger_y - prev_y) ** 2))

        print(f" finger=({finger_x},{finger_y}) movement={movement:.1f}", end="")

    print(f" | recording={state['recording']} traj_len={len(state['trajectory'])}")

    # ---- State machine ----
    if finger_x is not None:
        # Start recording as soon as hand is detected
        if not state["recording"]:
            state["recording"] = True
            state["predicted_spell"] = ""
            state["trajectory"] = []

        # Always collect points while hand is visible
        state["trajectory"].append((finger_x, finger_y))

        # If hand is moving, reset the still-timer
        if movement > MOVEMENT_THRESHOLD:
            state["last_move_time"] = current_time

    else:
        # No hand detected — reset timer so we don't predict on nothing
        state["last_move_time"] = current_time

    # Hand has been still long enough — predict
    if state["recording"] and (current_time - state["last_move_time"] > STOP_DELAY):
        state["recording"] = False
        result_pred = run_prediction(state["trajectory"])
        state["predicted_spell"] = result_pred.spell
        state["trajectory"] = []
        print(f"Predicted: {state['predicted_spell']}")

    return FrameResponse(
        spell=state["predicted_spell"],
        status="recording" if state["recording"] else "idle",
        recording=state["recording"],
        trajectory_length=len(state["trajectory"]),
        index_finger_x=finger_x,
        index_finger_y=finger_y,
    )


# ------------------------------------------------------------------
# POST /predict
# Accepts a raw trajectory (list of {x, y} points) and returns the
# predicted spell. Stateless — does not update the server session.
# ------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse, tags=["Detection"])
def predict_from_trajectory(payload: TrajectoryPayload):
    """
    Submit a pre-recorded trajectory and get a spell prediction.
    Useful when gesture capture happens on the client.
    """
    traj = [(p.x, p.y) for p in payload.points]
    return run_prediction(traj)


# ------------------------------------------------------------------
# GET /state  — inspect current server-side tracking state
# ------------------------------------------------------------------
@app.get("/state", response_model=StateResponse, tags=["State"])
def get_state():
    """Return the current gesture-tracking state."""
    return StateResponse(
        recording=state["recording"],
        trajectory_length=len(state["trajectory"]),
        last_move_time=state["last_move_time"],
        predicted_spell=state["predicted_spell"],
    )


# ------------------------------------------------------------------
# POST /reset  — clear the current trajectory
# ------------------------------------------------------------------
@app.post("/reset", tags=["State"])
def reset_state():
    """Reset trajectory and prediction state."""
    state["trajectory"] = []
    state["recording"] = False
    state["last_move_time"] = 0.0
    state["predicted_spell"] = ""
    return {"message": "State reset."}