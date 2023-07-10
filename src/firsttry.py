import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mouse
import cv2 as cv
import numpy as np

model_path = 'models/hand_landmarker.task'


BaseOptions = mp.tasks.BaseOptions(model_asset_path=model_path)
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Create a hand landmarker instance with the video mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions,
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2)
with HandLandmarker.create_from_options(options) as landmarker:
    # The landmarker is initialized. Use it here.
    timestamp = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("couldnt receive frame")
            continue

        if cv.waitKey(1) == ord('q'):
            break
        
        timestamp += cv.CAP_PROP_FPS
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        hand_landmarker_result = landmarker.detect_for_video(mp_image, timestamp)
        cv.imshow('frame', frame)



cap.release()
cv.destroyAllWindows()