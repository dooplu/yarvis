import mediapipe as mp
import mouse
import cv2 as cv
import numpy as np
import time

model_path = 'models/hand_landmarker.task'



BaseOptions = mp.tasks.BaseOptions(model_asset_path=model_path)
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Create a hand landmarker instance with the video mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions,
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2)
with HandLandmarker.create_from_options(options) as landmarker:
    # The landmarker is initialized. Use it here.
    start = time.time()
    x = 0
    y = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("couldnt receive frame")
            continue
        cv.flip(frame, 1, frame)
        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = int((time.time() - start) * 1000)
        hand_landmarker_result = landmarker.detect_for_video(frame, timestamp)
        
        try:
            x = hand_landmarker_result.hand_landmarks[0][8].x
            y = hand_landmarker_result.hand_landmarks[0][8].y
        except IndexError:
            pass
        
        mouse.move(x * 1920, y * 1080, True)
        cv.imshow('frame', frame.numpy_view())

        if cv.waitKey(1) == ord('q'):
            break



cap.release()
cv.destroyAllWindows()