import cv2 as cv
import time

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: continue
    print(time.thread_time_ns())
    cv.imshow('oog', frame)
    if cv.waitKey(1) == ord('q'): break

cap.release()
cv.destroyAllWindows()
