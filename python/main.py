import cv2
from kinect_bridge import KinectBridge
import traceback
import numpy as np

def main():
    try:
        print("Initializing Kinect...")
        kinect = KinectBridge()
        print("Kinect initialized successfully")
        
        while True:
            try:
                # Get frames from Kinect
                frames = kinect.get_frames()
                # bgr_frame = frames['bgr']
                depth_frame = frames['depth']
                ir_frame = frames['ir']
                
                ir_frame = ir_frame / 256.0
                ir_frame = ir_frame.astype(np.uint8)

                cv2.imshow('Depth', depth_frame / 4500.0)  # Normalize depth for visualization
                cv2.imshow('IR', ir_frame)


                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except RuntimeError as e:
                print(f"Frame capture error: {e}")
                print("Stack trace:")
                traceback.print_exc()
                continue
                
    except Exception as e:
        print(f"Error: {e}")
        print("Stack trace:")
        traceback.print_exc()
        
    finally:
        cv2.destroyAllWindows()
        print("Cleaning up...")

if __name__ == "__main__":
    main()