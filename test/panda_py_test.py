from panda_py import Panda
import numpy as np
import cv2

def move_to_start(arm: Panda):
    arm.move_to_start()
    
def loop_pose(arm: Panda, test_times:int = 25):
    for _ in range(test_times):
        ori = arm.get_orientation()
        posi = arm.get_position()
        pose = np.hstack((posi, ori))
        print(f'pose: {pose}')
        assert  pose.shape[0] == 7, f"pose should be 7D, but got {pose.shape[0]}D"


def preview_video_devices(start_idx: int = 0, end_idx: int = 60):
    window_name = "video_preview"

    try:
        for device_idx in range(start_idx, end_idx + 1):
            device_path = f"/dev/video{device_idx}"
            cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)

            if not cap.isOpened():
                print(f"skip {device_path}: cannot open")
                cap.release()
                continue

            print(f"showing {device_path}, press q for next camera, Esc to exit")

            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"{device_path}: failed to read frame, switch to next camera")
                    break

                cv2.putText(
                    frame,
                    f"{device_path} | q: next | esc: exit",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == 27:
                    return

            cap.release()
    finally:
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    # preview_video_devices()
    arm = Panda("192.168.1.208")
    # move_to_start(arm)
    loop_pose(arm)
    
    
