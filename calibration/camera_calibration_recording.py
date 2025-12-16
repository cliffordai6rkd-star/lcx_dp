import pyrealsense2 as rs
import numpy as np
import cv2
import json
import os
from pupil_apriltags import Detector

# ----------------- 参数部分 -----------------
serial_number = "333422301209"
save_path = "camera_calibration"
cur_path = os.path.dirname(os.path.abspath(__file__))
image_save_path = os.path.join(cur_path, save_path, "images")
raw_image_save_path = os.path.join(cur_path, save_path, "raw_images")
data_save_path = os.path.join(cur_path, save_path, "data.json")

# AprilTag 实际尺寸（单位：米）
tag_size = 0.055  # 你原来的数值

# 初始化 RealSense 管线
pipeline = rs.pipeline()
config = rs.config()
if serial_number is not None:
    config.enable_device(serial_number)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 开始采集
profile = pipeline.start(config)

# 相机内参
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
fx, fy = intrinsics.fx, intrinsics.fy
cx, cy = intrinsics.ppx, intrinsics.ppy
camera_params = [fx, fy, cx, cy]
print(f"Camera Intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

# 初始化 AprilTag 检测器
detector = Detector(families="tag36h11")

# 创建显示窗口
cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

# ----------------- 采集相关变量 -----------------

data = []               # 用来存每一张图片的标定数据
image_counter = 0       # 图片计数

def save_to_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

os.makedirs(image_save_path, exist_ok=True)
os.makedirs(raw_image_save_path, exist_ok=True)
try:
    while True:
        # 获取一帧 color 图像
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        raw_image = color_image.copy()
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # 每帧检测 AprilTag
        detections = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=tag_size
        )

        # 用来记录当前帧最后一个 tag 的位姿
        homogeneous_matrix = None

        # 画检测结果
        for detection in detections:
            # 画边框
            for idx in range(len(detection.corners)):
                pt1 = tuple(map(int, detection.corners[idx]))
                pt2 = tuple(map(int, detection.corners[(idx + 1) % 4]))
                cv2.line(color_image, pt1, pt2, (0, 255, 0), 2)

            # 中心点
            center = tuple(map(int, detection.center))
            cv2.circle(color_image, center, 5, (0, 0, 255), -1)

            # ID
            cv2.putText(color_image, f"ID: {detection.tag_id}",
                        (center[0] - 10, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # 位姿
            t = detection.pose_t.flatten()    # 3x1 平移
            R = detection.pose_R              # 3x3 旋转

            pose_text = f"XYZ: ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}) m"
            cv2.putText(color_image, pose_text, (center[0] - 50, center[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # 画坐标轴（X=红，Y=绿，Z=蓝）
            axis_length = 0.2
            axes = np.array([
                [axis_length, 0, 0],
                [0, axis_length, 0],
                [0, 0, axis_length]
            ])
            axes_img_pts = []
            for i in range(3):
                axis_end = t + R @ axes[i]
                x = int(fx * axis_end[0] / axis_end[2] + cx)
                y = int(fy * axis_end[1] / axis_end[2] + cy)
                axes_img_pts.append((x, y))

            cv2.line(color_image, center, axes_img_pts[0], (0, 0, 255), 2)   # X
            cv2.line(color_image, center, axes_img_pts[1], (0, 255, 0), 2)   # Y
            cv2.line(color_image, center, axes_img_pts[2], (255, 0, 0), 2)   # Z

            # 相机坐标到 AprilTag 坐标的 4x4 齐次矩阵
            homogeneous_matrix = np.eye(4)
            homogeneous_matrix[:3, :3] = R
            homogeneous_matrix[:3, 3] = t

        # 实时显示
        cv2.imshow('RealSense', color_image)
        key = cv2.waitKey(1) & 0xFF

        # ---------- 按键处理 ----------

        if key == ord('s'):
            # 手动按 s 保存当前画面
            if homogeneous_matrix is None:
                print("当前帧没有检测到 AprilTag，未保存。")
            else:
                image_filename = f'{image_save_path}/{image_counter:04d}.png'
                cv2.imwrite(image_filename, color_image)

                # 保存原始图像
                raw_image_filename = f'{raw_image_save_path}/{image_counter:04d}.png'
                cv2.imwrite(raw_image_filename, raw_image)

                # 这里只保存图像路径和 tag 的位姿（没有机器人矩阵了）
                data.append({
                    'image': image_filename,
                    "raw_image": raw_image_filename,
                    'homogeneous_matrix': homogeneous_matrix.tolist()
                })

                print(f"保存第 {image_counter} 张: {image_filename}")
                image_counter += 1

        elif key == ord('q'):
            # 退出前保存 intrinsics 和采集到的数据
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            intrinsics_data = {
                'width': intrinsics.width,
                'height': intrinsics.height,
                'ppx': intrinsics.ppx,
                'ppy': intrinsics.ppy,
                'fx': intrinsics.fx,
                'fy': intrinsics.fy,
                # 'model': intrinsics.model,
                'coeffs': intrinsics.coeffs
            }
            for key, value in intrinsics_data.items():
                print(f"{key} data {value}, type: {type(value)}")

            save_to_json(data_save_path, {
                'data': data,
                'intrinsics': intrinsics_data
            })
            print("按下 q，数据与内参已保存，退出。")
            break

finally:
    # 善后：关闭相机、窗口
    pipeline.stop()
    cv2.destroyAllWindows()
