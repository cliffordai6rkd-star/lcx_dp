import os
import json
import cv2
import numpy as np
from pupil_apriltags import Detector

# ----------------- 配置区域 -----------------
# 和采集脚本保持一致
save_path = "camera_calibration"      # 采集脚本用的同一个目录
cur_path = os.path.dirname(os.path.abspath(__file__))

data_json_path = os.path.join(cur_path, save_path, "data.json")
output_json_path = os.path.join(cur_path, save_path, "camera_params.json")

# AprilTag 实际尺寸（单位：米）——一定要和采集脚本一致
tag_size = 0.055

# 使用的 AprilTag family，要和采集脚本一致
tag_family = "tag36h11"

# ----------------- 工具函数 -----------------
def load_data(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    return data

def create_apriltag_detector():
    detector = Detector(families=tag_family)
    return detector

# 定义 tag 在自身坐标系下的 3D 角点
def get_tag_object_points(tag_size):
    half = tag_size / 2.0
    # 这里假定 detection.corners 的顺序是:
    # [top-left, top-right, bottom-right, bottom-left]
    objp = np.array([
        [-half,  half, 0.0],   # top-left
        [ half,  half, 0.0],   # top-right
        [ half, -half, 0.0],   # bottom-right
        [-half, -half, 0.0],   # bottom-left
    ], dtype=np.float32)
    return objp

# ----------------- 主流程 -----------------
def main():
    if not os.path.exists(data_json_path):
        raise FileNotFoundError(f"找不到 data.json: {data_json_path}")

    data_all = load_data(data_json_path)
    records = data_all.get("data", [])
    if len(records) == 0:
        raise RuntimeError("data.json 里的 data 为空，没有任何图像记录。")

    # 如果想看原始 RealSense 内参，可以从这里拿
    intrinsics_rs = data_all.get("intrinsics", None)

    detector = create_apriltag_detector()
    objp_tag = get_tag_object_points(tag_size)

    objpoints = []         # 每张图对应的 3D 点（这里 tag 4 个角点）
    imgpoints = []         # 每张图对应的 2D 像素点
    used_image_paths = []  # 实际用于标定的图像路径（可能有些图没检测到 tag 被跳过）

    image_size = None

    # --------- 遍历所有采集到的图片，重新检测 tag ----------
    for rec in records:
        img_path = rec["raw_image"]
        img = cv2.imread(img_path)
        if img is None:
            print(f"[警告] 无法读取图片: {img_path}，跳过。")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 这里只需要角点，不需要估计位姿，所以 estimate_tag_pose=False
        detections = detector.detect(
            gray,
            estimate_tag_pose=False
        )

        if len(detections) == 0:
            print(f"[提示] 图片中未检测到 AprilTag: {img_path}，跳过。")
            continue

        # 如果有多个 tag，只用第一个；你也可以按需过滤 ID
        det = detections[0]
        corners = det.corners.astype(np.float32)  # (4,2)

        if corners.shape != (4, 2):
            print(f"[警告] 检测到的角点数量异常: {img_path}，跳过。")
            continue

        # 记录这张图的 2D-3D 对应
        objpoints.append(objp_tag)
        imgpoints.append(corners)
        used_image_paths.append(img_path)

        if image_size is None:
            h, w = gray.shape
            image_size = (w, h)

    if len(objpoints) < 3:
        raise RuntimeError(
            f"有效标定图像太少（{len(objpoints)} 张），建议至少 5~10 张。"
        )

    print(f"用于标定的有效图片数量: {len(objpoints)}")
    print(f"图像尺寸: {image_size}")

    # --------- 使用 OpenCV 标定相机 ----------
    objpoints = [np.asarray(op, dtype=np.float32) for op in objpoints]
    imgpoints = [np.asarray(ip, dtype=np.float32) for ip in imgpoints]

    # 如果想用 RealSense 的内参作为初值，可以在这里构造 K_init 并加标志位
    camera_matrix_init = None
    dist_coeffs_init = None
    flags = 0

    # 如果 data.json 里面有 RealSense 内参，可以作为初值
    if intrinsics_rs is not None:
        fx = float(intrinsics_rs["fx"])
        fy = float(intrinsics_rs["fy"])
        cx = float(intrinsics_rs["ppx"])
        cy = float(intrinsics_rs["ppy"])

        camera_matrix_init = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # 简单初始化畸变为 0
        dist_coeffs_init = np.zeros((5, 1), dtype=np.float64)

        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        print("使用 RealSense 提供的内参作为初值标定。")

    print("开始标定相机（calibrateCamera）...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        camera_matrix_init,
        dist_coeffs_init,
        flags=flags
    )

    print(f"标定完成，重投影误差: {ret}")
    print("相机内参 K:")
    print(camera_matrix)
    print("畸变系数 dist:")
    print(dist_coeffs.ravel())

    # --------- 计算每张图的外参（tag -> camera 的 4x4 齐次矩阵） ----------
    extrinsics = []
    for img_path, rvec, tvec in zip(used_image_paths, rvecs, tvecs):
        R, _ = cv2.Rodrigues(rvec)      # 3x3
        t = tvec.reshape(3)             # 3,

        # 齐次矩阵：X_cam = R * X_tag + t
        T_tag_to_cam = np.eye(4, dtype=float)
        T_tag_to_cam[:3, :3] = R
        T_tag_to_cam[:3, 3] = t

        extrinsics.append({
            "image": img_path,
            "rvec": rvec.reshape(-1).tolist(),
            "tvec": t.tolist(),
            # 约定：X_cam = R * X_tag + t
            "T_tag_to_cam": T_tag_to_cam.tolist()
        })

    # --------- 组织要写入 JSON 的结果 ----------
    output = {
        "image_size": {
            "width": int(image_size[0]),
            "height": int(image_size[1])
        },
        "reprojection_error": float(ret),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
        "extrinsics": extrinsics,
    }

    # 顺便把 RealSense 原始内参也存进去方便对比（如果有）
    if intrinsics_rs is not None:
        output["realsense_intrinsics_raw"] = intrinsics_rs

    # 写入到指定 json
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"标定结果已保存到: {output_json_path}")


if __name__ == "__main__":
    main()
