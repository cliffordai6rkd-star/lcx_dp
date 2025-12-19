import json
import time
import zmq
import numpy as np
import cv2, threading
import glog as log

class G1UmiClient:
    """
    客户端：
      - 控制：REQ (ctrl_endpoint)
      - 图像：SUB (img_endpoint)
    """

    def __init__(
        self, server_ip,
        ctrl_endpoint: int = 5555,
        img_endpoint: int = 5556,
        require_control=True,
    ):
        self.requre_control = False if ctrl_endpoint is None else require_control
        self.ctrl_endpoint = f"tcp://{server_ip}:{ctrl_endpoint}"
        self.img_endpoint = f"tcp://{server_ip}:{img_endpoint}" if img_endpoint else None

        self._ctx = zmq.Context.instance()
        self._ctrl_lock = threading.Lock()   # 关键：保证 REQ 的 send/recv 成对且串行

        self._ctrl_socket = None
        if self.requre_control:
            self._ctrl_socket = self._make_ctrl_socket()
            
    def _make_ctrl_socket(self):
        s = self._ctx.socket(zmq.REQ)
        s.setsockopt(zmq.LINGER, 0)
        # s.setsockopt(zmq.IMMEDIATE, 1)
        s.connect(self.ctrl_endpoint)
        return s

    def _reset_ctrl_socket(self, reason=""):
        """close&rebuild REQ socket(Context 不动)"""
        with self._ctrl_lock:
            try:
                if self._ctrl_socket is not None:
                    try:
                        self._ctrl_socket.close(0)  # 0 表示不等待未发送完消息
                    except TypeError:
                        self._ctrl_socket.close()
            finally:
                self._ctrl_socket = self._make_ctrl_socket()
        log.warn(f"[G1UmiClient] REQ socket reset. reason={reason}")

    # ============ 控制接口 ============

    def _call(self, method, params=None, timeout_ms=600, retry_once=True):
        if not self.requre_control:
            log.warn(f'This interface does not enable the control demands!!!')
            return 
        
        params = params or {}
        req = {"method": method, "params": params, "id": int(time.time() * 1000)}
        
        # REQ 必须严格：send -> recv；且不能并发
        with self._ctrl_lock:
            try:
                self._ctrl_socket.send_string(json.dumps(req))

                poller = zmq.Poller()
                poller.register(self._ctrl_socket, zmq.POLLIN)
                socks = dict(poller.poll(timeout_ms))
                if self._ctrl_socket not in socks:
                    raise TimeoutError(f"Request {method} timeout {timeout_ms}ms")

                msg = self._ctrl_socket.recv_string()
                resp = json.loads(msg)
                if resp.get("status") != "ok":
                    raise RuntimeError(f"RPC error: {resp.get('error')}")
                return resp.get("result")

            except TimeoutError as e:
                # 超时后 REQ 很可能已经坏了（因为你没有 recv 到对应 reply）
                self._reset_ctrl_socket(reason=f"timeout on {method}")
                if retry_once:
                    # 重试一次（可选）
                    return self._call(method, params=params, timeout_ms=timeout_ms, retry_once=False)
                raise

            except zmq.ZMQError as e:
                # EFSM = Operation cannot be accomplished in current state
                if getattr(e, "errno", None) == zmq.EFSM:
                    self._reset_ctrl_socket(reason=f"EFSM on {method}")
                    if retry_once:
                        return self._call(method, params=params, timeout_ms=timeout_ms, retry_once=False)
                raise

    def ping(self):
        return self._call("ping")

    def set_neck_positions(self, rpy, wait=True):
        """
        rpy: [roll, pitch, yaw]
        """
        rpy = np.asarray(rpy, dtype=float).reshape(-1).tolist()
        return self._call("set_neck_positions", {"positions": rpy, "wait": wait})

    def get_neck_positions(self):
        return self._call("get_neck_positions")

    def set_gripper_command(self, command, key):
        return self._call(
            "set_gripper_command", {"command": float(command), "key": key}
        )

    def set_all_gripper_commands(self, left_cmd, right_cmd):
        return self._call(
            "set_all_gripper_commands", {"commands": [left_cmd, right_cmd]}
        )

    def get_all_gripper_positions(self):
        return self._call("get_all_gripper_positions")

    def get_camera_img_once(self, cam_name):
        """
            for debugging: only get one image for request and reply
        """
        result = self._call("get_camera_img_once", {"cam_name": cam_name})
        if result is None:
            return None
        jpg_list = result["jpeg"]
        jpg_bytes = bytes(jpg_list)
        buf = np.frombuffer(jpg_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img

    # ============ 图像订阅 ============

    def subscribe_images(self, cam_names):
        """
        返回一个 generator，不停 yield (cam_name, img, meta)
        cam_names: list[str]，比如 ["head"] 或 ["head", "left_fisheye"]
        """
        if self.img_endpoint is  None:
            log.warn()
        
        sub_socket = self._ctx.socket(zmq.SUB)
        sub_socket.connect(self.img_endpoint)

        # 设置订阅 topic
        if not cam_names:  # 订阅全部
            sub_socket.setsockopt(zmq.SUBSCRIBE, b"")
        else:
            for cam in cam_names:
                sub_socket.setsockopt(zmq.SUBSCRIBE, cam.encode("utf-8"))

        while True:
            try:
                topic, meta_bytes, jpg_bytes = sub_socket.recv_multipart()
            except Exception:
                break

            cam_name = topic.decode("utf-8")
            meta = json.loads(meta_bytes.decode("utf-8"))

            buf = np.frombuffer(jpg_bytes, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            yield cam_name, img, meta

        sub_socket.close()

    def close(self):
        if self.requre_control and self._ctrl_socket is not None:
            try:
                self._ctrl_socket.close(0)
            except TypeError:
                self._ctrl_socket.close()
# ================= 测试逻辑 =================

def test_neck(client: G1UmiClient):
    """
    通过网络测试云台电机：
      - 读取当前 rpy
      - 交互式输入 yaw pitch roll（单位度）并发送
    """
    log.info("===== Testing neck motor APIs (remote) =====")
    while True:
        try:
            cur_rpy = client.get_neck_positions()  # [roll, pitch, yaw]，具体取决于服务器实现
            if cur_rpy is None:
                print("neck 未启用或返回 None")
            else:
                cur_rpy_deg = [round(x * 180.0 / np.pi, 2) for x in cur_rpy]
                print(f"\n当前 neck rpy (deg) [roll, pitch, yaw]: {cur_rpy_deg}")
        except Exception as e:
            print(f"读取电机状态失败: {e}")

        s = input("请输入 yaw pitch roll (单位: 度, 例如: 0 10 -5)，或 q 返回主菜单: ")
        if s.strip().lower() == "q":
            break

        try:
            parts = s.replace(",", " ").split()
            if len(parts) != 3:
                print("输入格式不对，需要 3 个数：yaw pitch roll")
                continue
            roll_deg, pitch_deg, yaw_deg = map(float, parts)
        except Exception as e:
            print(f"解析失败: {e}")
            continue

        # 转换为弧度
        yaw = yaw_deg * np.pi / 180.0
        pitch = pitch_deg * np.pi / 180.0
        roll = roll_deg * np.pi / 180.0

        # 如果服务器端 set_neck_positions 期望的是 [roll, pitch, yaw]，
        # 记得在服务端或这里之一做一致性处理。
        cmd = [roll, pitch, yaw]
        print(f"正在发送指令(弧度): {cmd} ...")
        start = time.time()
        try:
            client.set_neck_positions(cmd, wait=True)
            recv = time.time() - start
            print(f'neck interface RTT: {recv:.4f}s')
            time.sleep(0.3)
            log.info("指令发送完成")
            cur_position = client.get_neck_positions()
            print(f'Curr position after control: {cur_position}')
        except Exception as e:
            print(f"发送指令失败: {e}")

def test_grippers(client: G1UmiClient):
    """
    通过网络测试 Pika 夹爪：
      - 读取当前左右夹爪位置
      - 支持命令: 'left 30', 'right 60', 'all 30 60'
    """
    log.info("===== Testing pika gripper APIs (remote) =====")
    while True:
        try:
            cur_pos = client.get_all_gripper_positions()
            if cur_pos is None:
                print("夹爪未启用或返回 None")
            else:
                print(f"\n当前 gripper 位置 (raw): left={cur_pos[0]}, right={cur_pos[1]}")
        except Exception as e:
            print(f"读取夹爪状态失败: {e}")

        s = input("输入命令: 'left 30', 'right 60', 'all 30 60' (角度 0~90)，或 q 返回主菜单: ")
        if s.strip().lower() == "q":
            break

        parts = s.replace(",", " ").split()
        if len(parts) < 2:
            print("命令格式不对")
            continue

        side = parts[0].lower()
        try:
            if side in ["left", "right"] and len(parts) == 2:
                deg = float(parts[1])
                print(f"正在设置 {side} 夹爪宽度为 {deg} mm")
                client.set_gripper_command(deg, side)
            elif side == "all" and len(parts) == 3:
                deg_left = float(parts[1])
                deg_right = float(parts[2])
                print(f"正在设置左右夹爪角度为: left={deg_left}, right={deg_right}")
                start = time.time()
                # 这里调用的是网络版本的 set_all_gripper_commands(left_cmd, right_cmd)
                client.set_all_gripper_commands(deg_left, deg_right)
                recv_time = time.time() - start
                print(f'all gripper RTT: {recv_time:.4f}s')
            else:
                print("命令格式不对")
                continue
        except Exception as e:
            print(f"设置失败: {e}")
            continue

        time.sleep(0.5)
        log.info("设置完成")


def test_cameras_rpc(client: G1UmiClient):
    """
    使用 RPC 单帧拉取的方式测试相机图像。
    这个方式对性能要求不高时可以用来简单验证联通性。
    """
    log.info("===== Testing camera APIs via RPC (slow) =====")
    possible_cameras = ['head', 'left_fisheye', 'right_fisheye']
    print("可用相机:", possible_cameras)

    while True:
        cam_name = input("输入相机名 (head / left_fisheye / right_fisheye)，或 q 返回主菜单: ").strip()
        if cam_name.lower() == "q":
            break
        if cam_name not in possible_cameras:
            print(f"未知相机名: {cam_name}")
            continue

        win_name = f"cam_rpc_{cam_name}"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        print(f"正在通过 RPC 拉取 {cam_name} 图像... 按 q 或 ESC 退出预览")

        try:
            while True:
                start = time.time()
                img = client.get_camera_img_once(cam_name)
                recv_time = time.time() - start
                if recv_time > 0:
                    print(f'RPC rec freq: {1.0/recv_time:.2f} Hz')
                else:
                    print('RPC rec freq: inf Hz (too fast to measure)')

                if img is None or img.size == 0:
                    print("获取图像失败或为空")
                    time.sleep(0.5)
                    continue

                cv2.imshow(win_name, img)

                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:
                    break

                # 简单的帧率控制，避免频繁 RPC 堵塞
                elapsed = time.time() - start
                if elapsed < 0.05:  # ~20 Hz
                    time.sleep(0.05 - elapsed)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"图像流异常中断: {e}")
        finally:
            cv2.destroyWindow(win_name)


def test_cameras_stream(client: G1UmiClient):
    """
    使用订阅图像流的方式测试相机图像（推荐，高帧率）。
    服务端需要在 G1UmiService 中配置好 stream_cameras 和 stream_fps。
    """
    log.info("===== Testing camera stream via PUB/SUB =====")
    base_cams = ['head', 'left_fisheye', 'right_fisheye']
    possible_cameras = base_cams + ['all']
    print("可用相机:", possible_cameras)

    def _ensure_bgr(img):
        if img is None:
            return None
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def _resize_to(img, wh):
        w, h = wh
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return img

    def _blank_tile(wh, text="empty"):
        w, h = wh
        tile = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(tile, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return tile

    while True:
        cam_name = input("输入要订阅的相机名 (head / left_fisheye / right_fisheye / all)，或 q 返回主菜单: ").strip()
        if cam_name.lower() == "q":
            break
        if cam_name not in possible_cameras:
            print(f"未知相机名: {cam_name}")
            continue

        win_name = f"cam_stream_{cam_name}"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        print(f"正在订阅 {cam_name} 图像流... 按 q 或 ESC 退出预览")

        last_t = None

        # all 模式缓存最新帧
        latest = {k: None for k in base_cams}
        target_wh = (640, 480)  # (W,H) 拼接用统一尺寸；需要可自行改

        try:
            sub_list = base_cams if cam_name == "all" else [cam_name]
            for topic_cam_name, img, meta in client.subscribe_images(sub_list):
                now = time.time()
                if last_t is not None:
                    dt = now - last_t
                    if dt > 0:
                        print(f"stream fps: {1.0/dt:.2f} Hz for {topic_cam_name}")
                    if dt > 1/30.0:
                        print(f"{'='*8} stream fps slow: {1.0/dt:.2f} for {topic_cam_name} {'='*8}")
                last_t = now

                img = _ensure_bgr(img)

                if cam_name != "all":
                    cv2.imshow(win_name, img)
                else:
                    # 更新对应相机的最新帧
                    if topic_cam_name in latest:
                        latest[topic_cam_name] = img

                    # 组装 2x2 拼图（缺帧用占位）
                    tiles = {}
                    for k in base_cams:
                        if latest[k] is None:
                            tiles[k] = _blank_tile(target_wh, f"{k}: N/A")
                        else:
                            t = _resize_to(_ensure_bgr(latest[k]), target_wh)
                            cv2.putText(t, k, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                            tiles[k] = t

                    blank = _blank_tile(target_wh, "empty")

                    top = np.concatenate([tiles["left_fisheye"], tiles["right_fisheye"]], axis=1)
                    bottom = np.concatenate([tiles["head"], blank], axis=1)
                    show = np.concatenate([top, bottom], axis=0)

                    cv2.imshow(win_name, show)

                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:
                    break

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"图像流异常中断: {e}")
        finally:
            cv2.destroyWindow(win_name)

# ================= 主程序 =================

if __name__ == "__main__":
    # 根据你的服务端配置来设置
    # 例如服务端：
    #   ctrl_endpoint="tcp://0.0.0.0:5555"
    #   img_endpoint="tcp://0.0.0.0:5556"
    #
    # 那客户端就写成：
    SERVER_IP = "192.168.1.99"  # 或 192.168.x.x 等
    CTRL_PORT = 5555
    IMG_PORT = 5556

    # ctrl_endpoint = f"tcp://{SERVER_IP}:{CTRL_PORT}"
    # img_endpoint = f"tcp://{SERVER_IP}:{IMG_PORT}"

    client = G1UmiClient(
        server_ip=SERVER_IP,
        ctrl_endpoint=CTRL_PORT,
        img_endpoint=IMG_PORT,
    )

    try:
        print("ping server:", client.ping())
    except Exception as e:
        print(f"连接服务器失败: {e}")
        exit(1)

    try:
        while True:
            print("\n========== 远程控制测试菜单 ==========")
            print(f"控制连接: {CTRL_PORT}")
            print(f"图像连接: {IMG_PORT}")
            print("1. 测试 neck（云台电机）")
            print("2. 测试 grippers（Pika 夹爪）")
            print("3. 测试 cameras（单帧 RPC 拉图）")
            print("4. 测试 cameras（高帧率流式订阅）")
            print("q. 退出程序")
            choice = input("请选择: ").strip().lower()

            if choice == "1":
                test_neck(client)
            elif choice == "2":
                test_grippers(client)
            elif choice == "3":
                test_cameras_rpc(client)
            elif choice == "4":
                test_cameras_stream(client)
            elif choice == "q":
                break
            else:
                print("无效选择，请重新输入。")
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        client.close()
        cv2.destroyAllWindows()
        print("客户端已退出。")
        
