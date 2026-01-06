from openpi_client import websocket_client_policy as _websocket_client_policy
import time, cv2
import numpy as np

host = "192.168.35.213" 
port = 8000
api_key = None
pi0_policy = _websocket_client_policy.WebsocketClientPolicy(
                host=host, port=port, api_key=api_key,)
print(f"Server metadata: {pi0_policy.get_server_metadata()}")

fake_obs = {
    "state": np.zeros(23),
    "right_fisheye_color": np.zeros((480, 640, 3), dtype=np.uint8),
    "left_fisheye_color": np.zeros((480, 640, 3), dtype=np.uint8),
    "head_color": np.zeros((480, 640, 3), dtype=np.uint8),
    "task": "hello, who you are!!!"
}

for key, value in fake_obs.items():
    if "color" not in key:
        continue
    fake_obs[key] = cv2.resize(value, dsize=[224, 224])
    print(f'{key} obs shape: {fake_obs[key].shape}')

test_num = 100
start = time.perf_counter()
for i in range(test_num):
    pi0_policy.infer(fake_obs)
print(f'avg time: {(time.perf_counter() - start) / test_num * 1000}ms')
