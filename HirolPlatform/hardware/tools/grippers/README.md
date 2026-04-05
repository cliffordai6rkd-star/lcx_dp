# Pika gripper usage

## Notice:
-  当你使用某个夹爪的时候请插拔查看夹爪的port， 你需要将这个port的权限下放； 示例： 如果你的port为`/dev/ttyUSB0`，然后调用`sudo chmod 666 /dev/ttyUSB0`来下放权限
-  使用`git clone https://github.com/agilexrobotics/pika_sdk.git
`下载pika_sdk 并且完成所有的python pika_sdk的安装
- 进入到pika_sdk的项目路径运行`python tools/multi_device_detector.py`来找到鱼眼相机的index