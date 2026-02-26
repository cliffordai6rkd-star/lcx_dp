# Pika trcker用法
## 硬件配置
按照[文件](https://agilexsupport.yuque.com/staff-hso6mo/peoot3/axi8hh9h9t2sh2su?singleDoc#HWz6F)里2.2来配置无线接收器以及pika sense上的tracker

## 配置
1. 如果是第一次使用pika sense， 请先下载pika_ros， 通过 `git clone https://github.com/agilexrobotics/pika_ros.git`
2. 在终端里运行
    ```bash
        cd pika_ros
        git submodule update --init --recursive
        sudo apt-get update && sudo apt install libjsoncpp-dev ros-noetic-ddynamic-reconfigure libpcap-dev  ros-noetic-serial ros-noetic-ros-numpy ros-noetic-librealsense2 python3-pcl libqt5serialport5-dev build-essential zlib1g-dev libx11-dev libusb-1.0-0-dev freeglut3-dev liblapacke-dev libopenblas-dev libatlas-base-dev cmake  git libssl-dev  pkg-config libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev g++  python3-pip  libopenvr-dev
        sudo apt install libopenvr-dev
        <!-- 配置规则 -->
        sudo cp scripts/81-vive.rules /etc/udev/rules.d/
        sudo udevadm control --reload-rules && sudo udevadm trigger
    ```
3. 解压校准文件安装包， 去到pika ros的文件夹下运行以下指令
    ```bash
        unzip source/install.zip -d .  
        cd install/lib
        sudo chmod 777 survive-cli
        ./survive-cli --force-calibrate
        ./survive-cli --globalscenesolver 0
    ```
    
保证两个基站配置完成， 运行完校准文件后输出为以下内容才到：
Info: Loaded drivers: GlobalSceneSolver, HTCVive
Info: Force calibrate flag set -- clearing position on all lighthouses
Info: Adding tracked object WM0 from HTC
Info: Device WM0 has watchman FW version 1592875850 and FPGA version 538/7/2; named '                       watchman'. Hardware id 0x84020109 Board rev: 3 (len 56)
Info: Detected LH gen 2 system.
Info: LightcapMode (WM0) 1 -> 2 (ff)
Info: Adding lighthouse ch 1 (idx: 0, cnt: 1)
Info: OOTX not set for LH in channel 1; attaching ootx decoder using device WM0
Info: Adding lighthouse ch 3 (idx: 1, cnt: 2)
Info: OOTX not set for LH in channel 3; attaching ootx decoder using device WM0
Info: (1) Preamble found
Info: (3) Preamble found
Info: Got OOTX packet 1 4602fcee
Info: Got OOTX packet 3 60ba9d4d
Info: MPFIT success 104122.009371/112.6705912418/0.0001332 (42 measurements, 1, MP_OK_CHI, 5 iters, up err 0.0003339, trace 0.0000091)
Info: Global solve with 1 scenes for 0 with error of 104122.009371/112.6705912418 (acc err 0.0010)
Info: Global solve with 1 scenes for 1 with error of 104122.009371/112.6705912418 (acc err 0.0013)
Info: Using LH 0 (4602fcee) as reference lighthouse
Info: MPFIT success 497.065942/453.1467622320/0.0002276 (84 measurements, 3, MP_OK_BOTH, 27 iters, up err 0.0002664, trace 0.0000221)
Info: Global solve with 2 scenes for 0 with error of 497.065942/453.1467622320 (acc err 0.0011)
Info: Global solve with 2 scenes for 1 with error of 497.065942/453.1467622320 (acc err 0.0013)，

一定要注意需要有两个 Preamble found才证明你两个基站配置完成， 如果失败请运行`rm ~/.config/libsurvive/config.json`, 然后在==运行第三步==

## 设备USB 端口绑定
1. 在'/etc/udev/rules.d'创建两个文件， 运行以下指令
```bash
    cd /etc/udev/rules.d
    sudo touch pika_serial.rules
    sudo chmod a+x pika_serial.rules
    sudo touch pika_fisheye.rules    
    sudo chmod a+x pika_fisheye.rules
```
2. 编写`pika_serial.rules`的规则内容(使用vim或者gedit)， 内容为下， 请注意： 根据以下方法来更改每个设备KERNELS里的内容： 首先插拔指定设备的USB口并使用`ls /dev | grep USB`确认当前设备的USB号， 然后使用以下指令 `udevadm info -a -n /dev/ttyUSB<你查到的USB号> | grep 'KERNELS=='`然后把输出的第一行KERNELS的号在文件中即进行更改
```bash
    ## pika grippers
    ACTION=="add", KERNELS=="1-4.3.1.4:1.0", SUBSYSTEMS=="usb", MODE:="0666", SYMLINK+="ttyUSB80"
    ACTION=="add", KERNELS=="1-3.1.4:1.0", SUBSYSTEMS=="usb", MODE:="0666", SYMLINK+="ttyUSB81"

    ## pika senses
    ACTION=="add", KERNELS=="1-7.4.4:1.0", SUBSYSTEMS=="usb", MODE:="0666", SYMLINK+="ttyUSB70"
    ACTION=="add", KERNELS=="1-7.3.4:1.0", SUBSYSTEMS=="usb", MODE:="0666", SYMLINK+="ttyUSB71"
```
3. 编写`pika_fisheye.rules`的规则内容，内容如下。
```bash
ACTION=="add", KERNEL=="video[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60]*", KERNELS=="1-7.1:1.0", SUBSYSTEMS=="usb", MODE:="0666", SYMLINK+="video80"
ACTION=="add", KERNEL=="video[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60]*", KERNELS=="1-7.1:1.0", SUBSYSTEMS=="usb", MODE:="0666", SYMLINK+="video81"

# pika sense, 查看相机ideo编号可以使用pika_sdk/tools下的check_multiple_device的脚本
ACTION=="add", KERNEL=="video[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60]*", KERNELS=="1-7.1:1.0", SUBSYSTEMS=="usb", MODE:="0666", SYMLINK+="video70"
ACTION=="add", KERNEL=="video[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60]*", KERNELS=="1-7.1:1.0", SUBSYSTEMS=="usb", MODE:="0666", SYMLINK+="video71"

```



