
```
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab/docker

python container.py start ros2 --files x11.yaml --env-files .env.ros2
```
使用vscode 的 container插件，找到isaac-lab-ros2这个container，右键Attach Shell
在新打开的terminal中，执行
```
isaaclab -s
```

点击vscode左下角的`Open a Remote Window` -> `Attach to Running Container ...`
```
git clone https://github.com/isaac-sim/IsaacSim-ros_workspaces.git
cd IsaacSim-ros_workspaces/
cd humble_ws/
cd src/
git clone https://github.com/ros-drivers/ackermann_msgs.git
cd ..
rosdep install -i --from-path src --rosdistro humble -y
sudo rosdep init
rosdep update
colcon build
```

按照[tutorial_ros2_manipulation](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/tutorial_ros2_manipulation.html) 进行操作，即可通过ros包控制仿真环境fr3