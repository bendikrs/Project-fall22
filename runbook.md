# Runbook
## Setup ROS2 environment on remote PC
1. Install Ubuntu 20.04.5 LTS (Focal Fossa) desktop image on the computer you want to run the EKF SLAM on
    1. [Ubuntu 20.04](https://releases.ubuntu.com/20.04/)
1. Install ROS2 Foxy on remote PC
    1. `wget https://raw.githubusercontent.com/ROBOTIS-GIT/robotis_tools/master/install_ros2_foxy.sh`
    1. `sudo chmod 755 ./install_ros2_foxy.sh`
    1. `bash ./install_ros2_foxy.sh`
1. Install Gazebo packages
    1. `sudo apt-get install ros-foxy-gazebo-*`
1. Install TurtleBot3 packages
    1. `source ~/.bashrc`
    1. `sudo apt install ros-foxy-dynamixel-sdk`
    1. `mkdir -p ~/turtlebot3_ws/src`
    1. `cd ~/turtlebot3_ws/src/`
    1. `git clone -b foxy-devel https://github.com/ROBOTIS-GIT/DynamixelSDK.git`
    1. `git clone -b foxy-devel https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git`
    1. `git clone -b foxy-devel https://github.com/ROBOTIS-GIT/turtlebot3.git`
    1. `git clone -b foxy-devel https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git`
    1. Move the ekf_slam_ros2 package from repo into the ~/turtlebot3_ws/src/ folder
    1. `cd ~/turtlebot3_ws && colcon build --symlink-install`
    1. `echo 'source ~/turtlebot3_ws/install/setup.bash' >> ~/.bashrc `
    1. `source ~/.bashrc`

## Setup SBC (Raspberry Pi) and OpenCR
1. Follow the Robotis e-Manual to setup the physical TurtleBot3
    1. [SBC Setup](https://emanual.robotis.com/docs/en/platform/turtlebot3/sbc_setup/#sbc-setup)
    1. [OpenCR Setup](https://emanual.robotis.com/docs/en/platform/turtlebot3/opencr_setup/#opencr-setup)
