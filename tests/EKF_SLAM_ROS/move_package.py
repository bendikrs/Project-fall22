import shutil
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--username', type=str, required=False)
args = parser.parse_args()
if args.username:
    username = args.username[1:-1]
else:
    username = 'bendikrs'

original = r'/home/' + username + r'/git/Project-fall22/tests/EKF_SLAM_ROS/ros2_package/ekf_slam_ros2'
target = r'/home/' + username + r'/turtlebot3_ws/src/ekf_slam_ros2'


# if folder exists, delete it
if os.path.exists(target):
    shutil.rmtree(target)

# copy the new folder
shutil.copytree(original, target)