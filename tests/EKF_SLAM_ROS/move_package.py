import shutil
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--username', type=str, required=True)
args = parser.parse_args()


original = r'/home/' + repr(args.username)[1:-1] + r'/turtlebot3_ws/src/ekf_slam_ros2'
target = r'/home/' + repr(args.username)[1:-1] + r'/git/Project-fall22/tests/EKF_SLAM_ROS/ros2_package'

shutil.copytree(original, target)