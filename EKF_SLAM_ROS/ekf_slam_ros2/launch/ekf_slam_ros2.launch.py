from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ekf_slam_ros2',
            executable='LANDMARK_DETECTION',
            name='LANDMARK_DETECTION',
            output='screen'
        ),
        Node(
            package='ekf_slam_ros2',
            executable='EKF_SLAM',
            name='EKF_SLAM',
            output='screen'
        ),
        Node(
            package='ekf_slam_ros2',
            executable='OCCUPANCY_GRID_MAP',
            name='OCCUPANCY_GRID_MAP',
            output='screen'
        ),
    ])
