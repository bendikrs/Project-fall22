from setuptools import setup
import os
from glob import glob
package_name = 'ekf_slam_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bendikrs',
    maintainer_email='bendik_sto@hotmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'EKF_SLAM = ekf_slam_ros2.EKF_SLAM:main',
            'OCCUPANCY_GRID_MAP = ekf_slam_ros2.OCCUPANCY_GRID_MAP:main',
            'LANDMARK_DETECTION = ekf_slam_ros2.LANDMARK_DETECTION:main'
        ],
    },
)
