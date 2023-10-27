from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os

params_file = '/home/do/ros2_ws/src/gv_recon/config/config.yaml'

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='gv_recon',
            executable='az_yoso',
            name='kfusion_node',
            output='screen',
            parameters=[params_file]
        ),
    ])
