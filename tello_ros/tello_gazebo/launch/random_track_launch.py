"""Simulate a Tello drone"""

import os
import random

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess


def generate_launch_description():
     world_path = os.path.join(get_package_share_directory('tello_gazebo'), 'worlds', 'demo_track.world')

     ns = 'drone1'
     urdf_path = os.path.join(get_package_share_directory('tello_description'), 'urdf', 'tello_1.urdf')

     gates_nodes = create_gates_random()

     return LaunchDescription([
          # Launch Gazebo, loading tello.world
          ExecuteProcess(
               cmd=['gazebo',
                    '--verbose',
                    '-s', 'libgazebo_ros_init.so',     # Publish /clock
                    '-s', 'libgazebo_ros_factory.so',  # Provide gazebo_ros::Node
                    world_path],
               output='screen'
          ),
          # Add the gates spawn entity list
          *gates_nodes,
          # Spawn tello.urdf
          Node(
               package='tello_gazebo',
               executable='inject_entity.py',
               output='screen',
               arguments=[urdf_path, '0', '0', '1', '1.57079632679']
          ),
          # Publish static transforms
          Node(
               package='robot_state_publisher',
               executable='robot_state_publisher',
               output='screen',
               arguments=[urdf_path]
          ),
          # Joystick driver, generates /namespace/joy messages
          Node(
               package='joy',
               executable='joy_node',
               output='screen',
               namespace=ns
          ),
          # Joystick controller, generates /namespace/cmd_vel messages
          Node(
               package='tello_driver',
               executable='tello_joy_main',
               output='screen',
               namespace=ns
          ),
     ])

def create_gates_random(
          total_gates=3,
          exclusive_color=None,
):
     colors = ['r', 'g']
     gates_nodes = []

     if not exclusive_color:
          for gate in range(1, total_gates+1):
               color = random.choice(colors)
               if color == 'r':
                    urdf_rgate_path = os.path.join(
                         get_package_share_directory('tello_description'),
                         'urdf',
                         f'rgate_{gate}.urdf'
                    )

                    new_node = Node(
                         package='tello_gazebo',
                         executable='inject_entity.py',
                         output='screen',
                         arguments=[urdf_rgate_path, '5', '0', '1', '1.57079632679']
                    )
               elif color == 'g':
                    urdf_ggate_path = os.path.join(
                         get_package_share_directory('tello_description'),
                         'urdf',
                         f'ggate_{gate}.urdf'
                    )

                    new_node = Node(
                         package='tello_gazebo',
                         executable='inject_entity.py',
                         output='screen',
                         arguments=[urdf_ggate_path, '5', '5', '1', '1.57079632679']
                    )
               elif color == 'b':
                    urdf_bgate_path = os.path.join(
                         get_package_share_directory('tello_description'),
                         'urdf',
                         f'bgate_{gate}.urdf'
                    )

                    new_node = Node(
                         package='tello_gazebo',
                         executable='inject_entity.py',
                         output='screen',
                         arguments=[urdf_bgate_path, '10', '5', '1', '1.57079632679']
                    )
               gates_nodes.append(new_node)

     return gates_nodes
