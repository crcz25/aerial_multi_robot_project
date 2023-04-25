"""Simulate a Tello drone"""

import os
import random

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess


def generate_launch_description():
     world_path = os.path.join(get_package_share_directory('tello_gazebo'), 'worlds', 'simple.world')

     ns = 'drone1'
     urdf_path = os.path.join(get_package_share_directory('tello_description'), 'urdf', 'tello_1.urdf')

     gates_nodes = create_gates_random()

     stop_sdf_path = os.path.join(
          get_package_share_directory('tello_gazebo'),
          'models',
          'stop_sign',
          'model.sdf'
     )
     pallet_sdf_path = os.path.join(
          get_package_share_directory('tello_gazebo'),
          'models',
          'euro_pallet',
          'model.sdf'
     )

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
          
          # Add the gates - spawn entity list
          *gates_nodes,

          # Add the landing platform (stop and pallet)
          Node(
               package='tello_gazebo',
               executable='inject_entity.py',
               output='screen',
               arguments=[pallet_sdf_path, '-0.4', '27', '1', '0.01', 'pallet_1']
          ),
          Node(
               package='tello_gazebo',
               executable='inject_entity.py',
               output='screen',
               arguments=[pallet_sdf_path, '0.41', '27', '1', '0.01',  'pallet_2']
          ),
          Node(
               package='tello_gazebo',
               executable='inject_entity.py',
               output='screen',
               arguments=[stop_sdf_path, '0', '27.8', '0', '0.01', 'stop_1']
          ),
          
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

def create_gates_random():
     colors = ['red', 'green', 'blue', 'white']
     shapes = ['circle', 'square']
     gates_nodes = []

     gates_locations = [
          # X, Y, Z, Theta
          ['0.45', '3', '1', '-0.35'],
          ['-0.45', '7', '1', '0.35'],
          ['0.45', '11', '1', '-0.35'],
          ['0.65', '15', '1', '-0.10'],
          ['-0.85', '19', '1', '0.10'],
          ['0.25', '23', '1', '0.01'],
     ]

     for gate_n, location in enumerate(gates_locations):
          color = random.choice(colors)
          shape = random.choice(shapes)

          if color == 'white':
               model_path = f'wgate_{shape}_{gate_n}.urdf'
               urdf_gate_path = os.path.join(
                    get_package_share_directory('tello_description'),
                    'urdf',
                    model_path
               )
          else:
               model_color_folder = f'{shape}_gate_{color}'
               
               urdf_gate_path = os.path.join(
                    get_package_share_directory('tello_gazebo'),
                    'models',
                    model_color_folder,
                    'model.sdf'
               )
     
          new_node = Node(
                    package='tello_gazebo',
                    executable='inject_entity.py',
                    output='screen',
                    arguments=[urdf_gate_path, *location, f'gate_{gate_n}']
               )
          gates_nodes.append(new_node)

     return gates_nodes
