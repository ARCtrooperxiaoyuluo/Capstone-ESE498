from setuptools import find_packages, setup
import os
import glob
package_name = 'driving'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),

    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/'+package_name+'/config',glob.glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='winston',
    maintainer_email='winston@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'speed_publisher = driving.speed_publisher:main',
            'speed_subscriber = driving.speed_subscriber:main',
            'keyboard_publisher = driving.keyboard_publisher:main',
            'motor_controller = driving.motor_controller:main',
            'pid_motor = driving.pid_motor:main',
            'lidar_publish = driving.lidar_publish:main',
            'lidar_sub = driving.lidar_sub:main',
            'speed_pub_zero = driving.speed_pub_zero:main',
            'pid_servo = driving.pid_servo:main',
            'lidar_publish_left = driving.lidar_publish_left:main',
            'north = driving.north:main',
            'pid_servo_logic = driving.pid_servo_logic:main',
            'lidar_pub = driving.lidar_pub:main',
            'lane = driving.lane:main',
            'mini_city = driving.mini_city:main',
            'pid_servo_new = driving.pid_servo_new:main',

        ],
    },
)
