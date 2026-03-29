"""
Odometry noise injector node.

Subscribes to /odom, adds configurable Gaussian noise to the twist,
republishes on /noisy_odom. Point your particle filter's odom_topic
param at /noisy_odom to test robustness.

Launch:
    ros2 run localization noise_injector --ros-args \
        -p velocity_noise_std:=0.1 \
        -p angular_noise_std:=0.05
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np


class NoiseInjector(Node):

    def __init__(self):
        super().__init__('noise_injector')

        self.declare_parameter('velocity_noise_std', 0.1)
        self.declare_parameter('angular_noise_std', 0.05)
        self.declare_parameter('input_topic', '/odom')
        self.declare_parameter('output_topic', '/noisy_odom')

        self.vel_std = self.get_parameter('velocity_noise_std').get_parameter_value().double_value
        self.ang_std = self.get_parameter('angular_noise_std').get_parameter_value().double_value
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        self.sub = self.create_subscription(Odometry, input_topic, self.odom_callback, 1)
        self.pub = self.create_publisher(Odometry, output_topic, 1)

        self.get_logger().info(
            f"Noise injector: {input_topic} -> {output_topic} "
            f"(vel_std={self.vel_std}, ang_std={self.ang_std})"
        )

    def odom_callback(self, msg):
        noisy = Odometry()
        noisy.header = msg.header
        noisy.child_frame_id = msg.child_frame_id

        # Pass pose through unchanged (PF should use twist anyway,
        # but we corrupt pose too so diff-based approaches also see noise)
        noisy.pose = msg.pose
        noisy.pose.pose.position.x += np.random.normal(0, self.vel_std * 0.02)
        noisy.pose.pose.position.y += np.random.normal(0, self.vel_std * 0.02)

        # Add noise to twist
        noisy.twist = msg.twist
        noisy.twist.twist.linear.x += np.random.normal(0, self.vel_std)
        noisy.twist.twist.linear.y += np.random.normal(0, self.vel_std)
        noisy.twist.twist.angular.z += np.random.normal(0, self.ang_std)

        self.pub.publish(noisy)


def main(args=None):
    rclpy.init(args=args)
    node = NoiseInjector()
    rclpy.spin(node)
    rclpy.shutdown()
