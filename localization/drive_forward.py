"""
Simple drive-forward script for testing the particle filter in simulation.

Publishes constant-velocity AckermannDriveStamped messages to /drive.
Drives straight for a set duration, then optionally turns.

Usage:
    ros2 run localization drive_forward
    ros2 run localization drive_forward --ros-args -p speed:=1.0 -p steering_angle:=0.2
"""

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
import time
import math


class DriveForward(Node):

    def __init__(self):
        super().__init__('drive_forward')

        self.declare_parameter('speed', 1.0)
        self.declare_parameter('steering_angle', 0.0)
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('rate', 20.0)  # Hz
        # Drive pattern: 'straight', 'circle', 'figure8'
        self.declare_parameter('pattern', 'straight')

        self.speed = self.get_parameter('speed').get_parameter_value().double_value
        self.steering_angle = self.get_parameter('steering_angle').get_parameter_value().double_value
        drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value
        rate = self.get_parameter('rate').get_parameter_value().double_value
        self.pattern = self.get_parameter('pattern').get_parameter_value().string_value

        self.pub = self.create_publisher(AckermannDriveStamped, drive_topic, 1)
        self.timer = self.create_timer(1.0 / rate, self.timer_callback)
        self.start_time = time.time()

        self.get_logger().info(
            f"Driving: pattern={self.pattern}, speed={self.speed}, "
            f"steering={self.steering_angle}, topic={drive_topic}"
        )

    def stop(self):
        msg = AckermannDriveStamped()
        msg.drive.speed = 0.0
        msg.drive.steering_angle = 0.0
        for _ in range(10):
            self.pub.publish(msg)
            time.sleep(0.01)

    def timer_callback(self):
        t = time.time() - self.start_time

        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()

        if self.pattern == 'straight':
            msg.drive.speed = self.speed
            msg.drive.steering_angle = self.steering_angle

        elif self.pattern == 'circle':
            msg.drive.speed = self.speed
            msg.drive.steering_angle = 0.3

        elif self.pattern == 'figure8':
            # Alternate steering every 5 seconds
            period = 10.0
            phase = (t % period) / period
            msg.drive.speed = self.speed
            msg.drive.steering_angle = 0.3 * math.sin(2 * math.pi * phase)

        self.pub.publish(msg)


def main(args=None):
    import signal

    rclpy.init(args=args)
    node = DriveForward()

    def on_sigint(sig, frame):
        node.stop()
        raise SystemExit()

    signal.signal(signal.SIGINT, on_sigint)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.try_shutdown()
