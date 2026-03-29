"""
Live data logger node.

Subscribes to the PF estimate and ground truth TF, logs synchronized
pose pairs to a CSV for offline analysis.

Launch:
    ros2 run localization data_logger --ros-args \
        -p output_file:=run_200p_noise0.1.csv

The CSV columns are:
    timestamp, gt_x, gt_y, gt_theta, pf_x, pf_y, pf_theta, pos_error, heading_error
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformListener
from tf_transformations import euler_from_quaternion

import numpy as np
import csv
import os
import time


class DataLogger(Node):

    def __init__(self):
        super().__init__('data_logger')

        self.declare_parameter('output_file', 'pf_log.csv')
        self.declare_parameter('pf_topic', '/pf/pose/odom')

        self.output_file = self.get_parameter('output_file').get_parameter_value().string_value
        pf_topic = self.get_parameter('pf_topic').get_parameter_value().string_value

        # TF listener for ground truth (map -> base_link)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribe to PF estimate
        self.pf_sub = self.create_subscription(Odometry, pf_topic, self.pf_callback, 1)

        # CSV setup
        self.csv_file = open(self.output_file, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'timestamp', 'gt_x', 'gt_y', 'gt_theta',
            'pf_x', 'pf_y', 'pf_theta', 'pos_error', 'heading_error'
        ])

        self.start_time = time.time()
        self.row_count = 0

        self.get_logger().info(f"Logging to {os.path.abspath(self.output_file)}")

    def pf_callback(self, msg):
        # Extract PF estimate
        pf_x = msg.pose.pose.position.x
        pf_y = msg.pose.pose.position.y
        pf_q = msg.pose.pose.orientation
        _, _, pf_theta = euler_from_quaternion([pf_q.x, pf_q.y, pf_q.z, pf_q.w])

        # Look up ground truth TF
        try:
            tf = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
        except Exception:
            return

        gt_x = tf.transform.translation.x
        gt_y = tf.transform.translation.y
        gt_q = tf.transform.rotation
        _, _, gt_theta = euler_from_quaternion([gt_q.x, gt_q.y, gt_q.z, gt_q.w])

        # Compute errors
        pos_error = np.sqrt((pf_x - gt_x) ** 2 + (pf_y - gt_y) ** 2)
        heading_error = abs(np.arctan2(
            np.sin(pf_theta - gt_theta),
            np.cos(pf_theta - gt_theta)
        ))

        t = time.time() - self.start_time

        self.csv_writer.writerow([
            f'{t:.4f}', f'{gt_x:.4f}', f'{gt_y:.4f}', f'{gt_theta:.4f}',
            f'{pf_x:.4f}', f'{pf_y:.4f}', f'{pf_theta:.4f}',
            f'{pos_error:.4f}', f'{heading_error:.4f}'
        ])

        self.row_count += 1
        if self.row_count % 100 == 0:
            self.csv_file.flush()
            self.get_logger().info(f"Logged {self.row_count} rows, latest pos_error={pos_error:.3f}m")

    def destroy_node(self):
        self.csv_file.close()
        self.get_logger().info(f"Saved {self.row_count} rows to {self.output_file}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DataLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
