import numpy as np
import time
import csv
import os

from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose, Quaternion
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

from rclpy.node import Node
import rclpy
from scipy.spatial.transform import Rotation as R

assert rclpy


class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "default")
        self.declare_parameter('num_particles', 200)
        self.declare_parameter('resample_noise_xy', 0.05)
        self.declare_parameter('resample_noise_theta', 0.01)

        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value
        self.resample_noise_xy = self.get_parameter('resample_noise_xy').get_parameter_value().double_value
        self.resample_noise_theta = self.get_parameter('resample_noise_theta').get_parameter_value().double_value

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.

        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan") #change for raecar

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

        # Publisher for particle visualization
        self.particle_pub = self.create_publisher(PoseArray, "/particles", 1)

        # Transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        # Read num_beams_per_particle after sensor model declares it
        self.num_beams_per_particle = self.get_parameter('num_beams_per_particle').get_parameter_value().integer_value

        # Particle state
        self.particles = np.zeros((self.num_particles, 3))
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.initialized = False

        # Track previous timestamp for twist-based odometry
        self.prev_odom_time = None

        # Runtime profiler (set profile=True to log timing data)
        self.declare_parameter('profile', False)
        self.profile = self.get_parameter('profile').get_parameter_value().bool_value
        self.profile_file = None
        self.profile_writer = None
        if self.profile:
            path = f'pf_profile_{self.num_particles}p_{self.num_beams_per_particle}b.csv'
            self.profile_file = open(path, 'w', newline='')
            self.profile_writer = csv.writer(self.profile_file)
            self.profile_writer.writerow([
                'timestamp', 'motion_model_ms', 'sensor_model_ms',
                'resample_ms', 'total_ms', 'num_particles', 'num_beams'
            ])
            self.get_logger().info(f"Profiling enabled, logging to {os.path.abspath(path)}")

        self.get_logger().info("=============+READY+=============")

    def odom_callback(self, msg):
        """
        Use the twist (velocity) from odometry to compute body-frame deltas,
        then apply the motion model.
        """
        if not self.initialized:
            return

        # Compute dt from message timestamps
        now = self.get_clock().now().nanoseconds
        if self.prev_odom_time is None:
            self.prev_odom_time = now
            return

        dt = (now - self.prev_odom_time) * 1e-9
        self.prev_odom_time = now

        if dt <= 0 or dt > 1.0:
            return

        # Twist is already in body frame
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        wz = msg.twist.twist.angular.z

        dx_body = vx * dt
        dy_body = vy * dt
        dtheta = wz * dt

        # Apply motion model
        odometry = np.array([dx_body, dy_body, dtheta])
        t0 = time.perf_counter() if self.profile else 0
        self.particles = self.motion_model.evaluate(self.particles, odometry)
        if self.profile:
            dt_ms = (time.perf_counter() - t0) * 1000
            self.profile_writer.writerow([
                f'{time.time():.4f}', f'{dt_ms:.3f}', '', '',
                f'{dt_ms:.3f}', self.num_particles, self.num_beams_per_particle
            ])

        # Publish the current estimate
        self.publish_estimate()

    def laser_callback(self, msg):
        """
        Use the sensor model to weight particles, then resample.
        """
        if not self.initialized:
            return

        # Lessen the laser scan to num_beams_per_particle
        ranges = np.array(msg.ranges)
        num_ranges = len(ranges)
        if num_ranges > self.num_beams_per_particle:
            indices = np.linspace(0, num_ranges - 1, self.num_beams_per_particle, dtype=int)
            observation = ranges[indices]
        else:
            observation = ranges

        # Clamp infinity and nan
        max_range = msg.range_max
        observation = np.where(np.isfinite(observation), observation, max_range)

        # Evaluate sensor model
        t0 = time.perf_counter() if self.profile else 0
        probabilities = self.sensor_model.evaluate(self.particles, observation)
        if self.profile:
            t_sensor = (time.perf_counter() - t0) * 1000
        if probabilities is None:
            return

        # Normalize weights
        total = np.sum(probabilities)
        if total > 0:
            self.weights = probabilities / total
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles

        # Resample particles based on weights
        t1 = time.perf_counter() if self.profile else 0
        indices = np.random.choice(
            self.num_particles,
            size=self.num_particles,
            replace=True,
            p=self.weights
        )
        self.particles = self.particles[indices]

        # Add small noise after resampling
        self.particles[:, 0] += np.random.normal(0, self.resample_noise_xy, self.num_particles)
        self.particles[:, 1] += np.random.normal(0, self.resample_noise_xy, self.num_particles)
        self.particles[:, 2] += np.random.normal(0, self.resample_noise_theta, self.num_particles)
        if self.profile:
            t_resample = (time.perf_counter() - t1) * 1000
            self.profile_writer.writerow([
                f'{time.time():.4f}', '', f'{t_sensor:.3f}', f'{t_resample:.3f}',
                f'{t_sensor + t_resample:.3f}', self.num_particles, self.num_beams_per_particle
            ])
            if np.random.random() < 0.02:  # flush occasionally
                self.profile_file.flush()

        # Reset weights and publish estimate
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.publish_estimate()
        self.publish_particles()

    def pose_callback(self, msg):
        """
        Initialize particles around the given pose estimate from RViz.
        """
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        _, _, theta = R.from_quat([
            orientation.x, orientation.y, orientation.z, orientation.w
        ]).as_euler("xyz")

        # Spread particles around the pose
        self.particles[:, 0] = x + np.random.normal(0, 0.5, self.num_particles)
        self.particles[:, 1] = y + np.random.normal(0, 0.5, self.num_particles)
        self.particles[:, 2] = theta + np.random.normal(0, 0.2, self.num_particles)

        self.weights = np.ones(self.num_particles) / self.num_particles
        self.initialized = True

        # Reset odometry tracking
        self.prev_odom_time = None

        self.get_logger().info(f"Particles initialized around ({x:.2f}, {y:.2f}, {theta:.2f})")
        self.publish_estimate()
        self.publish_particles()

    def compute_average_pose(self):
        """
        Compute the weighted average pose of the particles.
        Uses circular mean for theta to avoid the issue with wrapround at theta=pi.
        """
        avg_x = np.mean(self.particles[:, 0])
        avg_y = np.mean(self.particles[:, 1])

        avg_theta = np.arctan2(
            np.mean(np.sin(self.particles[:, 2])),
            np.mean(np.cos(self.particles[:, 2]))
        )

        return avg_x, avg_y, avg_theta

    def publish_estimate(self):
        """
        Publish the estimated pose as an Odometry message and a TF transform.
        """
        avg_x, avg_y, avg_theta = self.compute_average_pose()
        now = self.get_clock().now().to_msg()

        # Publish odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = now
        odom_msg.header.frame_id = "/map"
        odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y = avg_x, avg_y
        q = R.from_euler("xyz", (0, 0, avg_theta)).as_quat()
        odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w = q[0], q[1], q[2], q[3]
        self.odom_pub.publish(odom_msg)

        # Publish transform: map -> particle_filter_frame
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = "/map"
        t.child_frame_id = self.particle_filter_frame
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = avg_x, avg_y, 0.0
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q[0], q[1], q[2], q[3]
        self.tf_broadcaster.sendTransform(t)

    def publish_particles(self):
        """
        Publish particles as a PoseArray for visualization in RViz.
        """
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = "/map"

        for i in range(self.num_particles):
            pose = Pose()
            pose.position.x, pose.position.y = self.particles[i, 0], self.particles[i, 1]
            q = R.from_euler("xyz", (0, 0, self.particles[i, 2])).as_quat()
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q[0], q[1], q[2], q[3]
            pa.poses.append(pose)

        self.particle_pub.publish(pa)


def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
