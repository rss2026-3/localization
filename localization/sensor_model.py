import numpy as np
from scan_simulator_2d import PyScanSimulator2D
# Try to change to just `from scan_simulator_2d import PyScanSimulator2D`
# if any error re: scan_simulator_2d occurs

from scipy.spatial.transform import Rotation as R

from nav_msgs.msg import OccupancyGrid

import sys

np.set_printoptions(threshold=sys.maxsize)


class SensorModel:

    def __init__(self, node):
        node.declare_parameter('map_topic', "default")
        node.declare_parameter('num_beams_per_particle', 1)
        node.declare_parameter('scan_theta_discretization', 1.0)
        node.declare_parameter('scan_field_of_view', 1.0)
        node.declare_parameter('lidar_scale_to_map_scale', 1.0)

        self.map_topic = node.get_parameter('map_topic').get_parameter_value().string_value
        self.num_beams_per_particle = node.get_parameter('num_beams_per_particle').get_parameter_value().integer_value
        self.scan_theta_discretization = node.get_parameter(
            'scan_theta_discretization').get_parameter_value().double_value
        self.scan_field_of_view = node.get_parameter('scan_field_of_view').get_parameter_value().double_value
        self.lidar_scale_to_map_scale = node.get_parameter(
            'lidar_scale_to_map_scale').get_parameter_value().double_value

        ####################################
        # Adjust these parameters
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 8.0

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        ####################################

        node.get_logger().info("%s" % self.map_topic)
        node.get_logger().info("%s" % self.num_beams_per_particle)
        node.get_logger().info("%s" % self.scan_theta_discretization)
        node.get_logger().info("%s" % self.scan_field_of_view)

        # Precompute the sensor model table
        self.sensor_model_table = np.empty((self.table_width, self.table_width))
        self.precompute_sensor_model()

        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
            self.num_beams_per_particle,
            self.scan_field_of_view,
            0,  # This is not the simulator, don't add noise
            0.01,  # This is used as an epsilon
            self.scan_theta_discretization)

        # Subscribe to the map
        self.map = None
        self.map_set = False
        self.map_resolution = None
        self.map_subscriber = node.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            1)

    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.

        For each discrete computed range value, this provides the probability of
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A

        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """

        #Table axes: rows = measured z, cols = ground truth d
        z_max = self.table_width - 1
        z = np.arange(self.table_width)  # measured values
        d = np.arange(self.table_width)  # ground truth values

        #z_grid[i,j] = z[i], d_grid[i,j] = d[j]
        z_grid, d_grid = np.meshgrid(z, d, indexing='ij')

        #p_hit: Gaussian centered at d
        p_hit = np.exp(-0.5 * ((z_grid - d_grid) / self.sigma_hit) ** 2)
        col_sums = p_hit.sum(axis=0)
        col_sums[col_sums == 0] = 1.0
        p_hit = p_hit / col_sums

        #p_short: 2/d * (1 - z/d)
        p_short = np.zeros_like(z_grid, dtype=float)
        valid = (d_grid > 0) & (z_grid <= d_grid) & (z_grid >= 0)
        p_short[valid] = (2.0 / d_grid[valid]) * (1.0 - z_grid[valid] / d_grid[valid])

        #p_max: spike at z_max
        p_max = np.zeros_like(z_grid, dtype=float)
        p_max[z_grid == z_max] = 1.0

        #p_rand: uniform
        p_rand = np.ones_like(z_grid, dtype=float) / z_max

        #Combined and normalized table
        self.sensor_model_table = (
            self.alpha_hit * p_hit +
            self.alpha_short * p_short +
            self.alpha_max * p_max +
            self.alpha_rand * p_rand
        )

        col_sums = self.sensor_model_table.sum(axis=0)
        col_sums[col_sums == 0] = 1.0
        self.sensor_model_table = self.sensor_model_table / col_sums

    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar. THIS IS Z_K. Each range in Z_K is Z_K^i

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """

        if not self.map_set:
            return

        ####################################
        # Ray trace from all particles to get expected scans (N x num_beams)
        scans = self.scan_sim.scan(particles)

        # Convert from meters to pixels
        scale = self.map_resolution * self.lidar_scale_to_map_scale
        z_max = self.table_width - 1

        # Scale the observed and expected lidar readings
        obs_pixels = np.clip(observation / scale, 0, z_max)
        scans_pixels = np.clip(scans / scale, 0, z_max)

        # Convert to integer indices
        obs_indices = np.clip(obs_pixels.astype(int), 0, z_max)
        scan_indices = np.clip(scans_pixels.astype(int), 0, z_max)

        # Look up precomputed probabilities
        # sensor_model_table[z, d] = P(z | d)
        # obs_indices is the measured z, scan_indices is the ground truth d
        probs = self.sensor_model_table[obs_indices, scan_indices]

        # Multiply over all beams (with log-sum to avoid underflow)
        log_probs = np.sum(np.log(probs + 1e-300), axis=1)
        probabilities = np.exp(log_probs)

        return probabilities
        ####################################

    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double) / 100.
        self.map = np.clip(self.map, 0, 1)

        self.map_resolution = map_msg.info.resolution

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation

        quat = [origin_o.x, origin_o.y, origin_o.z, origin_o.w]
        yaw = R.from_quat(quat).as_euler("xyz")[2]

        origin = (origin_p.x, origin_p.y, yaw)

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
            self.map,
            map_msg.info.height,
            map_msg.info.width,
            map_msg.info.resolution,
            origin,
            0.5)

        self.map_set = True
        print("Map initialized")
