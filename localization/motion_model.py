import numpy as np


class MotionModel:

    def __init__(self, node):
        ####################################
        node.declare_parameter('deterministic', False)
        self.deterministic = node.get_parameter('deterministic').get_parameter_value().bool_value

        # Noise standard deviations for non-deterministic mode
        # These scale with the magnitude of the odometry
        node.declare_parameter('odom_noise_x', 0.1)
        node.declare_parameter('odom_noise_y', 0.1)
        node.declare_parameter('odom_noise_theta', 0.05)
        self.odom_noise_stds = np.array([
            node.get_parameter('odom_noise_x').get_parameter_value().double_value,
            node.get_parameter('odom_noise_y').get_parameter_value().double_value,
            node.get_parameter('odom_noise_theta').get_parameter_value().double_value,
        ])
        ####################################

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """

        ####################################
        dx, dy, dtheta = odometry

        if not self.deterministic:
            # Adds noise proportional to odometry magnitude, or at least 1e-6
            n = particles.shape[0]
            magnitude = np.sqrt(dx ** 2 + dy ** 2 + dtheta ** 2) + 1e-6
            noise = np.random.normal(0, self.odom_noise_stds * magnitude, size=(n, 3))
            dx = dx + noise[:, 0]
            dy = dy + noise[:, 1]
            dtheta = dtheta + noise[:, 2]

        cos_theta = np.cos(particles[:, 2])
        sin_theta = np.sin(particles[:, 2])

        # Rotate body-frame odometry into world frame and apply
        particles[:, 0] += cos_theta * dx - sin_theta * dy
        particles[:, 1] += sin_theta * dx + cos_theta * dy
        particles[:, 2] += dtheta

        return particles
        ####################################
