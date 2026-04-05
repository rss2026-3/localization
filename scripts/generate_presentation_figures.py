#!/usr/bin/env python3
"""
Generate presentation figures for MCL Lab 5.

Standalone script — no ROS dependency. Produces:
  1. Sensor model heatmap (P(z|d) lookup table)
  2. Sensor model component breakdown (1D slice)
  3. Particle initialization & convergence snapshots on map
  4. Motion model propagation fan

Usage:
    python3 scripts/generate_presentation_figures.py

Output goes to presentation_figures/
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.image as mpimg

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(REPO_DIR, "presentation_figures")
MAP_IMG_PATH = os.path.join(REPO_DIR, "maps", "stata_basement.png")

# Map metadata from stata_basement.yaml
MAP_RESOLUTION = 0.0504  # m/pixel
MAP_ORIGIN = np.array([25.9, 48.5])  # world coords of pixel (0, 0)
MAP_ORIGIN_YAW = 3.14  # ~pi, map is rotated 180 deg

# Sensor model parameters (from sensor_model.py)
ALPHA_HIT = 0.74
ALPHA_SHORT = 0.07
ALPHA_MAX = 0.07
ALPHA_RAND = 0.12
SIGMA_HIT = 8.0
TABLE_WIDTH = 201

# Motion model noise stds (from motion_model.py)
ODOM_NOISE_STDS = np.array([0.1, 0.1, 0.05])


def precompute_sensor_model():
    """Reproduce sensor model table computation (no ROS needed)."""
    z_max = TABLE_WIDTH - 1
    z = np.arange(TABLE_WIDTH)
    d = np.arange(TABLE_WIDTH)
    z_grid, d_grid = np.meshgrid(z, d, indexing='ij')

    # p_hit: Gaussian centered at d
    p_hit = np.exp(-0.5 * ((z_grid - d_grid) / SIGMA_HIT) ** 2)
    col_sums = p_hit.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    p_hit = p_hit / col_sums

    # p_short: linearly decreasing for z <= d
    p_short = np.zeros_like(z_grid, dtype=float)
    valid = (d_grid > 0) & (z_grid <= d_grid) & (z_grid >= 0)
    p_short[valid] = (2.0 / d_grid[valid]) * (1.0 - z_grid[valid] / d_grid[valid])

    # p_max: spike at z_max
    p_max = np.zeros_like(z_grid, dtype=float)
    p_max[z_grid == z_max] = 1.0

    # p_rand: uniform
    p_rand = np.ones_like(z_grid, dtype=float) / z_max

    # Combined table
    table = (ALPHA_HIT * p_hit + ALPHA_SHORT * p_short +
             ALPHA_MAX * p_max + ALPHA_RAND * p_rand)
    col_sums = table.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    table = table / col_sums

    return table, p_hit, p_short, p_max, p_rand


def motion_model_evaluate(particles, odometry):
    """Reproduce motion model (stochastic, no ROS needed)."""
    dx, dy, dtheta = odometry
    n = particles.shape[0]
    magnitude = np.sqrt(dx**2 + dy**2 + dtheta**2) + 1e-6
    noise = np.random.normal(0, ODOM_NOISE_STDS * magnitude, size=(n, 3))
    dx = dx + noise[:, 0]
    dy = dy + noise[:, 1]
    dtheta = dtheta + noise[:, 2]

    cos_theta = np.cos(particles[:, 2])
    sin_theta = np.sin(particles[:, 2])
    particles[:, 0] += cos_theta * dx - sin_theta * dy
    particles[:, 1] += sin_theta * dx + cos_theta * dy
    particles[:, 2] += dtheta
    return particles


def figure1_sensor_model_heatmap(table):
    """Plot the P(z|d) lookup table as a heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(table, origin='lower', aspect='auto',
                   cmap='inferno', interpolation='nearest',
                   vmin=0, vmax=0.03)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("P(z | d)", fontsize=13)
    ax.set_xlabel("True Range d (table index)", fontsize=13)
    ax.set_ylabel("Measured Range z (table index)", fontsize=13)
    ax.set_title("Sensor Model Lookup Table", fontsize=15, fontweight='bold')

    # Annotate regions
    ax.text(30, 180, r"$p_{\mathrm{max}}$", color='white', fontsize=14,
            fontweight='bold', ha='center')
    ax.text(150, 150, r"$p_{\mathrm{hit}}$", color='white', fontsize=14,
            fontweight='bold', ha='center')
    ax.text(150, 50, r"$p_{\mathrm{short}}$", color='white', fontsize=14,
            fontweight='bold', ha='center')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "sensor_model_heatmap.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def figure2_sensor_model_components(p_hit, p_short, p_max, p_rand):
    """Plot 1D slice at fixed d showing all four components."""
    d_val = 120  # slice at this true range
    z = np.arange(TABLE_WIDTH)

    hit = ALPHA_HIT * p_hit[:, d_val]
    short = ALPHA_SHORT * p_short[:, d_val]
    mx = ALPHA_MAX * p_max[:, d_val]
    rand = ALPHA_RAND * p_rand[:, d_val]
    combined = hit + short + mx + rand

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(z, 0, hit, alpha=0.5, color='#2196F3', label=r'$\alpha_{\mathrm{hit}} \cdot p_{\mathrm{hit}}$ (74%)')
    ax.fill_between(z, 0, short, alpha=0.5, color='#FF9800', label=r'$\alpha_{\mathrm{short}} \cdot p_{\mathrm{short}}$ (7%)')
    ax.fill_between(z, 0, rand, alpha=0.4, color='#4CAF50', label=r'$\alpha_{\mathrm{rand}} \cdot p_{\mathrm{rand}}$ (12%)')
    # p_max is a spike at z=200, mark it
    ax.axvline(x=TABLE_WIDTH - 1, color='#F44336', linestyle='--', linewidth=1.5,
               label=r'$\alpha_{\mathrm{max}} \cdot p_{\mathrm{max}}$ (7%)')
    ax.plot(z, combined, 'k-', linewidth=1.5, label='Combined P(z | d)')

    ax.set_xlabel("Measured Range z", fontsize=13)
    ax.set_ylabel("Probability Density", fontsize=13)
    ax.set_title(f"Sensor Model Components (d = {d_val})", fontsize=15, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.set_xlim(0, TABLE_WIDTH - 1)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "sensor_model_components.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def world_to_pixel(xy):
    """Convert world coordinates to pixel coordinates on the Stata map.

    The map origin is at pixel (0, height-1) in image coords with yaw ~pi,
    meaning the map x-axis points left and y-axis points down in pixel space.
    """
    # With yaw = pi: pixel = (origin - world) / resolution
    px = (MAP_ORIGIN[0] - xy[:, 0]) / MAP_RESOLUTION
    py = (MAP_ORIGIN[1] - xy[:, 1]) / MAP_RESOLUTION
    return px, py


def figure3_particle_convergence():
    """Show particle cloud initializing and converging on the map."""
    map_img = mpimg.imread(MAP_IMG_PATH)
    h, w = map_img.shape[:2]

    np.random.seed(42)
    num_particles = 200

    # Use coordinates from actual trajectory (around the loop in Stata basement)
    # Robot operates around x=[-5.5, -3.5], y=[-2.5, 0]
    true_x, true_y, true_theta = -4.5, -0.2, -1.5

    # Stage A: Initial spread (wide Gaussian around initial pose, sigma=0.5m)
    init_particles = np.zeros((num_particles, 3))
    init_particles[:, 0] = true_x + np.random.normal(0, 0.5, num_particles)
    init_particles[:, 1] = true_y + np.random.normal(0, 0.5, num_particles)
    init_particles[:, 2] = true_theta + np.random.normal(0, 0.2, num_particles)

    # Stage B: After several update cycles (tighter cloud, shifted by motion)
    mid_particles = np.zeros((num_particles, 3))
    mid_x, mid_y = true_x + 0.3, true_y - 0.4
    mid_particles[:, 0] = mid_x + np.random.normal(0, 0.15, num_particles)
    mid_particles[:, 1] = mid_y + np.random.normal(0, 0.15, num_particles)
    mid_particles[:, 2] = true_theta + np.random.normal(0, 0.08, num_particles)

    # Stage C: Converged (very tight cluster)
    conv_particles = np.zeros((num_particles, 3))
    conv_x, conv_y = true_x + 0.6, true_y - 0.8
    conv_particles[:, 0] = conv_x + np.random.normal(0, 0.05, num_particles)
    conv_particles[:, 1] = conv_y + np.random.normal(0, 0.05, num_particles)
    conv_particles[:, 2] = (true_theta - 0.1) + np.random.normal(0, 0.01, num_particles)

    stages = [
        (init_particles, "A: Initialization\n($\\sigma_{xy}$=0.5m)", '#FF5722'),
        (mid_particles, "B: After Updates\n($\\sigma_{xy}$=0.15m)", '#FF9800'),
        (conv_particles, "C: Converged\n($\\sigma_{xy}$=0.05m)", '#4CAF50'),
    ]

    # Compute a shared view window centered on the mean of all three clouds
    all_xy = np.vstack([s[0][:, :2] for s in stages])
    center_px, center_py = world_to_pixel(np.array([[all_xy[:, 0].mean(), all_xy[:, 1].mean()]]))
    center_px, center_py = float(center_px[0]), float(center_py[0])
    # Zoom level: ±250 pixels (~12.5m each side) to show corridor walls
    span = 250

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, (particles, title, color) in zip(axes, stages):
        ax.imshow(map_img, cmap='gray', extent=[0, w, h, 0])
        px, py = world_to_pixel(particles[:, :2])
        ax.scatter(px, py, s=10, c=color, alpha=0.7, zorder=5, edgecolors='none')

        # Draw heading arrows for a subset
        arrow_len = 10
        for i in range(0, num_particles, 8):
            adx = -arrow_len * np.cos(particles[i, 2])
            ady = -arrow_len * np.sin(particles[i, 2])
            ax.annotate('', xy=(px[i] + adx, py[i] + ady), xytext=(px[i], py[i]),
                        arrowprops=dict(arrowstyle='->', color=color, lw=0.7))

        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlim(center_px - span, center_px + span)
        ax.set_ylim(center_py + span, center_py - span)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Particle Filter Convergence on Map",
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "particle_convergence.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def figure4_motion_model_fan():
    """Show a single pose propagated into a cloud by the noisy motion model."""
    np.random.seed(7)
    num_particles = 200

    # Start all particles at the same pose
    start_x, start_y, start_theta = 0.0, 0.0, 0.0
    particles = np.tile([start_x, start_y, start_theta], (num_particles, 1))

    # Apply a forward odometry step
    odometry = np.array([0.5, 0.0, 0.15])  # move forward 0.5m, slight turn
    propagated = motion_model_evaluate(particles.copy(), odometry)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor('#f5f5f5')

    # Plot propagated particles
    ax.scatter(propagated[:, 0], propagated[:, 1], s=12, c='#2196F3',
               alpha=0.5, zorder=4, label='Propagated Particles')

    # Draw heading arrows for a subset of propagated particles
    arrow_len = 0.04
    for i in range(0, num_particles, 5):
        adx = arrow_len * np.cos(propagated[i, 2])
        ady = arrow_len * np.sin(propagated[i, 2])
        ax.annotate('', xy=(propagated[i, 0] + adx, propagated[i, 1] + ady),
                    xytext=(propagated[i, 0], propagated[i, 1]),
                    arrowprops=dict(arrowstyle='->', color='#1565C0', lw=0.6))

    # Draw the starting pose as a large arrow
    ax.annotate('', xy=(start_x + 0.08, start_y),
                xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle='->', color='#F44336', lw=3))
    ax.scatter([start_x], [start_y], s=100, c='#F44336', zorder=6,
               marker='o', edgecolors='black', linewidths=1, label='Initial Pose')

    # Draw the deterministic result (no noise)
    det_x = start_x + np.cos(start_theta) * odometry[0]
    det_y = start_y + np.sin(start_theta) * odometry[0]
    ax.scatter([det_x], [det_y], s=80, c='#FF9800', zorder=6,
               marker='D', edgecolors='black', linewidths=1, label='Deterministic Result')

    ax.set_xlabel("x (m)", fontsize=13)
    ax.set_ylabel("y (m)", fontsize=13)
    ax.set_title("Motion Model: Noise Proportional to Odometry", fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "motion_model_fan.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Generating presentation figures...\n")

    print("[1/4] Sensor model heatmap")
    table, p_hit, p_short, p_max, p_rand = precompute_sensor_model()
    figure1_sensor_model_heatmap(table)

    print("[2/4] Sensor model components")
    figure2_sensor_model_components(p_hit, p_short, p_max, p_rand)

    print("[3/4] Particle convergence snapshots")
    figure3_particle_convergence()

    print("[4/4] Motion model propagation fan")
    figure4_motion_model_fan()

    print(f"\nDone! All figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
