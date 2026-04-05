#!/usr/bin/env python3
"""
Generate synthetic PF data and produce charts via analyze_bag.py.
Simulates a car driving a loop with the PF converging then tracking.
"""

import numpy as np
import csv
import os
import sys

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'synthetic_data')


def generate_ground_truth(duration=60.0, dt=0.05):
    """Generate a figure-8 trajectory."""
    t = np.arange(0, duration, dt)
    scale = 5.0
    freq = 2 * np.pi / duration

    x = scale * np.sin(freq * t)
    y = scale * np.sin(2 * freq * t) / 2

    # Compute theta from velocity direction
    dx = np.gradient(x, dt)
    dy = np.gradient(y, dt)
    theta = np.arctan2(dy, dx)

    return t, x, y, theta


def simulate_pf_estimate(t, gt_x, gt_y, gt_theta,
                         initial_error=2.0, convergence_time=5.0,
                         steady_state_noise=0.08, heading_noise=0.03):
    """
    Simulate a PF estimate that starts with large error,
    converges, then tracks with small noise.
    """
    n = len(t)
    pf_x = np.zeros(n)
    pf_y = np.zeros(n)
    pf_theta = np.zeros(n)

    # Initial offset (simulating particles not yet converged)
    offset_x = np.random.uniform(-initial_error, initial_error)
    offset_y = np.random.uniform(-initial_error, initial_error)
    offset_theta = np.random.uniform(-0.5, 0.5)

    for i in range(n):
        # Exponential convergence toward ground truth
        decay = np.exp(-t[i] / convergence_time)
        noise_scale = steady_state_noise + (initial_error - steady_state_noise) * decay

        pf_x[i] = gt_x[i] + offset_x * decay + np.random.normal(0, noise_scale)
        pf_y[i] = gt_y[i] + offset_y * decay + np.random.normal(0, noise_scale)
        pf_theta[i] = gt_theta[i] + offset_theta * decay + np.random.normal(0, heading_noise + 0.2 * decay)

    return pf_x, pf_y, pf_theta


def write_csv(path, t, gt_x, gt_y, gt_theta, pf_x, pf_y, pf_theta):
    """Write a CSV in data_logger format."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp', 'gt_x', 'gt_y', 'gt_theta',
            'pf_x', 'pf_y', 'pf_theta', 'pos_error', 'heading_error'
        ])
        for i in range(len(t)):
            pos_err = np.sqrt((pf_x[i] - gt_x[i])**2 + (pf_y[i] - gt_y[i])**2)
            head_err = abs(np.arctan2(
                np.sin(pf_theta[i] - gt_theta[i]),
                np.cos(pf_theta[i] - gt_theta[i])
            ))
            writer.writerow([
                f'{t[i]:.4f}', f'{gt_x[i]:.4f}', f'{gt_y[i]:.4f}', f'{gt_theta[i]:.4f}',
                f'{pf_x[i]:.4f}', f'{pf_y[i]:.4f}', f'{pf_theta[i]:.4f}',
                f'{pos_err:.4f}', f'{head_err:.4f}'
            ])


def generate_profile_csv(path, num_particles_list, num_beams_list, samples=200):
    """Generate synthetic runtime profiling data."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['num_particles', 'num_beams', 'motion_model_ms', 'sensor_model_ms', 'resample_ms', 'total_ms'])
        for np_ in num_particles_list:
            for nb in num_beams_list:
                for _ in range(samples):
                    # Realistic scaling: motion ~ O(n), sensor ~ O(n*b), resample ~ O(n)
                    mm = max(0.01, np.random.normal(0.05 * np_ / 200, 0.01))
                    sm = max(0.1, np.random.normal(2.0 * (np_ / 200) * (nb / 100), 0.3))
                    rs = max(0.01, np.random.normal(0.1 * np_ / 200, 0.02))
                    writer.writerow([np_, nb, f'{mm:.3f}', f'{sm:.3f}', f'{rs:.3f}', f'{mm+sm+rs:.3f}'])


def main():
    np.random.seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating ground truth trajectory...")
    t, gt_x, gt_y, gt_theta = generate_ground_truth()

    # --- Run configs for noise robustness comparison ---
    configs = {
        'no_noise':       {'steady_state_noise': 0.05, 'heading_noise': 0.02, 'convergence_time': 3.0},
        'low_noise':      {'steady_state_noise': 0.10, 'heading_noise': 0.04, 'convergence_time': 5.0},
        'medium_noise':   {'steady_state_noise': 0.25, 'heading_noise': 0.08, 'convergence_time': 8.0},
        'high_noise':     {'steady_state_noise': 0.50, 'heading_noise': 0.15, 'convergence_time': 12.0},
    }

    csv_paths = []
    for name, params in configs.items():
        print(f"  Simulating PF run: {name}")
        pf_x, pf_y, pf_theta = simulate_pf_estimate(
            t, gt_x, gt_y, gt_theta,
            initial_error=2.0,
            **params
        )
        path = os.path.join(OUTPUT_DIR, f'{name}.csv')
        write_csv(path, t, gt_x, gt_y, gt_theta, pf_x, pf_y, pf_theta)
        csv_paths.append(path)

    # --- Runtime profiling data ---
    print("Generating runtime profiling data...")
    profile_path = os.path.join(OUTPUT_DIR, 'profile_data.csv')
    generate_profile_csv(
        profile_path,
        num_particles_list=[50, 100, 200, 500, 1000],
        num_beams_list=[25, 50, 100],
    )

    print(f"\nSynthetic data written to {OUTPUT_DIR}/")
    print("Files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {f}")

    return csv_paths, profile_path


if __name__ == '__main__':
    csv_paths, profile_path = main()
