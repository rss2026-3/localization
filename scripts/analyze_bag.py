#!/usr/bin/env python3
"""
Offline rosbag analyzer for particle filter experiments.

Reads ROS2 bag files (without needing ROS installed) using the `rosbags`
pip package, extracts PF estimates and ground truth, and generates charts.

Install:
    pip install rosbags matplotlib numpy

Usage:
    # Single bag analysis
    python3 scripts/analyze_bag.py path/to/bag_dir -o charts/

    # Compare multiple runs (overlay on same axes)
    python3 scripts/analyze_bag.py bags/run1 bags/run2 bags/run3 --compare -o comparison/

    # With map overlay on trajectory plot
    python3 scripts/analyze_bag.py path/to/bag_dir -o charts/ --map maps/stata_basement.png
"""

import argparse
import os
import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not found. Install with: pip install matplotlib")
    sys.exit(1)

HAS_ROSBAGS = False
try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import get_typestore, Stores
    HAS_ROSBAGS = True
except ImportError:
    pass


def extract_poses_from_bag(bag_path):
    """
    Extract PF estimate and ground truth poses from a rosbag.

    Returns:
        pf_data: list of (timestamp_sec, x, y, theta)
        gt_data: list of (timestamp_sec, x, y, theta)
    """
    if not HAS_ROSBAGS:
        print("rosbags not installed. Install with: pip install rosbags")
        sys.exit(1)
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    pf_data = []
    gt_data = []

    with Reader(bag_path) as reader:
        for connection, timestamp, rawdata in reader.messages():
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            t_sec = timestamp * 1e-9  # nanoseconds to seconds

            if connection.topic == '/pf/pose/odom':
                pos = msg.pose.pose.position
                ori = msg.pose.pose.orientation
                theta = quat_to_yaw(ori.x, ori.y, ori.z, ori.w)
                pf_data.append((t_sec, pos.x, pos.y, theta))

            elif connection.topic == '/odom':
                # Ground truth odometry from simulator
                pos = msg.pose.pose.position
                ori = msg.pose.pose.orientation
                theta = quat_to_yaw(ori.x, ori.y, ori.z, ori.w)
                gt_data.append((t_sec, pos.x, pos.y, theta))

    return np.array(pf_data), np.array(gt_data)


def quat_to_yaw(x, y, z, w):
    """Extract yaw from quaternion (no tf_transformations dependency)."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def align_by_time(pf_data, gt_data, max_dt=0.1):
    """
    For each PF sample, find the nearest GT sample within max_dt.
    Returns aligned arrays of equal length.
    """
    aligned_pf = []
    aligned_gt = []
    gt_times = gt_data[:, 0]

    for row in pf_data:
        t = row[0]
        idx = np.argmin(np.abs(gt_times - t))
        if abs(gt_times[idx] - t) < max_dt:
            aligned_pf.append(row)
            aligned_gt.append(gt_data[idx])

    return np.array(aligned_pf), np.array(aligned_gt)


def compute_errors(pf, gt):
    """Compute position and heading errors from aligned arrays."""
    t = pf[:, 0] - pf[0, 0]  # relative time
    pos_err = np.sqrt((pf[:, 1] - gt[:, 1])**2 + (pf[:, 2] - gt[:, 2])**2)
    head_err = np.abs(np.arctan2(
        np.sin(pf[:, 3] - gt[:, 3]),
        np.cos(pf[:, 3] - gt[:, 3])
    ))
    return t, pos_err, head_err


def generate_charts(pf, gt, t, pos_err, head_err, output_dir, label="", map_path=None):
    """Generate all charts for a single run."""
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{label}_" if label else ""

    # 1. Position error over time
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, pos_err, linewidth=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position Error (m)')
    ax.set_title('Position Error Over Time')
    ax.axhline(np.mean(pos_err), color='r', linestyle='--', label=f'mean={np.mean(pos_err):.3f}m')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{prefix}pos_error_time.png'), dpi=150)
    plt.close(fig)

    # 2. Heading error over time
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, np.degrees(head_err), linewidth=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Heading Error (deg)')
    ax.set_title('Heading Error Over Time')
    ax.axhline(np.mean(np.degrees(head_err)), color='r', linestyle='--',
               label=f'mean={np.mean(np.degrees(head_err)):.2f}°')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{prefix}heading_error_time.png'), dpi=150)
    plt.close(fig)

    # 3. Trajectory overlay
    fig, ax = plt.subplots(figsize=(8, 8))
    if map_path and os.path.exists(map_path):
        img = plt.imread(map_path)
        ax.imshow(img, cmap='gray', origin='lower', alpha=0.5)
    ax.plot(gt[:, 1], gt[:, 2], 'b-', linewidth=1.0, label='Ground Truth', alpha=0.7)
    ax.plot(pf[:, 1], pf[:, 2], 'r-', linewidth=1.0, label='PF Estimate', alpha=0.7)
    ax.plot(pf[0, 1], pf[0, 2], 'go', markersize=8, label='Start')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Trajectory: Ground Truth vs PF Estimate')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{prefix}trajectory.png'), dpi=150)
    plt.close(fig)

    # 4. Error histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(pos_err, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(np.median(pos_err), color='r', linestyle='--', label=f'median={np.median(pos_err):.3f}m')
    ax1.axvline(np.percentile(pos_err, 95), color='orange', linestyle='--',
                label=f'95th={np.percentile(pos_err, 95):.3f}m')
    ax1.set_xlabel('Position Error (m)')
    ax1.set_ylabel('Count')
    ax1.set_title('Position Error Distribution')
    ax1.legend()

    ax2.hist(np.degrees(head_err), bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(np.median(np.degrees(head_err)), color='r', linestyle='--',
                label=f'median={np.median(np.degrees(head_err)):.2f}°')
    ax2.set_xlabel('Heading Error (deg)')
    ax2.set_ylabel('Count')
    ax2.set_title('Heading Error Distribution')
    ax2.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{prefix}error_hist.png'), dpi=150)
    plt.close(fig)

    # Summary stats
    print(f"\n{'=' * 50}")
    print(f"Run: {label or 'default'}")
    print(f"  Duration:       {t[-1]:.1f}s ({len(t)} samples)")
    print(f"  Pos error:      mean={np.mean(pos_err):.3f}m, "
          f"median={np.median(pos_err):.3f}m, 95th={np.percentile(pos_err, 95):.3f}m")
    print(f"  Heading error:  mean={np.mean(np.degrees(head_err)):.2f}°, "
          f"median={np.median(np.degrees(head_err)):.2f}°")
    print(f"  Charts saved to {output_dir}/")
    print(f"{'=' * 50}")


def generate_comparison(all_results, output_dir):
    """Generate overlay charts comparing multiple runs."""
    os.makedirs(output_dir, exist_ok=True)

    # Position error comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, (t, pos_err, _) in all_results.items():
        ax.plot(t, pos_err, linewidth=0.8, label=f'{label} (mean={np.mean(pos_err):.3f}m)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position Error (m)')
    ax.set_title('Position Error Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'comparison_pos_error.png'), dpi=150)
    plt.close(fig)

    # Bar chart: mean error per run
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    labels = list(all_results.keys())
    mean_pos = [np.mean(v[1]) for v in all_results.values()]
    mean_head = [np.mean(np.degrees(v[2])) for v in all_results.values()]

    ax1.bar(labels, mean_pos, edgecolor='black', alpha=0.7)
    ax1.set_ylabel('Mean Position Error (m)')
    ax1.set_title('Position Error by Run')
    ax1.tick_params(axis='x', rotation=30)

    ax2.bar(labels, mean_head, edgecolor='black', alpha=0.7, color='orange')
    ax2.set_ylabel('Mean Heading Error (deg)')
    ax2.set_title('Heading Error by Run')
    ax2.tick_params(axis='x', rotation=30)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'comparison_bar.png'), dpi=150)
    plt.close(fig)

    print(f"\nComparison charts saved to {output_dir}/")


def generate_profile_charts(profile_csv, output_dir):
    """Generate runtime scaling charts from profiler CSV."""
    os.makedirs(output_dir, exist_ok=True)
    data = np.genfromtxt(profile_csv, delimiter=',', skip_header=1, dtype=None, encoding='utf-8')

    # Parse into structured arrays
    particles = np.array([int(row[0]) for row in data])
    beams = np.array([int(row[1]) for row in data])
    sensor_ms = np.array([float(row[3]) for row in data])
    total_ms = np.array([float(row[5]) for row in data])

    unique_particles = sorted(set(particles))
    unique_beams = sorted(set(beams))

    # 1. Total update time vs num_particles, grouped by beam count
    fig, ax = plt.subplots(figsize=(9, 5))
    for nb in unique_beams:
        means = []
        stds = []
        for np_ in unique_particles:
            mask = (particles == np_) & (beams == nb)
            means.append(np.mean(total_ms[mask]))
            stds.append(np.std(total_ms[mask]))
        ax.errorbar(unique_particles, means, yerr=stds, marker='o',
                    capsize=4, label=f'{nb} beams')
    ax.set_xlabel('Number of Particles')
    ax.set_ylabel('Total Update Time (ms)')
    ax.set_title('PF Update Time vs Particle Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'runtime_vs_particles.png'), dpi=150)
    plt.close(fig)

    # 2. Stacked bar: motion vs sensor vs resample for each config
    motion_ms = np.array([float(row[2]) for row in data])
    resample_ms = np.array([float(row[4]) for row in data])

    fig, ax = plt.subplots(figsize=(12, 5))
    configs = [(np_, nb) for np_ in unique_particles for nb in unique_beams]
    labels = [f'{np_}p/{nb}b' for np_, nb in configs]
    mm_means = [np.mean(motion_ms[(particles == np_) & (beams == nb)]) for np_, nb in configs]
    sm_means = [np.mean(sensor_ms[(particles == np_) & (beams == nb)]) for np_, nb in configs]
    rs_means = [np.mean(resample_ms[(particles == np_) & (beams == nb)]) for np_, nb in configs]

    x_pos = np.arange(len(configs))
    ax.bar(x_pos, mm_means, label='Motion Model', color='#4C72B0')
    ax.bar(x_pos, sm_means, bottom=mm_means, label='Sensor Model', color='#DD8452')
    ax.bar(x_pos, rs_means, bottom=np.array(mm_means) + np.array(sm_means),
           label='Resample', color='#55A868')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Time (ms)')
    ax.set_title('PF Update Breakdown by Configuration')
    ax.axhline(50, color='red', linestyle='--', alpha=0.5, label='20Hz threshold (50ms)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'runtime_breakdown.png'), dpi=150)
    plt.close(fig)

    # 3. Sensor model time vs num_beams, grouped by particle count
    fig, ax = plt.subplots(figsize=(9, 5))
    for np_ in unique_particles:
        means = []
        for nb in unique_beams:
            mask = (particles == np_) & (beams == nb)
            means.append(np.mean(sensor_ms[mask]))
        ax.plot(unique_beams, means, marker='s', label=f'{np_} particles')
    ax.set_xlabel('Number of Beams per Particle')
    ax.set_ylabel('Sensor Model Time (ms)')
    ax.set_title('Sensor Model Time vs Beam Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'sensor_vs_beams.png'), dpi=150)
    plt.close(fig)

    print(f"Profile charts saved to {output_dir}/")


def analyze_csv(csv_path):
    """Analyze a CSV produced by data_logger.py instead of a bag."""
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    t = data[:, 0] - data[0, 0]
    gt = np.column_stack([data[:, 0], data[:, 1], data[:, 2], data[:, 3]])
    pf = np.column_stack([data[:, 0], data[:, 4], data[:, 5], data[:, 6]])
    pos_err = data[:, 7]
    head_err = data[:, 8]
    return pf, gt, t, pos_err, head_err


def main():
    parser = argparse.ArgumentParser(description='Analyze particle filter rosbags or CSVs')
    parser.add_argument('inputs', nargs='*', help='Bag directories or CSV files')
    parser.add_argument('-o', '--output', default='charts/', help='Output directory')
    parser.add_argument('--compare', action='store_true', help='Overlay multiple runs')
    parser.add_argument('--map', default=None, help='Map image for trajectory overlay')
    parser.add_argument('--gt-topic', default='/odom', help='Ground truth topic in bag')
    parser.add_argument('--profile', default=None, help='Profile CSV to generate runtime charts')
    args = parser.parse_args()

    if args.profile:
        generate_profile_charts(args.profile, args.output)
        if not args.inputs or args.inputs == ['_']:
            return

    all_results = {}

    for input_path in args.inputs:
        label = os.path.basename(input_path.rstrip('/'))

        if input_path.endswith('.csv'):
            pf, gt, t, pos_err, head_err = analyze_csv(input_path)
        else:
            print(f"Reading bag: {input_path}")
            pf_data, gt_data = extract_poses_from_bag(input_path)

            if len(pf_data) == 0:
                print(f"  No /pf/pose/odom messages found, skipping.")
                continue
            if len(gt_data) == 0:
                print(f"  No ground truth messages found on {args.gt_topic}, skipping.")
                continue

            pf, gt = align_by_time(pf_data, gt_data)
            if len(pf) == 0:
                print(f"  Could not align timestamps, skipping.")
                continue

            t, pos_err, head_err = compute_errors(pf, gt)

        if not args.compare:
            generate_charts(pf, gt, t, pos_err, head_err, args.output, label, args.map)
        all_results[label] = (t, pos_err, head_err)

    if args.compare and len(all_results) > 1:
        generate_comparison(all_results, args.output)


if __name__ == '__main__':
    main()
