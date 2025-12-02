#!/usr/bin/env python3
"""
Script to update particle phases based on directory classification.
Assigns phase 2 to particles in 'donut' directory and phase 3 to particles in 'noise' directory.
"""

import pandas as pd
import os
from pathlib import Path

def extract_particle_number(filename):
    """
    Extract particle number from filename like 'particle_123.png'

    Args:
        filename: Image filename

    Returns:
        int: Particle number or None if parsing fails
    """
    try:
        # Remove extension and 'particle_' prefix
        num_str = filename.replace('particle_', '').replace('.png', '')
        return int(num_str)
    except (ValueError, AttributeError):
        return None

def get_particle_numbers_from_directory(directory_path):
    """
    Get all particle numbers from a directory.

    Args:
        directory_path: Path to directory containing particle images

    Returns:
        set: Set of particle numbers found in the directory
    """
    particle_numbers = set()

    if not os.path.exists(directory_path):
        print(f"Warning: Directory {directory_path} does not exist")
        return particle_numbers

    for filename in os.listdir(directory_path):
        if filename.endswith('.png'):
            particle_num = extract_particle_number(filename)
            if particle_num is not None:
                particle_numbers.add(particle_num)

    return particle_numbers

def update_particle_phases(csv_path, base_dir):
    """
    Update particle phases based on directory classification.

    Args:
        csv_path: Path to particle_phases.csv
        base_dir: Base directory containing donut and noise subdirectories
    """
    # Read the CSV
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} particles")

    # Get particle numbers from each directory
    donut_dir = os.path.join(base_dir, 'donut')
    noise_dir = os.path.join(base_dir, 'noise')

    print(f"\nScanning {donut_dir}...")
    donut_particles = get_particle_numbers_from_directory(donut_dir)
    print(f"Found {len(donut_particles)} particles in donut directory")

    print(f"\nScanning {noise_dir}...")
    noise_particles = get_particle_numbers_from_directory(noise_dir)
    print(f"Found {len(noise_particles)} particles in noise directory")

    # Update phases
    print("\nUpdating phases...")
    donut_updated = 0
    noise_updated = 0

    for idx, row in df.iterrows():
        particle_idx = row['particle_idx_seq']

        if particle_idx in donut_particles:
            df.at[idx, 'phase'] = 2
            donut_updated += 1
        elif particle_idx in noise_particles:
            df.at[idx, 'phase'] = 3
            noise_updated += 1

    print(f"Updated {donut_updated} particles to phase 2 (donut)")
    print(f"Updated {noise_updated} particles to phase 3 (noise)")

    # Save the updated CSV
    print(f"\nSaving updated CSV to {csv_path}...")
    df.to_csv(csv_path, index=False)
    print("Done!")

    # Print summary statistics
    print("\n--- Phase Distribution ---")
    phase_counts = df['phase'].value_counts().sort_index()
    for phase, count in phase_counts.items():
        print(f"Phase {phase}: {count} particles")

if __name__ == "__main__":
    # Set paths
    base_dir = "particle_images_filtered"
    csv_path = os.path.join(base_dir, "particle_df.csv")

    # Update phases
    update_particle_phases(csv_path, base_dir)
