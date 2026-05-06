"""
Synthetic Particle Preprocessing Script

Processes synthetic NetCDF particle files and exports standardized particle
images plus a metadata CSV.

Phase assignment is based on filename:
- Liquid files -> liquid phase
- Ice / Column files -> ice phase

This version intentionally does not run rectangle detection.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from skimage.transform import resize

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("Data")
OUTPUT_BASE = Path("synthetic_particle_images_filtered")
LIQUID_DIR = OUTPUT_BASE / "liquid"
ICE_DIR = OUTPUT_BASE / "ice"
METADATA_FILE = Path("synthetic_particle_metadata.csv")

TARGET_IMAGE_SIZE = 128
MAX_FIT_SIZE = 128


# ============================================================================
# FILE DISCOVERY
# ============================================================================

def discover_synthetic_files(data_dir):
    """Find synthetic NetCDF files and map them to a phase label."""
    discovered = []

    for file_path in sorted(data_dir.rglob("*.nc")):
        name_lower = file_path.name.lower()

        if "liquid" in name_lower:
            discovered.append((file_path, "liquid", 0, "liquid"))
        elif "column" in name_lower:
            discovered.append((file_path, "ice", 1, "column"))
        elif "ice" in name_lower:
            discovered.append((file_path, "ice", 1, "ice"))

    return discovered


# ============================================================================
# DATA LOADING AND FILTERING
# ============================================================================

def load_particle_dataset(netcdf_file):
    """Load a NetCDF particle dataset and add a sequential particle index."""
    print(f"\nLoading {netcdf_file.name}...")
    ds = xr.open_dataset(netcdf_file, decode_times=False)

    if "Time" not in ds.dims:
        raise ValueError(f"{netcdf_file.name} does not contain a Time dimension")

    if "probetime" in ds:
        origin = np.datetime64("2025-05-24T00:00:00")
        utc_times = origin + ds["probetime"].astype("timedelta64[s]")
        ds = ds.assign_coords(Time=utc_times)

    ds = ds.assign_coords(particle_idx_seq=("Time", np.arange(ds.sizes["Time"])))

    print(f"  Loaded {ds.sizes['Time']:,} particles")
    return ds


# ============================================================================
# IMAGE EXPORT
# ============================================================================

def plot_particle_standardized_with_id(ds, particle_index, global_id, output_dir,
                                       target_size=TARGET_IMAGE_SIZE, max_fit=MAX_FIT_SIZE):
    """Create a standardized particle image and save it with a global ID."""
    try:
        start_slice = int(ds["starty"].values[particle_index])
        stop_slice = int(ds["stopy"].values[particle_index])
        start_diode = int(ds["startx"].values[particle_index])
        stop_diode = int(ds["stopx"].values[particle_index])
    except Exception:
        return None

    cropped_image = ds["image"].values[start_slice:stop_slice, start_diode:stop_diode]
    cropped_image = (cropped_image > 0).astype(np.uint8)

    rows_with_data = np.any(cropped_image == 1, axis=1)
    cols_with_data = np.any(cropped_image == 1, axis=0)

    if not np.any(rows_with_data) or not np.any(cols_with_data):
        return None

    cropped_image = cropped_image[rows_with_data][:, cols_with_data]
    height, width = cropped_image.shape

    max_dim = max(height, width)
    if max_dim == 0:
        return None

    scale_factor = float(max_fit) / float(max_dim)
    new_height = max(1, int(round(height * scale_factor)))
    new_width = max(1, int(round(width * scale_factor)))

    resized_image = resize(
        cropped_image,
        (new_height, new_width),
        order=0,
        anti_aliasing=False,
        preserve_range=True,
    )
    resized_image = (resized_image > 0.5).astype(np.uint8)

    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    pad_y = (target_size - new_height) // 2
    pad_x = (target_size - new_width) // 2
    canvas[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized_image
    canvas = 1 - canvas

    canvas_f = canvas.astype(np.float32)
    out_img = (canvas_f * 255).astype(np.uint8)
    plt.imsave(output_dir / f"particle_{global_id}.png", out_img, cmap="gray", vmin=0, vmax=255)

    return canvas_f


def export_particles(ds, output_dir, start_global_id):
    """Export every particle in a filtered dataset using a unique global ID."""
    print(f"\nExporting particles to {output_dir}...")

    output_dir.mkdir(parents=True, exist_ok=True)

    n_particles = ds.sizes["Time"]
    exported = 0

    for particle_index in range(n_particles):
        global_id = start_global_id + particle_index
        result = plot_particle_standardized_with_id(ds, particle_index, global_id, output_dir)
        if result is not None:
            exported += 1

        if (particle_index + 1) % 500 == 0:
            print(f"  Processed {particle_index + 1:,} / {n_particles:,} particles")

    print(f"  Successfully exported {exported:,} particle images")
    return exported


# ============================================================================
# METADATA
# ============================================================================

def build_metadata_frame(ds, phase_name, phase_value, source_file, start_global_id):
    """Create a metadata frame for a filtered dataset."""
    columns = ["diam", "aspectratio", "arearatio", "arearatiofilled"]
    existing_columns = [column for column in columns if column in ds]

    frame = ds[existing_columns].to_dataframe().reset_index()
    frame["phase"] = phase_value
    frame["phase_name"] = phase_name
    frame["source_file"] = source_file.name
    frame["source_group"] = source_file.name.lower()
    frame["global_particle_id"] = range(start_global_id, start_global_id + len(frame))

    return frame


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    print("=" * 70)
    print("SYNTHETIC PARTICLE PREPROCESSING")
    print("=" * 70)

    for output_dir in [LIQUID_DIR, ICE_DIR]:
        output_dir.mkdir(parents=True, exist_ok=True)

    synthetic_files = discover_synthetic_files(DATA_DIR)
    if not synthetic_files:
        raise FileNotFoundError(
            f"No synthetic NetCDF files found in {DATA_DIR}. Expected names containing Liquid, Column, or ice."
        )

    print("\nDiscovered synthetic inputs:")
    for file_path, phase_name, _, source_group in synthetic_files:
        print(f"  {file_path.name} -> phase={phase_name}, source_group={source_group}")

    all_frames = []
    global_particle_id = 0
    total_exported = 0

    for file_path, phase_name, phase_value, _source_group in synthetic_files:
        print(f"\n{'=' * 70}")
        print(f"Processing {file_path.name}")
        print(f"{'=' * 70}")

        ds_filtered = load_particle_dataset(file_path)

        kept_count = ds_filtered.sizes["Time"]
        print(f"  Particles kept:       {kept_count:,}")

        output_dir = LIQUID_DIR if phase_name == "liquid" else ICE_DIR
        exported = export_particles(ds_filtered, output_dir, global_particle_id)
        total_exported += exported

        frame = build_metadata_frame(ds_filtered, phase_name, phase_value, file_path, global_particle_id)
        all_frames.append(frame)

        global_particle_id += len(frame)

    metadata = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    metadata.to_csv(METADATA_FILE, index=False)

    liquid_count = int((metadata["phase"] == 0).sum()) if not metadata.empty else 0
    ice_count = int((metadata["phase"] == 1).sum()) if not metadata.empty else 0

    print(f"\n{'=' * 70}")
    print("PROCESSING COMPLETE")
    print(f"{'=' * 70}")
    print(f"Liquid particles exported: {liquid_count:,}")
    print(f"Ice particles exported:    {ice_count:,}")
    print(f"Total exported:            {total_exported:,}")
    print(f"Metadata saved to:         {METADATA_FILE}")
    print(f"Output directories:")
    print(f"  Liquid: {LIQUID_DIR}")
    print(f"  Ice:    {ICE_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()