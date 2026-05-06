"""
Clean Particle Processing Script
Processes both F2DS probes (V and H) and exports filtered particle images
"""

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths
DATA_DIR = Path('Data')
PBP_FILES = [
    DATA_DIR / '20250524_022503_F2DS_V.pbp.nc',
    DATA_DIR / '20250524_022503_F2DS_H.pbp.nc'
]
ENV_FILE = DATA_DIR / 'RF02.20250524.004127_075522.PNI.nc'

# Output directories
OUTPUT_BASE = Path('particle_images_filtered')
LIQUID_DIR = OUTPUT_BASE / 'liquid'
SOLID_DIR = OUTPUT_BASE / 'solid'

# Create output directories
for dir_path in [LIQUID_DIR, SOLID_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Temperature thresholds for phase classification
LIQUID_TEMP_THRESHOLD = 1.0      # ATX >= 1°C = liquid
SOLID_TEMP_THRESHOLD = -40.0     # ATX <= -40°C = solid

# ============================================================================
# QUALITY FILTER PARAMETERS
# ============================================================================

# Size filters
MIN_DIAMETER = 100          # microns
MIN_XEXTENT = 10           # microns

AREARATIO_MAX = 0.95
ASPECTRATIO_MIN = 0.90
ASPECTRATIO_LINE_MAX = 0.2
VOID_THRESHOLD = 0.05 # Max allowable void fraction (e.g., 10%)

# Rectangle detection (improved)
MIN_ASPECT_RATIO = 0.05    # Reject perfect rectangles
MAX_ASPECT_RATIO = 0.97    # Reject near-perfect rectangles
MIN_AREARATIO_FILLED = 0.90  # Rectangle fill ratio threshold

# Out-of-focus donut detection
DONUT_MIN_SIZE = 200       # Only check large particles
DONUT_MAX_AREARATIO = 0.80 # Hollow threshold
DONUT_MAX_ASPECT = 0.25    # Circular shape

# Edge and quality flags
ALLOW_EDGE_TOUCH = False   # Reject particles touching edges
REQUIRE_ALLIN = True       # Only fully in field of view
REQUIRE_CENTERIN = True    # Only particles with center in view
REQUIRE_DOF = True         # Only in depth of field
MAX_REJECTION_FLAG = 0     # 0 = accepted by probe

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_pbp_data(pbp_file):
    """Load and prepare particle-by-particle data"""
    print(f"\nLoading {pbp_file.name}...")
    ds = xr.open_dataset(pbp_file, decode_times=False)

    # Add UTC time coordinate
    origin = np.datetime64('2025-05-24T00:00:00')
    utc_times = origin + ds['probetime'].astype('timedelta64[s]')
    ds['Time'] = utc_times

    # Add sequential particle index
    ds = ds.assign_coords(particle_idx_seq=("Time", np.arange(ds.dims["Time"])))

    print(f"  Loaded {ds.dims['Time']:,} particles")
    return ds


def get_phase_cutoff_times(env_file):
    """Get temperature cutoff times for liquid/solid classification"""
    print("\nDetermining phase cutoff times from environmental data...")
    ds_env = xr.open_dataset(env_file, decode_times=True)

    # Liquid cutoff (last time where temp >= threshold)
    liquid_mask = (ds_env.ATX >= LIQUID_TEMP_THRESHOLD)
    liquid_cutoff = ds_env.isel(Time=liquid_mask.values)['Time'].max().values

    # Solid cutoff (first time where temp <= threshold)
    solid_mask = (ds_env.ATX <= SOLID_TEMP_THRESHOLD)
    solid_cutoff = ds_env.isel(Time=solid_mask.values)['Time'].min().values

    print(f"  Liquid cutoff time: {liquid_cutoff}")
    print(f"  Solid cutoff time:  {solid_cutoff}")

    return liquid_cutoff, solid_cutoff

def create_exclusion_mask(ds,
                          arearatio_max=AREARATIO_MAX,
                          aspectratio_min=ASPECTRATIO_MIN,
                          aspectratio_line_max=ASPECTRATIO_LINE_MAX,
                          void_threshold=VOID_THRESHOLD,
                          size_threshold=100,
                          diodegaps_thresh=2):
    """
    Build a boolean exclusion mask (True = exclude) for particles in `ds`.
    Returns an xarray.DataArray aligned with ds['Time'].
    """
    # Donut / hollow / out-of-focus
    donut_mask = (ds['diodegaps'] > diodegaps_thresh)

    # Near-perfect rectangles / squares
    square_mask = (ds['arearatiofilled'] > arearatio_max) & (ds['aspectratio'] > aspectratio_min)

    # Line-like / very long rectangles
    line_mask = ds['aspectratio'] < aspectratio_line_max

    rect_mask = (
            (ds['arearatiofilled'] >= MIN_AREARATIO_FILLED) &  # Fills bounding box
            (ds['aspectratio'] < 0.15)                          # Long/thin
        )
    # Calculate the Void Index (Area of the Void / Total Filled Area)
    void_index = (ds['areafilled'] - ds['area']) / ds['areafilled']

    # Create a mask to REJECT particles with a high void index
    # We use >= to ensure the filter works correctly
    donut_mask = void_index >= void_threshold

    # Small particles (size cutoff)
    size_mask = ds['diam'] <= size_threshold

    exclusion_mask = donut_mask | square_mask | line_mask | size_mask | rect_mask
    return exclusion_mask


def apply_quality_filters(ds):
    """
    Apply comprehensive quality filters to remove artifacts.

    Filters out:
    - Particles too small or too large
    - Perfect rectangles (edge artifacts)
    - Out-of-focus donuts
    - Edge-touching particles
    - Particles flagged by probe
    """
    print("\nApplying quality filters...")

    # Start with all particles
    n_original = ds.dims['Time']

    # Build filter mask
    mask = create_exclusion_mask(ds)

    # Apply filter
    ds_filtered = ds.isel(Time=~mask.values)
    n_filtered = ds_filtered.dims['Time']

    print(f"  Original particles:  {n_original:,}")
    print(f"  After filtering:     {n_filtered:,}")
    print(f"  Removed:             {n_original - n_filtered:,} ({(n_original - n_filtered)/n_original*100:.1f}%)")

    return ds_filtered


def separate_by_phase(ds, liquid_cutoff, solid_cutoff):
    """Separate particles into liquid and solid based on temperature"""
    print("\nSeparating particles by phase...")

    # Liquid particles (temp >= threshold)
    liquid_mask = (ds['Time'] <= liquid_cutoff)
    liquid_particles = ds.isel(Time=liquid_mask.values)
    liquid_particles['phase'] = 0

    # Solid particles (temp <= threshold)
    solid_mask = (ds['Time'] >= solid_cutoff)
    solid_particles = ds.isel(Time=solid_mask.values)
    solid_particles['phase'] = 1

    print(f"  Liquid particles: {liquid_particles.dims['Time']:,}")
    print(f"  Solid particles:  {solid_particles.dims['Time']:,}")

    return liquid_particles, solid_particles


def plot_particle_standardized(ds, particle_index, output_dir,
                                target_size=128, max_fit=128):
    """
    Create standardized 128x128 particle image for CNN input.

    Returns clean grayscale image normalized to [0, 1].
    """
    # Get particle number for filename
    pnumber = str(ds['particle_idx_seq'][particle_index].values)

    # Extract image boundaries
    try:
        start_slice = int(ds['starty'].values[particle_index])
        stop_slice = int(ds['stopy'].values[particle_index])
        start_diode = int(ds['startx'].values[particle_index])
        stop_diode = int(ds['stopx'].values[particle_index])
    except Exception:
        return None

    # Extract and binarize particle image
    cropped_image = ds['image'].values[start_slice:stop_slice, start_diode:stop_diode]
    cropped_image = (cropped_image > 0).astype(np.uint8)

    # Trim empty rows/columns
    rows_with_data = np.any(cropped_image == 1, axis=1)
    cols_with_data = np.any(cropped_image == 1, axis=0)

    if not np.any(rows_with_data) or not np.any(cols_with_data):
        return None

    cropped_image = cropped_image[rows_with_data][:, cols_with_data]
    H_current, W_current = cropped_image.shape

    # Scale preserving aspect ratio
    max_dim = max(H_current, W_current)
    if max_dim == 0:
        return None

    scale_factor = float(max_fit) / float(max_dim)
    new_H = max(1, int(round(H_current * scale_factor)))
    new_W = max(1, int(round(W_current * scale_factor)))

    # Resize using nearest-neighbor
    resized_image = resize(
        cropped_image,
        (new_H, new_W),
        order=0,
        anti_aliasing=False,
        preserve_range=True
    )
    resized_image = (resized_image > 0.5).astype(np.uint8)

    # Center on canvas
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    pad_y = (target_size - new_H) // 2
    pad_x = (target_size - new_W) // 2
    canvas[pad_y:pad_y + new_H, pad_x:pad_x + new_W] = resized_image

    # Invert colors (black particles on white background)
    canvas = 1 - canvas

    # Normalize to [0, 1]
    canvas_f = canvas.astype(np.float32)

    # Save as PNG
    out_img = (canvas_f * 255).astype(np.uint8)
    plt.imsave(output_dir / f'particle_{pnumber}.png', out_img, cmap='gray', vmin=0, vmax=255)

    return canvas_f


def export_particles(particles, output_dir, phase_name):
    """Export all particles in dataset to images"""
    print(f"\nExporting {phase_name} particles...")
    n_particles = particles.dims['Time']

    exported = 0
    for i in range(n_particles):
        result = plot_particle_standardized(particles, i, output_dir)
        if result is not None:
            exported += 1

        # Progress indicator
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1:,} / {n_particles:,} particles")

    print(f"  Successfully exported {exported:,} {phase_name} particle images")
    return exported


def combine_particles_from_probes(particle_lists, probe_names):
    """
    Combine particles from multiple probes with unique sequential IDs.
    
    Since different probes have different Slice dimensions, we can't use xr.concat.
    Instead, we process each probe separately and combine at the dataframe level.
    """
    all_data = []
    unique_id = 0
    
    for probe_idx, (particles, probe_name) in enumerate(zip(particle_lists, probe_names)):
        print(f"  Processing {probe_name}: {particles.dims['Time']:,} particles")
        
        # Convert to dataframe (drops the large image array)
        df = particles[['particle_idx_seq', 'phase', 'diam', 'aspectratio',
                        'arearatio', 'arearatiofilled', 'Time']].to_dataframe().reset_index(drop=True)
        
        # Add probe identifier
        df['probe'] = probe_name
        
        # Add unique global particle ID
        df['global_particle_id'] = range(unique_id, unique_id + len(df))
        unique_id += len(df)
        
        # Store the xarray dataset reference for image export
        df['_xr_dataset'] = [particles.to_dataarray()] * len(df)
        df['_xr_index'] = range(len(df))
        
        all_data.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    return combined_df




def export_particles_from_df(df, output_dir, phase_name):
    """Export particles from dataframe (multiple probes combined)"""
    print(f"\nExporting {phase_name} particles...")
    n_particles = len(df)
    
    exported = 0
    for idx, row in df.iterrows():
        # Get the xarray dataset and index for this particle
        ds = row['_xr_dataset']
        particle_idx = row['_xr_index']
        global_id = row['global_particle_id']
        
        # Export with global ID as filename
        result = plot_particle_standardized_with_id(ds, particle_idx, global_id, output_dir)
        if result is not None:
            exported += 1
        
        # Progress indicator
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1:,} / {n_particles:,} particles")
    
    print(f"  Successfully exported {exported:,} {phase_name} particle images")
    return exported


def plot_particle_standardized_with_id(ds, particle_index, global_id, output_dir,
                                        target_size=128, max_fit=128):
    """
    Create standardized 128x128 particle image using global unique ID.
    """
    # Extract image boundaries
    try:
        start_slice = int(ds['starty'].values[particle_index])
        stop_slice = int(ds['stopy'].values[particle_index])
        start_diode = int(ds['startx'].values[particle_index])
        stop_diode = int(ds['stopx'].values[particle_index])
    except Exception:
        return None

    # Extract and binarize particle image
    cropped_image = ds['image'].values[start_slice:stop_slice, start_diode:stop_diode]
    cropped_image = (cropped_image > 0).astype(np.uint8)

    # Trim empty rows/columns
    rows_with_data = np.any(cropped_image == 1, axis=1)
    cols_with_data = np.any(cropped_image == 1, axis=0)

    if not np.any(rows_with_data) or not np.any(cols_with_data):
        return None

    cropped_image = cropped_image[rows_with_data][:, cols_with_data]
    H_current, W_current = cropped_image.shape

    # Scale preserving aspect ratio
    max_dim = max(H_current, W_current)
    if max_dim == 0:
        return None

    scale_factor = float(max_fit) / float(max_dim)
    new_H = max(1, int(round(H_current * scale_factor)))
    new_W = max(1, int(round(W_current * scale_factor)))

    # Resize using nearest-neighbor
    resized_image = resize(
        cropped_image,
        (new_H, new_W),
        order=0,
        anti_aliasing=False,
        preserve_range=True
    )
    resized_image = (resized_image > 0.5).astype(np.uint8)

    # Center on canvas
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    pad_y = (target_size - new_H) // 2
    pad_x = (target_size - new_W) // 2
    canvas[pad_y:pad_y + new_H, pad_x:pad_x + new_W] = resized_image

    # Invert colors (black particles on white background)
    canvas = 1 - canvas

    # Normalize to [0, 1]
    canvas_f = canvas.astype(np.float32)

    # Save as PNG with global unique ID
    out_img = (canvas_f * 255).astype(np.uint8)
    plt.imsave(output_dir / f'particle_{global_id}.png', out_img, cmap='gray', vmin=0, vmax=255)

    return canvas_f

def create_metadata_csv(liquid_particles, solid_particles, output_file='particle_metadata.csv'):
    """Create CSV with particle metadata for CNN training"""
    print(f"\nCreating metadata CSV: {output_file}")

    # Convert to dataframes
    df_liquid = liquid_particles[['particle_idx_seq', 'phase', 'diam', 'aspectratio',
                                   'arearatio', 'arearatiofilled']].to_dataframe()
    df_solid = solid_particles[['particle_idx_seq', 'phase', 'diam', 'aspectratio',
                                 'arearatio', 'arearatiofilled']].to_dataframe()

    # Combine and save
    df_all = pd.concat([df_liquid, df_solid]).reset_index()
    df_all.to_csv(output_file, index=False)

    print(f"  Saved {len(df_all):,} particle records")
    return df_all


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    """Main processing pipeline"""
    print("=" * 70)
    print("PARTICLE DATA PROCESSING PIPELINE")
    print("=" * 70)

    # Load environmental data to get phase cutoffs
    liquid_cutoff, solid_cutoff = get_phase_cutoff_times(ENV_FILE)

    # Process each probe file
    all_liquid_particles = []
    all_solid_particles = []

    for pbp_file in PBP_FILES:
        print(f"\n{'='*70}")
        print(f"Processing {pbp_file.name}")
        print(f"{'='*70}")

        # Load data
        ds = load_pbp_data(pbp_file)

        # Apply quality filters
        ds_filtered = apply_quality_filters(ds)

        # Separate by phase
        liquid, solid = separate_by_phase(ds_filtered, liquid_cutoff, solid_cutoff)

        all_liquid_particles.append(liquid)
        all_solid_particles.append(solid)

    # Combine particles from both probes
    print(f"\n{'='*70}")
    print("COMBINING DATA FROM BOTH PROBES")
    print(f"{'='*70}")
    
    probe_names = [f.stem for f in PBP_FILES]
    
    # Combine liquid particles
    print("\nCombining liquid particles:")
    df_liquid = combine_particles_from_probes(all_liquid_particles, probe_names)
    
    # Combine solid particles  
    print("\nCombining solid particles:")
    df_solid = combine_particles_from_probes(all_solid_particles, probe_names)
    
    print(f"\nTotal liquid particles: {len(df_liquid):,}")
    print(f"Total solid particles:  {len(df_solid):,}")

    # Export images using dataframe
    liquid_count = export_particles_from_df(df_liquid, LIQUID_DIR, 'liquid')
    solid_count = export_particles_from_df(df_solid, SOLID_DIR, 'solid')

    # Create metadata CSV
    df_metadata = pd.concat([df_liquid, df_solid], ignore_index=True)
    # Drop internal xarray references before saving
    df_metadata_clean = df_metadata.drop(columns=['_xr_dataset', '_xr_index'])
    df_metadata_clean.to_csv('particle_metadata.csv', index=False)
    print(f"\nSaved {len(df_metadata_clean):,} particle records to particle_metadata.csv")

    # Summary
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Liquid particles exported: {liquid_count:,}")
    print(f"Solid particles exported:  {solid_count:,}")
    print(f"Total particles:           {liquid_count + solid_count:,}")
    print(f"\nOutput directories:")
    print(f"  Liquid: {LIQUID_DIR}")
    print(f"  Solid:  {SOLID_DIR}")
    print(f"  Metadata: particle_metadata.csv")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
