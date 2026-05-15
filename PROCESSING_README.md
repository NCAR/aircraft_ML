# Particle Data Preprocessing

This document covers how to run the two preprocessing scripts that convert raw netCDF probe files into the image and metadata format expected by the training notebooks.

---

## Scripts

### `process_particle_data.py`

Processes real F2DS probe data (PBP netCDF files). Applies quality filters, extracts standardized 128x128 particle images, and writes a metadata CSV for training.

### `synthetic_preprocessing.py`

Processes synthetic netCDF files simulating 2D-S instrument output. Assigns phase labels based on filename, extracts 128x128 images, and writes a metadata CSV. Does not apply donut or rectangle filters since the synthetic data does not contain those artifact types.

---

## Running the Scripts

### F2DS (real flight data)

Place the raw PBP netCDF files in the `Data/` directory, then run:

```bash
python3 process_particle_data.py
```

This will:
1. Load both `.pbp.nc` probe files (vertical and horizontal orientations) from `Data/`
2. Apply quality filters (size, shape, donut, rectangle, edge)
3. Export 128x128 grayscale PNG images to `particle_images_filtered/liquid/` and `particle_images_filtered/solid/`
4. Write `particle_df.csv` with per-particle metadata and labels

### Synthetic data

Place the synthetic `.nc` files in the `Data/` directory, then run:

```bash
python3 synthetic_preprocessing.py
```

This will:
1. Discover all `.nc` files in `Data/` and assign labels by filename: files containing `Liquid` go to liquid (phase 0), files containing `ice` or `Column` go to ice (phase 1)
2. Export 128x128 grayscale PNG images to `synthetic_particle_images_filtered/liquid/` and `synthetic_particle_images_filtered/ice/`
3. Write `synthetic_particle_metadata.csv`

---

## Output

### F2DS output

Images are saved as:

```
particle_images_filtered/
    liquid/
        particle_0.png
        particle_1.png
        ...
    solid/
        particle_90741.png
        ...
```

`particle_df.csv` columns:

| Column | Description |
| --- | --- |
| `Time` | UTC timestamp |
| `particle_idx_seq` | Particle index — matches the image filename number |
| `phase` | 0 = liquid, 1 = solid |
| `diam` | Diameter (microns) |
| `aspectratio` | Aspect ratio (minor / major axis) |
| `arearatio` | Area ratio (solid vs. filled bounding box) |
| `arearatiofilled` | Filled area ratio |

### Synthetic output

Images are saved as:

```
synthetic_particle_images_filtered/
    liquid/
        particle_0.png
        ...
    ice/
        particle_90741.png
        ...
```

`synthetic_particle_metadata.csv` columns:

| Column | Description |
| --- | --- |
| `Time` | Timestamp, if present in the netCDF file |
| `global_particle_id` | Unique particle ID across all files — matches the image filename number |
| `phase` | 0 = liquid, 1 = ice |
| `phase_name` | `liquid` or `ice` |
| `source_file` | Source netCDF filename |
| `diam` | Diameter (microns) |
| `aspectratio` | Aspect ratio |
| `arearatio` | Area ratio |
| `arearatiofilled` | Filled area ratio |

---

## Quality Filters (F2DS only)

The following filters are applied to the real probe data. A particle is removed if any condition is true:

| Filter | Criterion |
| --- | --- |
| Too small | diameter <= 100 microns |
| Hollow / donut-like | void index >= 0.05 |
| Near-perfect square | arearatiofilled > 0.95 AND aspectratio > 0.90 |
| Line-like | aspectratio < 0.20 |
| Long rectangle | arearatiofilled >= 0.90 AND aspectratio < 0.15 |

These thresholds are defined at the top of `process_particle_data.py` and can be adjusted if needed.

---

## Configuration

All tunable parameters are defined at the top of each script.

### Size filters (`process_particle_data.py`)

```python
MIN_DIAMETER = 100          # microns — particles smaller than this are removed
```

### Rectangle detection

```python
MIN_AREARATIO_FILLED = 0.90 # particles filling >= 90% of bounding box flagged as rectangles
MAX_ASPECT_RATIO     = 0.15 # combined with above to catch long thin artifacts
```

### Donut detection

```python
DONUT_MIN_SIZE      = 200   # only check particles larger than this (microns)
DONUT_MAX_AREARATIO = 0.80  # hollow threshold
DONUT_MAX_ASPECT    = 0.25  # circular shape threshold
```

---

## Tuning the Filters

### Too many rectangular artifacts remaining

Tighten the rectangle filter:

```python
MIN_AREARATIO_FILLED = 0.85  # catch more rectangles (default 0.90)
```

### Good particles being incorrectly removed

Loosen the rectangle filter:

```python
MIN_AREARATIO_FILLED = 0.95  # only catch very rectangular shapes
```

### Donut artifacts passing through

Tighten the donut filter:

```python
DONUT_MAX_AREARATIO = 0.85   # catch less hollow particles (default 0.80)
DONUT_MAX_ASPECT    = 0.30   # catch less circular particles (default 0.25)
```

### Donut filter incorrectly removing ice crystals

Loosen the donut filter:

```python
DONUT_MIN_SIZE      = 300    # only check larger particles (default 200)
DONUT_MAX_AREARATIO = 0.70   # require more hollow to flag (default 0.80)
DONUT_MAX_ASPECT    = 0.20   # require more circular to flag (default 0.25)
```

---

## Inspecting Individual Particles

To check whether a specific particle passes or fails the filters, add the following to the end of the script:

```python
def inspect_particle(ds, idx):
    print(f"\nParticle {idx}:")
    print(f"  Diameter:        {ds['diam'].values[idx]:.1f} um")
    print(f"  Aspect ratio:    {ds['aspectratio'].values[idx]:.3f}")
    print(f"  Area ratio:      {ds['arearatio'].values[idx]:.3f}")
    print(f"  Area ratio fill: {ds['arearatiofilled'].values[idx]:.3f}")

    is_rectangle = (
        (ds['arearatiofilled'].values[idx] >= 0.90) and
        (ds['aspectratio'].values[idx] < 0.15)
    )
    is_donut = (
        (ds['diam'].values[idx] >= 200) and
        (ds['arearatio'].values[idx] < 0.80) and
        (ds['aspectratio'].values[idx] < 0.25)
    )

    if is_rectangle:
        print("  FILTERED: rectangle artifact")
    elif is_donut:
        print("  FILTERED: donut artifact")
    else:
        print("  KEPT")

# inspect_particle(ds, 700)    # example: rectangle
# inspect_particle(ds, 9)      # example: donut
# inspect_particle(ds, 8731)   # example: legitimate ice crystal
```