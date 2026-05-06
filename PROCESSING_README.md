# Particle Data Processing Script

## Overview

The `process_particle_data.py` script is a clean, production-ready version of your particle processing pipeline that:

1. ✅ Processes **both F2DS probe files** (V and H orientations)
2. ✅ Applies **improved quality filters** (rectangles, donuts, edge artifacts)
3. ✅ Exports standardized **128×128 particle images**
4. ✅ Creates **metadata CSV** for CNN training
5. ✅ Well-documented and configurable

---

## Quick Start

```bash
cd /Users/srunkel/dev/aircraft_ML
python3 process_particle_data.py
```

The script will:
- Load both `.pbp.nc` files from the `Data/` directory
- Apply quality filters
- Export images to `particle_images_filtered/liquid/` and `particle_images_filtered/solid/`
- Create `particle_metadata.csv`

---

## What's New/Improved

### 1. **Improved Rectangle Filter**

The previous filter missed some rectangles. The new filter detects rectangles using:

```python
# Rectangles have:
# - High arearatiofilled (fill their bounding box)
# - Low aspect ratio (long/thin shape)
~(
    (ds['arearatiofilled'] >= 0.90) &  # Fills 90%+ of bounding box
    (ds['aspectratio'] < 0.15)          # Long/thin
)
```

This catches edge artifacts like particle_700 that are:
- Thin vertical/horizontal lines
- Fill most of their bounding box
- Have very low aspect ratio

### 2. **Smart Donut Filter**

Only filters out-of-focus donuts, not ice crystals with gaps:

```python
# Only reject particles that are ALL of:
~(
    (ds['diam'] >= 200) &           # Large
    (ds['arearatio'] < 0.80) &      # Hollow
    (ds['aspectratio'] < 0.25)      # Circular
)
```

### 3. **Processes Both Probes**

Automatically processes:
- `20250524_022503_F2DS_V.pbp.nc` (Vertical orientation)
- `20250524_022503_F2DS_H.pbp.nc` (Horizontal orientation)

Combines particles from both into single dataset.

### 4. **Clean Code Structure**

- Clear configuration section at top
- Well-documented functions
- Progress indicators
- Comprehensive statistics

---

## Configuration

All parameters are at the top of the script. Easy to tune:

### Size Filters
```python
MIN_DIAMETER = 100          # microns
MAX_DIAMETER = 5000         # microns
MIN_AREA = 50              # pixels
```

### Rectangle Detection
```python
MIN_ASPECT_RATIO = 0.05     # Reject perfect rectangles
MAX_ASPECT_RATIO = 0.97     # Reject near-perfect rectangles
MIN_AREARATIO_FILLED = 0.90 # Rectangle fill ratio
```

### Donut Detection
```python
DONUT_MIN_SIZE = 200        # Only check large particles
DONUT_MAX_AREARATIO = 0.80  # Hollow threshold
DONUT_MAX_ASPECT = 0.25     # Circular shape
```

### Quality Flags
```python
ALLOW_EDGE_TOUCH = False    # Reject edge particles
REQUIRE_ALLIN = True        # Only fully in view
REQUIRE_CENTERIN = True     # Center in view
REQUIRE_DOF = True          # In depth of field
```

---

## Output

### Directory Structure
```
particle_images_filtered/
├── liquid/
│   ├── particle_0.png
│   ├── particle_1.png
│   └── ...
└── solid/
    ├── particle_90741.png
    ├── particle_90742.png
    └── ...
```

### Metadata CSV
`particle_metadata.csv` contains:
- `Time`: Timestamp
- `particle_idx_seq`: Particle number (matches image filename)
- `phase`: 0=liquid, 1=solid
- `diam`: Diameter (microns)
- `aspectratio`: Aspect ratio
- `arearatio`: Area ratio (solid vs. filled)
- `arearatiofilled`: Area ratio filled

---

## Expected Results

### Before Filtering (Per Probe):
- ~3.7 million raw particles
- Many artifacts

### After Filtering (Per Probe):
- ~100,000 quality particles
- ~10,000-12,000 particles after phase separation
- ~97-98% artifacts removed

### Combined (Both Probes):
- ~4,000-5,000 liquid particles
- ~20,000-25,000 solid particles
- Clean, ready for CNN training

---

## Tuning Guide

### If too many rectangles remain:

Make rectangle filter **stricter**:
```python
MIN_AREARATIO_FILLED = 0.85  # Catch more rectangles (was 0.90)
MAX_ASPECT_RATIO = 0.95       # Stricter (was 0.97)
```

### If too many good particles removed:

Make rectangle filter **more lenient**:
```python
MIN_AREARATIO_FILLED = 0.95  # Only catch very rectangular (was 0.90)
MIN_ASPECT_RATIO = 0.02       # More lenient (was 0.05)
```

### If donuts still getting through:

Make donut filter **stricter**:
```python
DONUT_MAX_AREARATIO = 0.85   # Catch less hollow donuts (was 0.80)
DONUT_MAX_ASPECT = 0.30       # Catch less circular (was 0.25)
```

---

## Testing Specific Particles

Add this at the end of the script to test filtering on specific particles:

```python
def test_specific_particle(ds, idx):
    """Test if a specific particle passes filters"""
    print(f"\nParticle {idx}:")
    print(f"  Diameter:         {ds['diam'].values[idx]:.1f} μm")
    print(f"  Aspect ratio:     {ds['aspectratio'].values[idx]:.3f}")
    print(f"  Area ratio:       {ds['arearatio'].values[idx]:.3f}")
    print(f"  Area ratio fill:  {ds['arearatiofilled'].values[idx]:.3f}")
    print(f"  Edge touch:       {ds['edgetouch'].values[idx]}")

    # Check rectangle
    is_rectangle = (
        (ds['arearatiofilled'].values[idx] >= 0.90) &
        (ds['aspectratio'].values[idx] < 0.15)
    )

    # Check donut
    is_donut = (
        (ds['diam'].values[idx] >= 200) &
        (ds['arearatio'].values[idx] < 0.80) &
        (ds['aspectratio'].values[idx] < 0.25)
    )

    if is_rectangle:
        print("  ❌ FILTERED: Rectangle")
    elif is_donut:
        print("  ❌ FILTERED: Donut")
    else:
        print("  ✅ KEPT")

# Test problematic particles
# test_specific_particle(ds, 700)   # Rectangle
# test_specific_particle(ds, 9)     # Donut
# test_specific_particle(ds, 8731)  # Good crystal
```

---

## Differences from Original Script

### Original `pbp_plotting.ipynb`:
- ⚠️ Manual, cell-by-cell execution
- ⚠️ Only processed one probe file
- ⚠️ Basic filtering
- ⚠️ Rectangle filter missed some artifacts

### New `process_particle_data.py`:
- ✅ Single command execution
- ✅ Processes both probe files
- ✅ Enhanced filtering (improved rectangle detection)
- ✅ Progress indicators
- ✅ Comprehensive statistics
- ✅ Production-ready code
- ✅ Easy to configure and tune

---

## Integration with CNN

After running this script, update your CNN notebook to load data:

```python
# In your CNN notebook
df = pd.read_csv('particle_metadata.csv')

# Images are in:
liquid_images_dir = 'particle_images_filtered/liquid/'
solid_images_dir = 'particle_images_filtered/solid/'
```

The particle numbers in the CSV match the image filenames!

---

## Troubleshooting

### "File not found" error:
- Check that `Data/` directory contains both `.pbp.nc` files
- Check that environmental file `RF02.20250524.004127_075522.PNI.nc` exists

### Too few particles:
- Relax filters (see Tuning Guide above)
- Check filter statistics in output

### Too many rectangles still present:
- Decrease `MIN_AREARATIO_FILLED` to 0.85 or 0.80
- Increase `MAX_ASPECT_RATIO` to 0.95

### Out of memory:
- Process one probe at a time
- Reduce batch size when exporting

---

## Next Steps

1. Run the script: `python3 process_particle_data.py`
2. Check output statistics
3. Visually inspect sample images from both directories
4. Tune filters if needed
5. Use `particle_metadata.csv` for CNN training

Good luck with your CNN training!
