"""
check_classification_accuracy.py

Evaluates classification accuracy on unambiguous particles using temperature as
ground truth, matching the same approach used in process_particle_data.py:

    Particles before last warm time (ATX >= 1 C)  -> expected Liquid (phase 0)
    Particles after first cold time (ATX <= -40 C) -> expected Solid  (phase 1)

Requires:
    - inference_results/particle_phase_predictions.csv  (from particle_phase_inference.ipynb)
    - The environmental netCDF file for the same flight  (ATX variable)

Usage:
    python3 check_classification_accuracy.py
    Or paste directly into a Colab cell.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
base_path = './'   # adjust if needed

PREDICTIONS_CSV = base_path + 'inference_results/particle_phase_predictions.csv'
ENV_FILE        = base_path + 'Data/cgwaves/RF02.20250524.004127_075522.PNI.nc'

LIQUID_THRESHOLD =  1.0   # C — last time ATX >= this defines liquid region
SOLID_THRESHOLD  = -40.0  # C — first time ATX <= this defines solid region

# ── Get phase cutoff times from environmental data ────────────────────────────
# Mirrors get_phase_cutoff_times() from process_particle_data.py
print(f"Loading environmental data: {Path(ENV_FILE).name}")
ds_env = xr.open_dataset(ENV_FILE, decode_times=True)

liquid_mask  = ds_env.ATX >= LIQUID_THRESHOLD
solid_mask   = ds_env.ATX <= SOLID_THRESHOLD
liquid_cutoff = pd.Timestamp(ds_env.isel(Time=liquid_mask.values)['Time'].max().values)
solid_cutoff  = pd.Timestamp(ds_env.isel(Time=solid_mask.values)['Time'].min().values)
ds_env.close()

print(f"  Liquid cutoff (last warm time) : {liquid_cutoff}")
print(f"  Solid  cutoff (first cold time): {solid_cutoff}")

# ── Load predictions CSV ──────────────────────────────────────────────────────
print(f"\nLoading predictions: {Path(PREDICTIONS_CSV).name}")
df = pd.read_csv(PREDICTIONS_CSV, parse_dates=['time'])
print(f"  Total particles: {len(df):,}")
print(f"  Predicted phases: {df['predicted_label'].value_counts().to_dict()}")

# ── Assign ground truth labels by time ───────────────────────────────────────
# Only particles in unambiguous temperature regions get a ground truth label.
# Particles in the mixed-phase window (between the two cutoffs) are excluded.
df['gt_phase'] = np.where(
    df['time'] <= liquid_cutoff, 0,          # liquid region
    np.where(df['time'] >= solid_cutoff, 1,  # solid region
             -1)                              # mixed-phase — no ground truth
)

# Exclude mixed-phase and donut/filtered particles from evaluation
eval_df = df[(df['gt_phase'] >= 0) & (df['predicted_phase'].isin([0, 1]))].copy()

print(f"\n  Unambiguous particles available for evaluation: {len(eval_df):,}")
print(f"    Expected liquid : {(eval_df['gt_phase'] == 0).sum():,}")
print(f"    Expected solid  : {(eval_df['gt_phase'] == 1).sum():,}")

# ── Compute accuracy ──────────────────────────────────────────────────────────
for probe in sorted(df['probe'].unique()):
    probe_df = eval_df[eval_df['probe'] == probe]
    if len(probe_df) == 0:
        continue

    liq = probe_df[probe_df['gt_phase'] == 0]
    sol = probe_df[probe_df['gt_phase'] == 1]

    acc_liq = (liq['predicted_phase'] == 0).mean() if len(liq) > 0 else float('nan')
    acc_sol = (sol['predicted_phase'] == 1).mean() if len(sol) > 0 else float('nan')
    acc_all = (probe_df['predicted_phase'] == probe_df['gt_phase']).mean()

    print(f"\nProbe {probe}")
    print(f"  Liquid accuracy : {(liq['predicted_phase']==0).sum():,} / {len(liq):,}  ({acc_liq*100:.1f}%)")
    print(f"  Solid  accuracy : {(sol['predicted_phase']==1).sum():,} / {len(sol):,}  ({acc_sol*100:.1f}%)")
    print(f"  Overall         : {(probe_df['predicted_phase']==probe_df['gt_phase']).sum():,} / {len(probe_df):,}  ({acc_all*100:.1f}%)")

# ── Combined across all probes ────────────────────────────────────────────────
liq_all = eval_df[eval_df['gt_phase'] == 0]
sol_all = eval_df[eval_df['gt_phase'] == 1]

acc_liq = (liq_all['predicted_phase'] == 0).mean() if len(liq_all) > 0 else float('nan')
acc_sol = (sol_all['predicted_phase'] == 1).mean() if len(sol_all) > 0 else float('nan')
acc_all = (eval_df['predicted_phase'] == eval_df['gt_phase']).mean()

print(f"\n{'='*50}")
print(f"Combined (all probes)")
print(f"  Liquid accuracy : {(liq_all['predicted_phase']==0).sum():,} / {len(liq_all):,}  ({acc_liq*100:.1f}%)")
print(f"  Solid  accuracy : {(sol_all['predicted_phase']==1).sum():,} / {len(sol_all):,}  ({acc_sol*100:.1f}%)")
print(f"  Overall         : {(eval_df['predicted_phase']==eval_df['gt_phase']).sum():,} / {len(eval_df):,}  ({acc_all*100:.1f}%)")
