#!/usr/bin/env python3
"""A presentation slide summarizing the processing pipeline challenges & fixes."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
OUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out")

fig,ax=plt.subplots(figsize=(14,9)); ax.axis('off')
ax.set_title("Processing challenges & solutions — TanDEM-X DEM pipeline",
             fontsize=15,fontweight='bold',pad=18)

txt = [
 ("1. Orbit-number rollover (2026 data)", "#1f77b4",
  "• DLR orbit counter crossed 100 000 in 2026 → 5→6 digit numbers in filenames\n"
  "• Two fixed-character string ops broke (cut -c 34-41 for path; cut -c -44 for strip name)\n"
  "  → scenes mis-sorted / wrong filenames\n"
  "✓ FIXED: replaced fixed positions with structure-aware parsing (awk fields, sed suffix)"),
 ("2. GAMMA SLC_cat concatenation (2026 — UNSOLVED)", "firebrick",
  "• 4 winter 2026 dates: two sub-scenes must be concatenated into one strip\n"
  "• GAMMA SLC_cat (dopflg=1, Doppler interp) reads 1 line past array edge at a\n"
  "  knife-edge geometry (exactly 1 PRF = 3802 lines overlap margin)\n"
  "• dopflg=0 fixes concat but then multilook fails on the join geometry\n"
  "✗ Needs GAMMA-internal fix or DLR re-delivery — low impact (8 other 2025 winter DEMs)"),
 ("3. Range-sample mismatch (2013/14 descending)", "#ff7f0e",
  "• Two sub-scenes per date had different widths (17332 vs 17076 range samples)\n"
  "  → cannot concatenate frames of unequal width\n"
  "✓ FIXED: max_cat=1 → process each scene singly (no concatenation needed)"),
 ("4. Corrupt / partial DEMs", "#2ca02c",
  "• Some scenes failed at geocoding (empty-array) or produced noise (NMAD>1000 m)\n"
  "✓ FIXED: automated outlier rejection (stable-terrain NMAD, |anomaly| thresholds)\n"
  "  + median epoch-stacking (robust to per-scene artefacts)"),
 ("5. Empty SLC / extraction glitches", "#9467bd",
  "• 1 file extracted 0 bytes (HPC glitch, not data); some runs left stale .shp artefacts\n"
  "✓ FIXED: re-extraction + clearing collision artefacts; clean_slc=0 preserves intermediates"),
]
y=0.92
for title,col,body in txt:
    ax.text(0.02,y,title,fontsize=12,fontweight='bold',color=col,transform=ax.transAxes)
    ax.text(0.04,y-0.035,body,fontsize=9.5,va='top',transform=ax.transAxes,family='monospace')
    y-=0.185
fig.savefig(OUT/"PRES_D_challenges.png",dpi=140,bbox_inches="tight")
print("-> PRES_D_challenges.png")
