#!/usr/bin/env python3
"""Preview PNGs of the cleaned glacier-rate tifs (m/yr)."""
import numpy as np, rasterio, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
OUT = Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out")

for name in ["summer","winter"]:
    f = OUT/f"diff_{name}_rate_m_per_yr_glacieronly.tif"
    with rasterio.open(f) as s:
        a = s.read(1).astype(float); a[a==s.nodata]=np.nan
        b = s.bounds
    v = a[np.isfinite(a)]
    vlim = np.nanpercentile(np.abs(v), 98)        # robust symmetric color limit
    med = np.nanmedian(v)
    fig, ax = plt.subplots(figsize=(9,8))
    im = ax.imshow(a, cmap="RdBu", vmin=-vlim, vmax=vlim,
                   extent=[b.left,b.right,b.bottom,b.top])
    cb = plt.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Elevation-change rate  [m yr$^{-1}$]")
    ax.set_title(f"Klyuchevskaya glaciers — {name} dH rate\n"
                 f"median = {med:+.3f} m/yr   (RdBu: blue=gain, red=loss)")
    ax.set_xlabel("UTM 57N Easting [m]"); ax.set_ylabel("Northing [m]")
    out = OUT/f"PREVIEW_{name}_rate_glacier.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  {name}: vlim=±{vlim:.2f} m/yr  -> {out.name}")
print("DONE")
