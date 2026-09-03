#!/usr/bin/env python3
"""
Difference DEM between two temporal epochs for reg_utm57_N_b.

Two comparisons are computed:
  1. Summer-Summer: mean(2013-07, 2014-07)  vs  mean(2019-06, 2019-07)
  2. Winter-Winter: 2012-02                 vs  mean(2024-02, 2025-02)

For each comparison:
  - DEMs are read on their common intersection at 30 m resolution
  - Epoch mean DEM is computed (nanmean, nodata masked)
  - dH = late_mean − early_mean  [m]
  - Output GeoTIFF and a 3-panel figure are saved

Run:
  /home/student/anaconda3/envs/TANDEMX/bin/python3 diff_dem_calc.py
"""

import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_bounds
from rasterio.enums import Resampling
from pathlib import Path

warnings.filterwarnings("ignore")

BASE = Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
OUT  = Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out")
OUT.mkdir(exist_ok=True)

NODATA_OUT = -9999.0

# ── DEM paths ─────────────────────────────────────────────────────────────────
def dem(year, acq, fname):
    return BASE / year / acq / "prc07" / fname

SUMMER_EARLY = [
    dem("2013", "2013-07-15_193817_TST_SM_D_33745_011_0045+1-2",
        "DEM_FNL_2013-07-15_193817_0-030m00_011_0045+1-2.tif"),
    dem("2014", "2014-07-02_193821_TST_SM_D_39089_011_0045+1-2",
        "DEM_FNL_2014-07-02_193821_0-030m00_011_0045+1-2.tif"),
]
SUMMER_LATE = [
    dem("2019", "2019-06-20_065923_TDT_SM_A_66621_155_0045+1-2",
        "DEM_FNL_2019-06-20_065923_0-030m00_155_0045+1-2.tif"),
    dem("2019", "2019-07-01_065923_TDT_SM_A_66788_155_0045+1-2",
        "DEM_FNL_2019-07-01_065923_0-030m00_155_0045+1-2.tif"),
]
WINTER_EARLY = [
    dem("2012", "2012-02-13_065833_TDT_SM_A_25873_155_0045_HH.",
        "DEM_FNL_2012-02-13_065833_0-030m00_155_0045_HH..tif"),
    dem("2017", "2017-02-12_065902_TDT_SM_A_53595_155_0045_HH.",
        "DEM_FNL_2017-02-12_065902_0-030m00_155_0045_HH..tif"),
]
WINTER_LATE = [
    dem("2024", "2024-02-08_065948_TDT_SM_A_92339_155_0045+1-2",
        "DEM_FNL_2024-02-08_065948_0-030m00_155_0045+1-2.tif"),
    dem("2025", "2025-02-16_065952_TDT_SM_A_98017_155_0045+1-2",
        "DEM_FNL_2025-02-16_065952_0-030m00_155_0045+1-2.tif"),
]

# ── Grid helpers ───────────────────────────────────────────────────────────────

def intersection_bounds(paths, res=30.0):
    """Compute intersection of all raster extents, aligned to res grid."""
    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    for p in paths:
        with rasterio.open(p) as src:
            b = src.bounds
            xmins.append(b.left);  xmaxs.append(b.right)
            ymins.append(b.bottom); ymaxs.append(b.top)
    xmin = max(xmins);  xmax = min(xmaxs)
    ymin = max(ymins);  ymax = min(ymaxs)
    # snap to grid
    xmin = np.ceil(xmin  / res) * res
    xmax = np.floor(xmax / res) * res
    ymin = np.ceil(ymin  / res) * res
    ymax = np.floor(ymax / res) * res
    cols = int(round((xmax - xmin) / res))
    rows = int(round((ymax - ymin) / res))
    return xmin, ymin, xmax, ymax, rows, cols


def read_to_grid(path, xmin, ymin, xmax, ymax, rows, cols):
    """Read a DEM resampled to the target grid; nodata → NaN."""
    with rasterio.open(path) as src:
        window = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax,
                                              transform=src.transform)
        data = src.read(
            1,
            window=window,
            out_shape=(rows, cols),
            resampling=Resampling.bilinear,
        ).astype(np.float64)
        nd = src.nodata
    if nd is not None:
        data[np.isclose(data, nd, rtol=0, atol=1e6)] = np.nan
    return data


def epoch_mean(paths, xmin, ymin, xmax, ymax, rows, cols):
    """Stack DEMs on common grid and return nanmean."""
    stack = np.stack([
        read_to_grid(p, xmin, ymin, xmax, ymax, rows, cols)
        for p in paths
    ])
    return np.nanmean(stack, axis=0)


def save_geotiff(arr, xmin, ymax, res, crs_wkt, path):
    transform = rasterio.transform.from_origin(xmin, ymax, res, res)
    arr_out = arr.copy()
    arr_out[np.isnan(arr_out)] = NODATA_OUT
    with rasterio.open(
        path, "w", driver="GTiff",
        height=arr.shape[0], width=arr.shape[1],
        count=1, dtype="float32",
        crs=crs_wkt, transform=transform,
        nodata=NODATA_OUT,
        compress="lzw",
    ) as dst:
        dst.write(arr_out.astype("float32"), 1)


def get_crs(path):
    with rasterio.open(path) as src:
        return src.crs

# ── Stats ─────────────────────────────────────────────────────────────────────

def stats(dh):
    valid = dh[np.isfinite(dh)]
    if valid.size == 0:
        return {}
    med  = float(np.nanmedian(valid))
    nmad = float(1.4826 * np.nanmedian(np.abs(valid - med)))
    return {
        "n":      valid.size,
        "mean":   float(np.nanmean(valid)),
        "std":    float(np.nanstd(valid)),
        "median": med,
        "nmad":   nmad,
    }

# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(early_mean, late_mean, dh, label, early_label, late_label, out_png):
    s = stats(dh)
    valid = dh[np.isfinite(dh)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Difference DEM  —  {label}\n"
        f"dH = {late_label} − {early_label}   "
        f"(median={s['median']:+.2f} m,  NMAD={s['nmad']:.2f} m,  n={s['n']:,})",
        fontsize=11,
    )

    # Panel 1 — early mean DEM
    ax = axes[0]
    p5, p95 = np.nanpercentile(early_mean[np.isfinite(early_mean)], [2, 98])
    im = ax.imshow(early_mean, cmap="terrain", vmin=p5, vmax=p95, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Elevation  [m]", shrink=0.8)
    ax.set_title(f"Early epoch mean DEM\n{early_label}")
    ax.set_xlabel("column"); ax.set_ylabel("row")

    # Panel 2 — dH spatial map
    ax = axes[1]
    vlim = max(abs(np.nanpercentile(valid, 2)), abs(np.nanpercentile(valid, 98)))
    vlim = min(vlim, 50)   # cap at ±50 m for display
    im2 = ax.imshow(dh, cmap="RdBu_r", vmin=-vlim, vmax=vlim, interpolation="nearest")
    plt.colorbar(im2, ax=ax, label="dH  [m]", shrink=0.8)
    ax.set_title(f"dH = late − early  [m]\n{late_label} − {early_label}")
    ax.set_xlabel("column")

    # Panel 3 — histogram
    ax = axes[2]
    clip = np.clip(valid, -3*vlim, 3*vlim)
    ax.hist(clip, bins=300, color="steelblue", alpha=0.75, density=True)
    ax.axvline(s["median"], color="black",  lw=1.8, label=f"median = {s['median']:+.2f} m")
    ax.axvline(s["mean"],   color="orange", lw=1.5, linestyle="--", label=f"mean = {s['mean']:+.2f} m")
    ax.axvline(0,           color="grey",   lw=1.0, linestyle=":")
    ax.set_xlabel("dH  [m]"); ax.set_ylabel("density")
    ax.set_title("dH histogram")
    ax.legend(fontsize=9)
    ax.set_xlim(-3*vlim, 3*vlim)

    # Annotate NMAD / std on histogram
    ax.text(0.97, 0.97,
            f"NMAD = {s['nmad']:.2f} m\nStd  = {s['std']:.2f} m\nn     = {s['n']:,}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  figure → {out_png}")

# ── Main ──────────────────────────────────────────────────────────────────────

def run_comparison(early_paths, late_paths, label, early_label, late_label, tag):
    print(f"\n{'─'*60}")
    print(f"Comparison: {label}")
    print(f"  early: {[p.parent.parent.name[:16] for p in early_paths]}")
    print(f"  late : {[p.parent.parent.name[:16] for p in late_paths]}")

    all_paths = early_paths + late_paths
    xmin, ymin, xmax, ymax, rows, cols = intersection_bounds(all_paths)
    print(f"  intersection: {cols}×{rows} px  ({(xmax-xmin)/1000:.1f}×{(ymax-ymin)/1000:.1f} km)")

    print("  reading early epoch …")
    early_dem = epoch_mean(early_paths, xmin, ymin, xmax, ymax, rows, cols)
    print("  reading late epoch …")
    late_dem  = epoch_mean(late_paths,  xmin, ymin, xmax, ymax, rows, cols)

    dh = late_dem - early_dem

    s = stats(dh)
    print(f"  dH stats: median={s['median']:+.2f} m  NMAD={s['nmad']:.2f} m  "
          f"mean={s['mean']:+.2f} m  std={s['std']:.2f} m  n={s['n']:,}")

    crs = get_crs(all_paths[0])
    tif_early = OUT / f"diff_{tag}_early_mean.tif"
    tif_late  = OUT / f"diff_{tag}_late_mean.tif"
    tif_dh    = OUT / f"diff_{tag}_dH.tif"
    save_geotiff(early_dem, xmin, ymax, 30.0, crs, tif_early)
    save_geotiff(late_dem,  xmin, ymax, 30.0, crs, tif_late)
    save_geotiff(dh,        xmin, ymax, 30.0, crs, tif_dh)
    print(f"  TIFs → {tif_early.name}, {tif_late.name}, {tif_dh.name}")

    out_png = OUT / f"diff_{tag}_figure.png"
    make_figure(early_dem, late_dem, dh, label, early_label, late_label, out_png)


def main():
    run_comparison(
        SUMMER_EARLY, SUMMER_LATE,
        label="Summer  (June–July)",
        early_label="mean(2013-07 + 2014-07)",
        late_label="mean(2019-06 + 2019-07)",
        tag="summer",
    )
    run_comparison(
        WINTER_EARLY, WINTER_LATE,
        label="Winter  (January–February)",
        early_label="mean(2012-02 + 2017-02)",
        late_label="mean(2024-02 + 2025-02)",
        tag="winter",
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
