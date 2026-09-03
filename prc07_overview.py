#!/usr/bin/env python3
"""
Overview of co-registered TanDEM-X DEMs in prc07 folders.
For each acquisition in DEM-utm57_N_b:
  - lists key products present / missing in prc07
  - reads correg_shft_params (dx, dy, dz co-registration shifts)
  - computes DEM_FNL minus CP30 reference difference statistics
  - flags outliers where |dH| > OUTLIER_THRESHOLD metres
  - saves per-acquisition outlier plots and a summary CSV

Run:
  /home/student/anaconda3/envs/TANDEMX/bin/python3 prc07_overview.py
"""

import os
import re
import csv
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore")

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("WARNING: rasterio not found — DEM diff stats and outlier plots disabled.")

# ── Configuration ─────────────────────────────────────────────────────────────
BASE       = Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
OUT_DIR    = Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out")
OUT_DIR.mkdir(exist_ok=True)

# dH values beyond this threshold (metres) are flagged as outliers
OUTLIER_THRESHOLD = 50.0

# Key products expected in every prc07 folder
EXPECTED_PRODUCTS = [
    "DEM_FNL", "DEM_VER", "DEM_STB",
    "DH_FNL",  "DH_VER",  "COH_CAS",
    "MSK_DAT", "SUM_MSK",
]
# ──────────────────────────────────────────────────────────────────────────────


def find_tif(folder: Path, prefix: str) -> Path | None:
    """Return the first .tif in folder whose name starts with prefix."""
    for f in sorted(folder.glob(f"{prefix}_*.tif")):
        return f
    return None


def read_correg_shifts(prc07: Path):
    """Parse correg_shft_params → (dx, dy, dz) or (None, None, None)."""
    p = prc07 / "correg_shft_params"
    if not p.exists():
        return None, None, None
    try:
        lines = p.read_text().splitlines()
        vals  = lines[1].split()
        return float(vals[0]), float(vals[1]), float(vals[2])
    except Exception:
        return None, None, None


def dem_diff_stats(dem_fnl: Path, ref_dem: Path):
    """
    Compute (mean, std, median, nmad, n_outliers, outlier_frac) of
    DEM_FNL - CP30_ref.  Returns dict of stats, or None if rasterio missing.
    """
    if not HAS_RASTERIO:
        return None
    try:
        with rasterio.open(dem_fnl) as src:
            d = src.read(1).astype(float)
            d[d == src.nodata] = np.nan
        with rasterio.open(ref_dem) as src:
            r = src.read(
                1,
                out_shape=(src.count, d.shape[0], d.shape[1]),
                resampling=rasterio.enums.Resampling.bilinear,
            ).astype(float)
            r[r == src.nodata] = np.nan
        dh = d - r
        valid = dh[np.isfinite(dh)]
        if valid.size == 0:
            return None
        median = float(np.nanmedian(valid))
        nmad   = float(1.4826 * np.nanmedian(np.abs(valid - median)))
        out    = np.abs(valid - median) > OUTLIER_THRESHOLD
        return {
            "dh_mean":        float(np.nanmean(valid)),
            "dh_std":         float(np.nanstd(valid)),
            "dh_median":      median,
            "dh_nmad":        nmad,
            "n_valid":        int(valid.size),
            "n_outliers":     int(out.sum()),
            "outlier_frac_%": round(100 * out.sum() / valid.size, 2),
        }, dh
    except Exception as e:
        print(f"    DEM diff error: {e}")
        return None, None


def plot_outliers(dh: np.ndarray, acq_name: str, out_path: Path,
                  dx: float, dy: float, dz: float):
    """
    Two-panel plot:
      left  — spatial map of dH with outliers highlighted
      right — histogram of dH with threshold lines
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{acq_name}\ndx={dx:.2f}m  dy={dy:.2f}m  dz={dz:.2f}m", fontsize=10)

    # Clip colour range for display
    vmin, vmax = np.nanpercentile(dh, [2, 98])
    vmax_disp  = max(abs(vmin), abs(vmax))

    # Spatial map
    ax = axes[0]
    im = ax.imshow(dh, cmap="RdBu_r",
                   vmin=-vmax_disp, vmax=vmax_disp, interpolation="nearest")
    # Overlay outlier mask
    out_mask = np.abs(dh - np.nanmedian(dh)) > OUTLIER_THRESHOLD
    ax.imshow(np.where(out_mask, 1, np.nan), cmap="autumn", alpha=0.6,
              vmin=0, vmax=1, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="dH  DEM_FNL − COP30  [m]", shrink=0.8)
    ax.set_title("dH map  (orange = outlier)")
    ax.set_xlabel("column"); ax.set_ylabel("row")

    # Histogram
    ax2   = axes[1]
    valid = dh[np.isfinite(dh)]
    ax2.hist(valid, bins=200, color="steelblue", alpha=0.7, density=True)
    for sign in (-1, 1):
        ax2.axvline(sign * OUTLIER_THRESHOLD, color="red", linestyle="--",
                    linewidth=1.5, label=f"±{OUTLIER_THRESHOLD} m threshold")
    ax2.axvline(float(np.nanmedian(valid)), color="black", linestyle="-",
                linewidth=1.5, label=f"median={np.nanmedian(valid):.2f} m")
    ax2.set_xlabel("dH  [m]"); ax2.set_ylabel("density")
    ax2.set_title("dH histogram")
    ax2.legend(fontsize=8)
    ax2.set_xlim(-3 * OUTLIER_THRESHOLD, 3 * OUTLIER_THRESHOLD)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rows  = []
    total = 0

    for year_dir in sorted(BASE.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        for acq_dir in sorted(year_dir.iterdir()):
            if not acq_dir.is_dir():
                continue
            total += 1
            acq   = acq_dir.name
            prc07 = acq_dir / "prc07"

            row = {
                "acquisition": acq,
                "year":        year_dir.name,
                "prc07":       "YES" if prc07.exists() else "NO",
            }

            # Date from acquisition name
            m = re.match(r"(\d{4}-\d{2}-\d{2})", acq)
            row["date"] = m.group(1) if m else ""

            if not prc07.exists():
                print(f"  [{year_dir.name}] {acq[:60]}  — NO prc07")
                row.update({p: "MISSING" for p in EXPECTED_PRODUCTS})
                rows.append(row)
                continue

            # Product presence
            missing = []
            for prod in EXPECTED_PRODUCTS:
                tif = find_tif(prc07, prod)
                row[prod] = "OK" if tif else "MISSING"
                if not tif:
                    missing.append(prod)

            # Co-registration shifts
            dx, dy, dz = read_correg_shifts(prc07)
            row["dx_m"] = round(dx, 3) if dx is not None else ""
            row["dy_m"] = round(dy, 3) if dy is not None else ""
            row["dz_m"] = round(dz, 3) if dz is not None else ""

            # DEM_FNL vs COP30 difference
            dem_fnl = find_tif(prc07, "DEM_FNL")
            ref_dem = acq_dir / "prc05_refdem" / "CP30_strp_dem.tif"

            status_parts = []
            if missing:
                status_parts.append(f"MISSING: {','.join(missing)}")

            if HAS_RASTERIO and dem_fnl and ref_dem.exists():
                result, dh = dem_diff_stats(dem_fnl, ref_dem)
                if result:
                    row.update(result)
                    if result["outlier_frac_%"] > 5:
                        status_parts.append(
                            f"HIGH_OUTLIERS({result['outlier_frac_%']:.1f}%)"
                        )
                    # Plot
                    plot_path = OUT_DIR / f"{acq[:80]}_dH_outliers.png"
                    plot_outliers(
                        dh, acq, plot_path,
                        dx or 0, dy or 0, dz or 0
                    )
                    row["plot"] = plot_path.name
                else:
                    status_parts.append("DEM_DIFF_FAILED")
            elif not ref_dem.exists():
                status_parts.append("NO_REF_DEM")

            row["status"] = " | ".join(status_parts) if status_parts else "OK"

            shift_str = (f"dx={dx:.2f} dy={dy:.2f} dz={dz:.2f}"
                         if dx is not None else "no shifts")
            print(f"  [{year_dir.name}] {acq[:55]:55s}  {shift_str}  {row['status']}")
            rows.append(row)

    # Write CSV
    if not rows:
        print("No acquisitions found.")
        return

    fieldnames = ["acquisition", "year", "date", "prc07", "status",
                  "dx_m", "dy_m", "dz_m"] + EXPECTED_PRODUCTS + [
                  "dh_mean", "dh_std", "dh_median", "dh_nmad",
                  "n_valid", "n_outliers", "outlier_frac_%", "plot"]

    csv_path = OUT_DIR / "prc07_overview.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Console summary
    n_ok      = sum(1 for r in rows if r.get("status") == "OK")
    n_no_prc  = sum(1 for r in rows if r.get("prc07") == "NO")
    n_missing = sum(1 for r in rows if "MISSING" in r.get("status", ""))
    n_out     = sum(1 for r in rows if "HIGH_OUTLIERS" in r.get("status", ""))

    print(f"\n{'─'*60}")
    print(f"Total acquisitions : {total}")
    print(f"  prc07 present    : {total - n_no_prc}")
    print(f"  prc07 missing    : {n_no_prc}")
    print(f"  fully OK         : {n_ok}")
    print(f"  missing products : {n_missing}")
    print(f"  high outliers    : {n_out}  (>{OUTLIER_THRESHOLD} m, >5% of pixels)")
    print(f"\nCSV  → {csv_path}")
    print(f"Plots→ {OUT_DIR}/")


if __name__ == "__main__":
    main()