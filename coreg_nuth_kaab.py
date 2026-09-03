#!/usr/bin/env python3
"""
Nuth & Kääb co-registration for summer and winter difference DEMs.

Workflow per comparison:
  1. Load early_mean and late_mean TIFs (EPSG:32657 / UTM 57N)
  2. Rasterize GLINV glacier outline → glacier mask
  3. stable_mask = ~glacier & both DEMs valid
  4. Fit NuthKaab on stable terrain, apply to late DEM
  5. Compute corrected dH = late_corrected − early
  6. Save corrected TIFs and updated figure

Run:
  /home/student/anaconda3/envs/TANDEMX/bin/python3 coreg_nuth_kaab.py
"""

import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
import rasterio.features
import rasterio.transform
from rasterio.crs import CRS
import geopandas as gpd
import xdem
from pathlib import Path

warnings.filterwarnings("ignore")

OUT  = Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out")
CRS_UTM57N = CRS.from_epsg(32657)

# Glacier inventory — same file is present in every prc05_refdem, pick one
GLINV_SHP = Path(
    "/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b"
    "/2013/2013-07-15_193817_TST_SM_D_33745_011_0045+1-2"
    "/prc05_refdem/GLINV_rgi06_ttl_utm57_N_b_UTM.shp"
)

COMPARISONS = [
    dict(
        tag="summer",
        early_tif=OUT / "diff_summer_early_mean.tif",
        late_tif =OUT / "diff_summer_late_mean.tif",
        label="Summer  (June–July)",
        early_label="mean(2013-07 + 2014-07)",
        late_label ="mean(2019-06 + 2019-07)",
    ),
    dict(
        tag="winter",
        early_tif=OUT / "diff_winter_early_mean.tif",
        late_tif =OUT / "diff_winter_late_mean.tif",
        label="Winter  (January–February)",
        early_label="mean(2012-02 + 2017-02)",
        late_label ="mean(2024-02 + 2025-02)",
    ),
]

NODATA_IN  = -9999.0
NODATA_OUT = -9999.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_dem(path):
    """Read TIF → (array float64 with NaN, transform, shape)."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float64)
        transform = src.transform
        nd = src.nodata if src.nodata is not None else NODATA_IN
    data[np.isclose(data, nd, rtol=0, atol=1)] = np.nan
    return data, transform


def rasterize_glacier_mask(shp_path, transform, shape, crs):
    """
    Burn glacier polygons into a boolean array (True = glacier).
    Reprojects shapefile to crs if needed.
    """
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None or gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    mask = rasterio.features.rasterize(
        [(geom, 1) for geom in gdf.geometry if geom is not None],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    ).astype(bool)
    return mask


def save_tif(arr, transform, path):
    arr_out = arr.copy()
    arr_out[np.isnan(arr_out)] = NODATA_OUT
    with rasterio.open(
        path, "w", driver="GTiff",
        height=arr.shape[0], width=arr.shape[1],
        count=1, dtype="float32",
        crs=CRS_UTM57N, transform=transform,
        nodata=NODATA_OUT, compress="lzw",
    ) as dst:
        dst.write(arr_out.astype("float32"), 1)


def stats(dh):
    valid = dh[np.isfinite(dh)]
    if valid.size == 0:
        return {}
    med  = float(np.nanmedian(valid))
    nmad = float(1.4826 * np.nanmedian(np.abs(valid - med)))
    return dict(n=valid.size, mean=float(np.nanmean(valid)),
                std=float(np.nanstd(valid)), median=med, nmad=nmad)


def make_figure(early, dh_raw, dh_cor, label, early_label, late_label, out_png,
                stable_mask, glacier_mask):
    s_raw = stats(dh_raw)
    s_cor = stats(dh_cor)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"Nuth & Kääb Co-registration  —  {label}\n"
        f"dH = {late_label} − {early_label}",
        fontsize=11,
    )

    def imshow_dh(ax, arr, title, vlim=None):
        valid = arr[np.isfinite(arr)]
        if vlim is None:
            vlim = min(max(abs(np.nanpercentile(valid, 2)),
                           abs(np.nanpercentile(valid, 98))), 30)
        im = ax.imshow(arr, cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                       interpolation="nearest")
        plt.colorbar(im, ax=ax, label="dH  [m]", shrink=0.8)
        ax.set_title(title)

    # Row 0 — raw dH
    ax = axes[0, 0]
    p5, p95 = np.nanpercentile(early[np.isfinite(early)], [2, 98])
    im = ax.imshow(early, cmap="terrain", vmin=p5, vmax=p95, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Elevation  [m]", shrink=0.8)
    ax.set_title(f"Early epoch mean DEM\n{early_label}")

    imshow_dh(axes[0, 1], dh_raw,
              f"dH  BEFORE coreg\nmedian={s_raw['median']:+.2f} m  NMAD={s_raw['nmad']:.2f} m")

    ax = axes[0, 2]
    valid_raw = np.clip(dh_raw[np.isfinite(dh_raw)], -50, 50)
    ax.hist(valid_raw, bins=300, color="steelblue", alpha=0.7, density=True, label="all")
    ax.axvline(s_raw["median"], color="black", lw=1.8,
               label=f"median {s_raw['median']:+.2f} m")
    ax.axvline(0, color="grey", lw=1, linestyle=":")
    ax.set_xlabel("dH  [m]"); ax.set_ylabel("density")
    ax.set_title("Histogram  BEFORE coreg")
    ax.set_xlim(-50, 50); ax.legend(fontsize=8)
    ax.text(0.97, 0.97,
            f"NMAD={s_raw['nmad']:.2f} m\nStd={s_raw['std']:.2f} m\nn={s_raw['n']:,}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    # Row 1 — corrected dH
    # Stable terrain overlay on dH map
    stable_disp = np.where(stable_mask, np.nan, 0.5)  # shade non-stable lightly
    imshow_dh(axes[1, 0], dh_cor,
              f"dH  AFTER coreg\nmedian={s_cor['median']:+.2f} m  NMAD={s_cor['nmad']:.2f} m")

    # Glacier-only dH after correction
    dh_glac = dh_cor.copy()
    dh_glac[~glacier_mask] = np.nan
    imshow_dh(axes[1, 1], dh_glac,
              "dH  on glacier pixels only\n(after coreg)")

    ax = axes[1, 2]
    valid_cor = np.clip(dh_cor[np.isfinite(dh_cor)], -50, 50)
    ax.hist(valid_cor, bins=300, color="steelblue", alpha=0.7, density=True, label="all")
    # Stable terrain histogram
    valid_stb = np.clip(dh_cor[stable_mask & np.isfinite(dh_cor)], -50, 50)
    if valid_stb.size > 0:
        ax.hist(valid_stb, bins=300, color="orange", alpha=0.5, density=True,
                label="stable terrain")
    ax.axvline(s_cor["median"], color="black", lw=1.8,
               label=f"median {s_cor['median']:+.2f} m")
    ax.axvline(0, color="grey", lw=1, linestyle=":")
    ax.set_xlabel("dH  [m]"); ax.set_ylabel("density")
    ax.set_title("Histogram  AFTER coreg")
    ax.set_xlim(-50, 50); ax.legend(fontsize=8)
    ax.text(0.97, 0.97,
            f"NMAD={s_cor['nmad']:.2f} m\nStd={s_cor['std']:.2f} m\nn={s_cor['n']:,}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    plt.tight_layout()
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  figure → {out_png}")


def make_corrected_figure(early, dh_cor, label, early_label, late_label, out_png,
                          glacier_mask):
    """Clean 3-panel standalone figure for the corrected dH result."""
    s = stats(dh_cor)
    valid = dh_cor[np.isfinite(dh_cor)]
    vlim  = min(max(abs(np.nanpercentile(valid, 2)), abs(np.nanpercentile(valid, 98))), 30)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Corrected Difference DEM  —  {label}\n"
        f"dH = {late_label} − {early_label}   "
        f"(median={s['median']:+.2f} m,  NMAD={s['nmad']:.2f} m,  n={s['n']:,})",
        fontsize=11,
    )

    # Panel 1 — early mean DEM
    ax = axes[0]
    p5, p95 = np.nanpercentile(early[np.isfinite(early)], [2, 98])
    im = ax.imshow(early, cmap="terrain", vmin=p5, vmax=p95, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Elevation  [m]", shrink=0.8)
    ax.set_title(f"Early epoch mean DEM\n{early_label}")
    ax.set_xlabel("column"); ax.set_ylabel("row")

    # Panel 2 — corrected dH spatial map
    ax = axes[1]
    im2 = ax.imshow(dh_cor, cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                    interpolation="nearest")
    plt.colorbar(im2, ax=ax, label="dH  [m]", shrink=0.8)
    ax.set_title(f"dH corrected  [m]\n{late_label} − {early_label}")
    ax.set_xlabel("column")

    # Panel 3 — histogram (all vs glacier only)
    ax = axes[2]
    valid_all  = np.clip(valid, -3*vlim, 3*vlim)
    valid_glac = np.clip(dh_cor[glacier_mask & np.isfinite(dh_cor)], -3*vlim, 3*vlim)
    ax.hist(valid_all,  bins=300, color="steelblue", alpha=0.6, density=True, label="all terrain")
    if valid_glac.size > 0:
        s_glac = stats(dh_cor[glacier_mask & np.isfinite(dh_cor)])
        ax.hist(valid_glac, bins=300, color="crimson", alpha=0.6, density=True,
                label=f"glacier only  (median={s_glac['median']:+.2f} m)")
    ax.axvline(s["median"], color="black",  lw=1.8, label=f"median (all) = {s['median']:+.2f} m")
    ax.axvline(0,           color="grey",   lw=1.0, linestyle=":")
    ax.set_xlabel("dH  [m]"); ax.set_ylabel("density")
    ax.set_title("dH histogram  (after coreg)")
    ax.legend(fontsize=8)
    ax.set_xlim(-3*vlim, 3*vlim)
    ax.text(0.97, 0.97,
            f"NMAD = {s['nmad']:.2f} m\nStd  = {s['std']:.2f} m\nn     = {s['n']:,}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    plt.tight_layout()
    plt.savefig(out_png, dpi=800, bbox_inches="tight")
    plt.close()
    print(f"  corrected figure → {out_png}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(cfg):
    tag         = cfg["tag"]
    early_label = cfg["early_label"]
    late_label  = cfg["late_label"]
    label       = cfg["label"]

    print(f"\n{'─'*60}")
    print(f"Co-registration: {label}")

    early, transform = load_dem(cfg["early_tif"])
    late,  _         = load_dem(cfg["late_tif"])
    shape = early.shape

    print(f"  grid: {shape[1]}×{shape[0]} px")

    # Glacier mask
    print("  rasterizing glacier mask …")
    glacier_mask = rasterize_glacier_mask(GLINV_SHP, transform, shape, CRS_UTM57N)
    stable_mask  = (~glacier_mask) & np.isfinite(early) & np.isfinite(late)
    n_stable = stable_mask.sum()
    n_glac   = (glacier_mask & np.isfinite(early) & np.isfinite(late)).sum()
    print(f"  stable terrain pixels: {n_stable:,}  |  glacier pixels: {n_glac:,}")

    # Raw dH before correction
    dh_raw = late - early

    # xDEM Nuth & Kääb
    print("  fitting Nuth & Kääb …")
    ref_xdem = xdem.DEM.from_array(
        early.astype(np.float32), transform=transform,
        crs=CRS_UTM57N, nodata=NODATA_OUT,
    )
    tba_xdem = xdem.DEM.from_array(
        late.astype(np.float32), transform=transform,
        crs=CRS_UTM57N, nodata=NODATA_OUT,
    )
    coreg = xdem.coreg.NuthKaab()
    coreg.fit(ref_xdem, tba_xdem, inlier_mask=stable_mask, random_state=42)

    # Print estimated shifts
    try:
        meta = coreg.meta
        print(f"  estimated shifts → {meta}")
    except Exception:
        pass

    print("  applying correction …")
    late_corrected = coreg.apply(tba_xdem)
    late_cor_arr   = late_corrected.data.filled(np.nan).astype(np.float64)
    if late_cor_arr.ndim == 3:
        late_cor_arr = late_cor_arr[0]

    dh_cor = late_cor_arr - early

    # Remove residual vertical bias on stable terrain
    stable_median = float(np.nanmedian(dh_cor[stable_mask & np.isfinite(dh_cor)]))
    dh_cor -= stable_median
    late_cor_arr -= stable_median
    print(f"  residual vertical bias on stable terrain: {stable_median:+.3f} m → removed")

    s_raw = stats(dh_raw)
    s_cor = stats(dh_cor)
    print(f"  BEFORE coreg: median={s_raw['median']:+.3f} m  NMAD={s_raw['nmad']:.3f} m")
    print(f"  AFTER  coreg: median={s_cor['median']:+.3f} m  NMAD={s_cor['nmad']:.3f} m")

    # Save corrected TIFs
    save_tif(late_cor_arr, transform, OUT / f"diff_{tag}_late_corrected.tif")
    save_tif(dh_cor,       transform, OUT / f"diff_{tag}_dH_corrected.tif")
    print(f"  saved diff_{tag}_late_corrected.tif  &  diff_{tag}_dH_corrected.tif")

    # 6-panel before/after comparison figure
    out_png = OUT / f"diff_{tag}_coreg_figure.png"
    make_figure(early, dh_raw, dh_cor, label, early_label, late_label, out_png,
                stable_mask, glacier_mask)

    # Clean 3-panel standalone figure for the corrected result
    out_png_clean = OUT / f"diff_{tag}_dH_corrected.png"
    make_corrected_figure(early, dh_cor, label, early_label, late_label,
                          out_png_clean, glacier_mask)


def main():
    for cfg in COMPARISONS:
        run(cfg)
    print("\nDone.")


if __name__ == "__main__":
    main()
