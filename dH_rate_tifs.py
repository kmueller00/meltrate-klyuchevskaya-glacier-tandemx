#!/usr/bin/env python3
"""
Per-year elevation-change RATE rasters (m/yr) over GLACIER areas.

For each epoch (summer, winter) reads the coreg-corrected dH tif,
divides by the epoch time-span (yr) -> per-pixel rate (m/yr),
masks to glacier pixels (GLINV/RGI), and writes:
   diff_<epoch>_rate_m_per_yr.tif            (all pixels, glacier kept)
   diff_<epoch>_rate_m_per_yr_glacieronly.tif (non-glacier set to NoData)
Also prints glacier-mean rate +/- uncertainty.
"""
import numpy as np, rasterio, rasterio.features, geopandas as gpd
from pathlib import Path
from datetime import date

OUT = Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out")
GLINV = Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b/"
             "2025/2025-01-14_065954_TDT_SM_A_97516_155_0045+1-2/prc05_refdem/"
             "GLINV_rgi06_ttl_utm57_N_b_UTM.shp")
RES = 30.0
CORR_LENGTH_M = 500.0
ND = -9999.0
NSIG = 3.0          # outlier cutoff: |dH - median| > NSIG * NMAD  (on glacier pixels)

EPOCHS = {
  "summer": dict(early=[date(2013,7,15),date(2014,7,2)],
                 late =[date(2019,6,20),date(2019,7,1)],
                 tif=OUT/"diff_summer_dH_corrected.tif"),
  "winter": dict(early=[date(2012,2,13),date(2017,2,12)],
                 late =[date(2024,2,8), date(2025,2,16)],
                 tif=OUT/"diff_winter_dH_corrected.tif"),
}
mean_ord = lambda ds: np.mean([d.toordinal() for d in ds])

for name,e in EPOCHS.items():
    dt = (mean_ord(e["late"])-mean_ord(e["early"]))/365.25
    with rasterio.open(e["tif"]) as src:
        dh = src.read(1).astype("float32"); prof = src.profile
        nd = src.nodata
        shp = gpd.read_file(GLINV).to_crs(src.crs)
        gmask = rasterio.features.rasterize(
            [(g,1) for g in shp.geometry if g is not None],
            out_shape=(src.height,src.width), transform=src.transform,
            fill=0, dtype="uint8").astype(bool)
    dh = np.where((nd is not None) & (dh==nd), np.nan, dh)

    # ── outlier rejection on GLACIER dH (robust median/NMAD, NSIG cutoff) ──────
    gvals = dh[gmask & np.isfinite(dh)]
    g_med = np.median(gvals)
    g_nmad = 1.4826 * np.median(np.abs(gvals - g_med))
    lo, hi = g_med - NSIG*g_nmad, g_med + NSIG*g_nmad
    outlier = np.isfinite(dh) & ((dh < lo) | (dh > hi))   # pixels to drop
    n_out = int((outlier & gmask).sum())
    dh_clean = np.where(outlier, np.nan, dh)              # outliers -> NaN

    rate = dh_clean / dt                     # m/yr per pixel (outliers removed)
    prof.update(dtype="float32", nodata=ND, compress="lzw")

    # (1) full rate raster (outliers removed)
    full = np.where(np.isfinite(rate), rate, ND).astype("float32")
    with rasterio.open(OUT/f"diff_{name}_rate_m_per_yr.tif","w",**prof) as dst:
        dst.write(full,1)
    # (2) glacier-only rate raster (outliers removed)
    glac = np.where(gmask & np.isfinite(rate), rate, ND).astype("float32")
    with rasterio.open(OUT/f"diff_{name}_rate_m_per_yr_glacieronly.tif","w",**prof) as dst:
        dst.write(glac,1)

    g = rate[gmask & np.isfinite(rate)]
    med=np.median(g); nmad=1.4826*np.median(np.abs(g-med)); mean=np.mean(g)
    n_eff=max(1.0, g.size*RES*RES/(np.pi*CORR_LENGTH_M**2))
    err=(nmad/np.sqrt(n_eff))
    print(f"[{name}] dt={dt:.2f} yr  glacier px(kept)={g.size:,}  "
          f"outliers_dropped={n_out:,} (|dH-{g_med:+.2f}|>{NSIG}*{g_nmad:.2f}m, "
          f"range [{lo:+.1f},{hi:+.1f}] m)")
    print(f"   rate median = {med:+.3f} +/- {err:.3f} m/yr   mean = {mean:+.3f} m/yr")
    print(f"   -> diff_{name}_rate_m_per_yr.tif  &  _glacieronly.tif")
print("\nDONE")
