#!/usr/bin/env python3
"""
Yearly elevation-change rates (m/yr ± uncertainty) on GLACIER pixels only.
Reads the coreg-corrected dH tifs + glacier mask from prc07_overview_out.

Rate = dH / dt, where dt = (late epoch mean date) - (early epoch mean date).
Uncertainty on the glacier-mean rate combines:
  - per-pixel NMAD scaled to standard error over N_eff (spatial autocorrelation),
  - divided by dt.
"""
import numpy as np, rasterio, rasterio.features, geopandas as gpd
from pathlib import Path
from datetime import date

OUT = Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out")
GLINV = Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b/"
             "2025/2025-01-14_065954_TDT_SM_A_97516_155_0045+1-2/prc05_refdem/"
             "GLINV_rgi06_ttl_utm57_N_b_UTM.shp")
RES = 30.0
CORR_LENGTH_M = 500.0  # typical TDX dH spatial autocorrelation length for N_eff

# epoch mean dates (mean of acquisition dates per epoch) — match diff_dem_calc.py
EPOCHS = {
  "summer": dict(
    early=[date(2013,7,15), date(2014,7,2)],
    late =[date(2019,6,20), date(2019,7,1)],
    tif=OUT/"diff_summer_dH_corrected.tif"),
  "winter": dict(
    early=[date(2012,2,13), date(2017,2,12)],
    late =[date(2024,2,8),  date(2025,2,16)],
    tif=OUT/"diff_winter_dH_corrected.tif"),
}

def mean_date(ds):
    return date.fromordinal(int(np.mean([d.toordinal() for d in ds])))

def glacier_mask(tif):
    with rasterio.open(tif) as src:
        shp = gpd.read_file(GLINV).to_crs(src.crs)
        m = rasterio.features.rasterize(
            [(g,1) for g in shp.geometry if g is not None],
            out_shape=(src.height,src.width), transform=src.transform,
            fill=0, dtype="uint8").astype(bool)
    return m

print(f"{'epoch':8s} {'dt_yr':>6s} {'N_glac':>9s} "
      f"{'rate_med':>10s} {'rate_mean':>10s} {'± (m/yr)':>9s}")
print("-"*60)
results={}
for name,e in EPOCHS.items():
    dt = (mean_date(e["late"]).toordinal()-mean_date(e["early"]).toordinal())/365.25
    with rasterio.open(e["tif"]) as src:
        dh = src.read(1).astype(float); nd=src.nodata
    if nd is not None: dh[dh==nd]=np.nan
    gm = glacier_mask(e["tif"])
    g = dh[gm & np.isfinite(dh)]
    if g.size==0:
        print(f"{name:8s}  no glacier pixels"); continue
    med=np.median(g); nmad=1.4826*np.median(np.abs(g-med)); mean=np.mean(g)
    # effective sample size for autocorrelation
    pix_area=RES*RES; corr_area=np.pi*CORR_LENGTH_M**2
    n_eff=max(1.0, g.size*pix_area/corr_area)
    se=nmad/np.sqrt(n_eff)               # std error of mean dH (m)
    rate_med=med/dt; rate_mean=mean/dt; rate_err=se/dt
    results[name]=(rate_med,rate_mean,rate_err,dt,g.size)
    print(f"{name:8s} {dt:6.2f} {g.size:9,d} "
          f"{rate_med:+10.3f} {rate_mean:+10.3f} {rate_err:9.3f}")
print("-"*60)
print("\nInterpretation (glacier pixels only, coreg-corrected):")
for name,(rm,rmean,re,dt,n) in results.items():
    print(f"  {name}: {rm:+.3f} ± {re:.3f} m/yr (median)  over {dt:.1f} yr, {n:,} px")
