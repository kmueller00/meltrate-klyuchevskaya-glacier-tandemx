#!/usr/bin/env python3
import sys; sys.path.insert(0,"/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/code/ash_analysis"); import _style
"""
STEP 28: Where exactly is melt largest/smallest, and what explains it there?

Uses summer_trend_rate_glacieronly.tif (2026-08-04, script 32: per-pixel robust
linear trend using every available summer 2012-2025 DEM at each pixel, 96.1%
glacier coverage) as the melt-rate field. SUPERSEDES the earlier two-bucket
early/late median-difference rate (summer_robust_rate_glacieronly.tif, 65.1%
coverage) -- the coverage diagnostic (2026-08-04, scratch coverage_diag.py)
showed the two-bucket estimator's ~35% gap was mostly an artifact of requiring
BOTH a pre-2018 AND post-2018 value at every pixel: the northern branches have
zero pre-2018 coverage on this track but full post-2018 coverage, and were
being dropped rather than estimated from the data that does exist. Only ~2%
of the glacier is a genuine zero-coverage gap. Where both estimators have data,
they agree closely (median difference -0.035 m/yr, see TREND_vs_TWOBUCKET_compare.png).
For each pixel computes: elevation, slope, aspect, curvature (all derived
from the same DEM used to build the rate grid), plus ash-cover fraction
(union of all four eruption ash masks, reprojected onto this grid).

Then:
  (1) Defines HOTSPOT (fastest thinning, rate <= 5th percentile) and
      COLDSPOT (slowest thinning / thickening, rate >= 95th percentile)
      pixel sets, RESTRICTED to pixels with a long (>=MIN_BASELINE_YR) trend
      baseline -- the newly-recovered northern-branch pixels have a much
      shorter (~4-7yr, post-2018-only) baseline and visibly noisier per-pixel
      rates (see PREVIEW_trend_rate.png), so including them in the extreme-
      percentile ranking would conflate short-baseline noise with genuine
      13-year thinning/thickening extremes. They still appear on the map,
      colored by rate, just excluded from the hot/coldspot definition itself.
  (2) Reports the elevation/slope/aspect/ash-cover distribution in each set
      vs the glacier-wide distribution (so "is it high, is it ash-covered,
      is it a particular aspect" is answered directly, not inferred).
  (3) Per-elevation-band (200 m), reports mean rate, mean slope, ash%, and
      the dominant aspect octant -- i.e. the full multi-variable picture at
      each elevation, not just the univariate hypsometric curve from step 21.
  (4) Partial correlations of rate vs {elevation, slope, ash%, aspect-sin/cos}
      pixel-wise, and partial (controlling for elevation) since elevation is
      the dominant confounder (step 23 already showed this for ash dose).
Outputs: HOTSPOT_COLDSPOT_map.png, ENVIRO_BY_ELEVATION.png,
         HOTSPOT_COLDSPOT_stats.csv, PARTIAL_CORR.csv
"""
import numpy as np, rasterio, rasterio.features, geopandas as gpd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path
from rasterio.warp import reproject, Resampling
from scipy import stats
import csv, warnings
warnings.filterwarnings("ignore")

AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation")
GLINV=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/code/ash_analysis/GLINV_main_massif.gpkg")
RATE_TIF=AOUT/"summer_trend_rate_glacieronly.tif"
BASELINE_TIF=AOUT/"summer_trend_baseline_years.tif"
MIN_BASELINE_YR=10.0   # hotspot/coldspot ranking restricted to long-baseline pixels
ASH_TIFS=[AOUT/"ash_analysis/ASHMASK_2019_Oct_moderate.tif",
          AOUT/"ash_analysis/ASHMASK_2020_Dec_moderate.tif",
          AOUT/"ash_analysis/ASHMASK_2022_Nov_moderate.tif",
          AOUT/"ash_analysis/ASHMASK_2023_PAROXYSM_moderate.tif"]
crs="EPSG:32657"; RES=30.0

with rasterio.open(RATE_TIF) as s:
    rate=s.read(1).astype("float32"); rate[rate==s.nodata]=np.nan
    tr=s.transform; rows,cols=s.shape
with rasterio.open(BASELINE_TIF) as s:
    baseline_yr=s.read(1).astype("float32"); baseline_yr[baseline_yr==s.nodata]=np.nan

shp=gpd.read_file(GLINV).to_crs(crs)
gm=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
    out_shape=(rows,cols),transform=tr,fill=0,dtype='uint8').astype(bool)

# reference elevation grid: script 32's multi-scene per-pixel mean elevation
# (summer_trend_elevation_mean.tif), not a single reference scene. A single
# scene's own footprint (previously 2024-09-04) only covers 72.3% of the
# glacier polygon -- missing almost entirely the northern branches, the same
# short-baseline area script 32 already had to specially handle for the rate
# itself. The multi-scene composite covers 97.3%, closing most of that gap.
with rasterio.open(RATE_TIF.parent/"summer_trend_elevation_mean.tif") as s:
    Z=s.read(1).astype("float32"); Z[Z==s.nodata]=np.nan
gy,gx=np.gradient(Z,RES)
slope_deg=np.degrees(np.arctan(np.hypot(gx,gy)))
aspect_deg=(np.degrees(np.arctan2(gx,-gy))+360)%360   # 0=N,90=E,180=S,270=W

# ash union (any of the 4 eruption masks flagged ash at this pixel)
ash_any=np.zeros((rows,cols),bool); ash_n=0
for at in ASH_TIFS:
    if not at.exists(): continue
    with rasterio.open(at) as s:
        a=s.read(1)
        o=np.zeros((rows,cols),"float32")
        reproject((a==1).astype("float32"),o,src_transform=s.transform,src_crs=s.crs,
                  dst_transform=tr,dst_crs=crs,resampling=Resampling.average)
    ash_any |= (o>0.3); ash_n+=1
print(f"ash union built from {ash_n} eruption masks")

valid=gm&np.isfinite(rate)&np.isfinite(Z)&np.isfinite(slope_deg)
print(f"valid glacier pixels: {valid.sum():,} / {gm.sum():,} ({100*valid.sum()/gm.sum():.1f}%)")

r=rate[valid]; z=Z[valid]; sl=slope_deg[valid]; asp=aspect_deg[valid]; ash=ash_any[valid]

# ---- (1) hotspot/coldspot definition, restricted to long-baseline pixels ----
# (see module docstring: short-baseline northern-branch pixels are noisier and
# would otherwise dominate the extreme percentiles without representing a real
# 13-year thinning/thickening extreme)
longbase=valid&np.isfinite(baseline_yr)&(baseline_yr>=MIN_BASELINE_YR)
print(f"long-baseline (>={MIN_BASELINE_YR:.0f}yr) pixels eligible for hotspot/coldspot ranking: "
      f"{longbase.sum():,} / {valid.sum():,} ({100*longbase.sum()/max(valid.sum(),1):.1f}%)")
p05,p95=np.nanpercentile(rate[longbase],[5,95])
hot=longbase & (rate<=p05)     # fastest thinning (most negative)
cold=longbase & (rate>=p95)    # slowest thinning / thickening
print(f"hotspot (rate<={p05:.2f} m/yr): {hot.sum():,} px | coldspot (rate>={p95:.2f} m/yr): {cold.sum():,} px")

def summarize(mask,name):
    zz=Z[mask]; ss=slope_deg[mask]; aa=aspect_deg[mask]; ashp=100*ash_any[mask].mean()
    # dominant aspect octant
    oct_names=["N","NE","E","SE","S","SW","W","NW"]
    octs=((aa+22.5)//45 %8).astype(int)
    dom=oct_names[np.bincount(octs,minlength=8).argmax()]
    return dict(set=name,n_px=int(mask.sum()),area_km2=mask.sum()*RES*RES/1e6,
                elev_mean=np.nanmean(zz),elev_med=np.nanmedian(zz),
                slope_mean=np.nanmean(ss),ash_pct=ashp,dominant_aspect=dom,
                rate_mean=np.nanmean(rate[mask]),baseline_yr_mean=np.nanmean(baseline_yr[mask]))

rows_sum=[summarize(valid,"whole glacier"),summarize(hot,"HOTSPOT (fastest thinning, <=P5)"),
          summarize(cold,"COLDSPOT (slowest/thickening, >=P95)")]
for rr in rows_sum:
    print(f"  {rr['set']:35s} n={rr['n_px']:>8,} area={rr['area_km2']:6.1f}km2 "
          f"elev={rr['elev_mean']:6.0f}m slope={rr['slope_mean']:5.1f}deg "
          f"ash={rr['ash_pct']:5.1f}% aspect={rr['dominant_aspect']:>2s} rate={rr['rate_mean']:+.2f}m/yr")

with open(AOUT/"HOTSPOT_COLDSPOT_stats.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows_sum[0].keys()),lineterminator="\n"); w.writeheader()
    for rr in rows_sum: w.writerow(rr)
print("-> HOTSPOT_COLDSPOT_stats.csv")

# ---- (3) per-elevation-band multi-variable picture ----
bands=np.arange(1300,4600,200); band_rows=[]
oct_names=["N","NE","E","SE","S","SW","W","NW"]
for b0 in bands:
    b1=b0+200; sel=valid&(Z>=b0)&(Z<b1)
    n=sel.sum()
    if n<50: continue
    octs=((aspect_deg[sel]+22.5)//45 %8).astype(int)
    dom=oct_names[np.bincount(octs,minlength=8).argmax()]
    band_rows.append(dict(elev_mid=b0+100,n_px=int(n),area_km2=n*RES*RES/1e6,
        rate_mean=np.nanmean(rate[sel]),rate_med=np.nanmedian(rate[sel]),
        slope_mean=np.nanmean(slope_deg[sel]),ash_pct=100*ash_any[sel].mean(),
        dominant_aspect=dom))
with open(AOUT/"ENVIRO_BY_ELEVATION.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(band_rows[0].keys()),lineterminator="\n"); w.writeheader()
    for rr in band_rows: w.writerow(rr)
print("-> ENVIRO_BY_ELEVATION.csv")

# ---- (4) partial correlations (rate vs each variable, raw and elevation-controlled) ----
def partial_corr(y,x,control):
    # residualize y and x on control (linear), then correlate residuals
    A=np.c_[control,np.ones_like(control)]
    by=np.linalg.lstsq(A,y,rcond=None)[0]; ry=y-A@by
    bx=np.linalg.lstsq(A,x,rcond=None)[0]; rx=x-A@bx
    return stats.pearsonr(rx,ry)

n_sub=min(200000,r.size)
rng=np.random.default_rng(0); idx=rng.choice(r.size,n_sub,replace=False)
rs,zs,sls,ashs=r[idx],z[idx],sl[idx],ash[idx].astype(float)
asp_s=asp[idx]; sin_a=np.sin(np.radians(asp_s)); cos_a=np.cos(np.radians(asp_s))

pc_rows=[]
raw_z=stats.pearsonr(zs,rs); pc_rows.append(dict(variable="elevation",raw_r=raw_z[0],raw_p=raw_z[1],partial_r=np.nan,partial_p=np.nan))
for name,var in [("slope",sls),("ash_cover",ashs),("aspect_sin",sin_a),("aspect_cos",cos_a)]:
    raw=stats.pearsonr(var,rs)
    part=partial_corr(rs,var,zs)
    pc_rows.append(dict(variable=name,raw_r=raw[0],raw_p=raw[1],partial_r=part[0],partial_p=part[1]))
    print(f"  {name:12s} raw r={raw[0]:+.3f} (p={raw[1]:.1e})  | partial|elev r={part[0]:+.3f} (p={part[1]:.1e})")
with open(AOUT/"PARTIAL_CORR.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(pc_rows[0].keys()),lineterminator="\n"); w.writeheader()
    for rr in pc_rows: w.writerow(rr)
print("-> PARTIAL_CORR.csv")

# ---- figure 1: hotspot/coldspot map ----
# Use real map-unit extent (not just pixel indices) so the full shapefile-delineated
# glacier outline can be drawn and any no-data gaps WITHIN the glacier are visually
# distinct from truly off-glacier terrain, instead of both reading as blank background.
import rasterio.transform
left,bottom,right,top=rasterio.transform.array_bounds(rows,cols,tr)
ext=[left,right,bottom,top]

fig,ax=plt.subplots(figsize=(9,10))
# full glacier extent (shapefile-rasterized mask gm), filled neutral first so the
# TRUE outline is always visible even where rate/Z/slope data is missing
gm_show=np.where(gm,1,np.nan)
ax.imshow(gm_show,cmap=ListedColormap(["#c9c2b4"]),extent=ext,zorder=1)
show=np.where(valid,rate,np.nan)
im=ax.imshow(show,cmap='RdBu',vmin=-2,vmax=2,extent=ext,zorder=2)
plt.colorbar(im,ax=ax,shrink=0.7,label="dH rate 2012→2025 [m/yr]")
hot_show=np.where(hot,1,np.nan); cold_show=np.where(cold,1,np.nan)
ax.imshow(hot_show,cmap=ListedColormap(["#000000"]),alpha=0.55,extent=ext,zorder=3)
ax.imshow(cold_show,cmap=ListedColormap(["#00ff00"]),alpha=0.55,extent=ext,zorder=3)
# shapefile boundary outline, drawn directly from the source polygons (not the
# rasterized mask), so the true glacier delineation is unambiguous
shp.boundary.plot(ax=ax,color='k',linewidth=0.6,zorder=4)
gap_px=int((gm&~valid).sum())
ax.set_title(f"Klyuchevskaya main massif: HOTSPOT (black, fastest thinning ≤P5={p05:.2f}m/yr)\n"
             f"vs COLDSPOT (green, slowest/thickening ≥P95={p95:.2f}m/yr), among ≥{MIN_BASELINE_YR:.0f}yr-baseline pixels\n"
             f"tan = full glacier extent (shapefile); {gap_px:,} glacier px ({100*gap_px/max(gm.sum(),1):.1f}%) lack valid rate/elevation data "
             f"(per-pixel trend fit, 2026-08-04; short-baseline branches shown but excluded from hot/coldspot ranking)")
ax.set_xlim(left,right); ax.set_ylim(bottom,top)
ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig(AOUT/"HOTSPOT_COLDSPOT_map.png",dpi=600,bbox_inches="tight")
print(f"-> HOTSPOT_COLDSPOT_map.png ({gap_px:,} glacier px lacking data, {100*gap_px/max(gm.sum(),1):.1f}%)")

# ---- figure 2: enviro-by-elevation, 4 panel ----
el=[b['elev_mid'] for b in band_rows]; rt=[b['rate_mean'] for b in band_rows]
slm=[b['slope_mean'] for b in band_rows]; ashp=[b['ash_pct'] for b in band_rows]
area=[b['area_km2'] for b in band_rows]
fig,axes=plt.subplots(1,4,figsize=(18,7),sharey=True)
axes[0].barh(el,area,height=180,color="#8ab4d2",edgecolor="#3a6d90",lw=0.5)
axes[0].set_xlabel("area [km²]"); axes[0].set_title("hypsometry"); axes[0].set_ylabel("elevation [m a.s.l.]")
axes[1].axvline(0,color='k',lw=0.8,ls=':')
axes[1].plot(rt,el,'o-',color="#2f7cb2"); axes[1].set_xlabel("mean dH rate [m/yr]"); axes[1].set_title("melt rate")
axes[2].barh(el,slm,height=180,color="#7a8c7d",edgecolor="#3f4a40",lw=0.5)
axes[2].set_xlabel("mean slope [deg]"); axes[2].set_title("terrain slope")
axes[3].barh(el,ashp,height=180,color="#c9a227",edgecolor="#7d6416",lw=0.5)
axes[3].set_xlabel("ash-cover [%]"); axes[3].set_title("tephra cover (any eruption)")
fig.suptitle("Klyuchevskaya — environmental variables by elevation band (long-term 2012–2025 rate)",fontsize=13)
fig.tight_layout(); fig.savefig(AOUT/"ENVIRO_BY_ELEVATION.png",dpi=600,bbox_inches="tight")
print("-> ENVIRO_BY_ELEVATION.png")
