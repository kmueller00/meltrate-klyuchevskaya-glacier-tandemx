#!/usr/bin/env python3
"""
STEP 34: Delineate the HOTSPOT (script 28's fastest-thinning zone) from an
ice-dynamics (surge) explanation, using ITS_LIVE glacier surface velocity.

Motivation: the hotspot (rate<=P5 among long-baseline pixels, mean elev 3241m,
steep, SE-facing) has only been tested against ash-cover/elevation/slope/aspect
so far. A surge (rapid ice outflow) would read as "thinning" in an elevation-
change map without being a real ablation/climate signal. This script checks
whether the hotspot is measurably FASTER-FLOWING than the glacier background
(spatial test) and whether its flow shows an episodic acceleration coincident
with any known event (temporal test) -- either would argue for a dynamics
explanation over a pure-melt one. Geothermal delineation (thermal-anomaly
co-location) is a separate follow-on, out of scope here -- this is velocity only.

Data: ITS_LIVE (NASA/JPL) via the public STAC catalog (stac.itslive.cloud,
collection 'itslive-cubes'), same pystac_client pattern as s2_util.py. Cubes
are Zarr on public S3 (anon access), CRS EPSG:3413, ~120m res, one row per
image-pair (irregular cadence, not a clean time series -- aggregate by year).
The glacier straddles TWO adjacent datacube tiles; each is processed
independently (its own reprojected mask) and pixel values pooled, rather than
merging two separate Zarr stores into one grid.
"""
import sys; sys.path.insert(0,"/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/code/ash_analysis"); import _style
import numpy as np, pandas as pd, rasterio, rasterio.features, geopandas as gpd
import xarray as xr, s3fs, pystac_client
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
import warnings; warnings.filterwarnings("ignore")

AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation")
GLINV=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/code/ash_analysis/GLINV_main_massif.gpkg")
RATE_TIF=AOUT/"summer_trend_rate_glacieronly.tif"
BASELINE_TIF=AOUT/"summer_trend_baseline_years.tif"
MIN_BASELINE_YR=10.0
crs="EPSG:32657"
BBOX=[159.889,55.472,161.359,56.522]

# eruption record, reused from 04_combined_signal_2015.py
ERUPT=[(date(2015,1,2),date(2015,3,24),"2015 Strombolian"),
 (date(2015,8,27),date(2015,9,10),"2015 Aug"),
 (date(2016,4,1),date(2016,5,1),"2016 Apr explosive"),
 (date(2019,4,1),date(2019,11,1),"2019 eruption"),
 (date(2020,10,2),date(2021,3,25),"2020-21 cycle"),
 (date(2022,11,20),date(2022,12,20),"2022 Nov"),
 (date(2023,6,22),date(2024,1,15),"2023-24 PAROXYSM"),
 (date(2024,8,1),date(2024,10,15),"2024 Aug-Oct")]

# ── (1) hotspot/coldspot masks: verbatim reuse of 28_hotspot_coldspot_analysis.py ──
with rasterio.open(RATE_TIF) as s:
    rate=s.read(1).astype("float32"); rate[rate==s.nodata]=np.nan
    tr=s.transform; rows,cols=s.shape
with rasterio.open(BASELINE_TIF) as s:
    baseline_yr=s.read(1).astype("float32"); baseline_yr[baseline_yr==s.nodata]=np.nan

shp=gpd.read_file(GLINV).to_crs(crs)
gm=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
    out_shape=(rows,cols),transform=tr,fill=0,dtype="uint8").astype(bool)

valid=gm&np.isfinite(rate)
longbase=valid&np.isfinite(baseline_yr)&(baseline_yr>=MIN_BASELINE_YR)
p05,p95=np.nanpercentile(rate[longbase],[5,95])
hot=longbase&(rate<=p05)
cold=longbase&(rate>=p95)
print(f"hotspot (rate<={p05:.2f} m/yr): {hot.sum():,} px | coldspot (rate>={p95:.2f} m/yr): {cold.sum():,} px | glacier valid: {valid.sum():,} px")

# ── (2) find ITS_LIVE tiles overlapping the glacier, open each independently ──
CAT=pystac_client.Client.open("https://stac.itslive.cloud")
items=list(CAT.search(collections=["itslive-cubes"],bbox=BBOX).items())
gl3413=shp.to_crs("EPSG:3413"); gminx,gminy,gmaxx,gmaxy=gl3413.total_bounds

overlapping=[]
for it in items:
    x0,y0,x1,y1=it.properties["proj:bbox"]
    if not (gmaxx<x0 or gminx>x1 or gmaxy<y0 or gminy>y1):
        overlapping.append(it)
print(f"ITS_LIVE tiles overlapping glacier: {len(overlapping)}")

fs=s3fs.S3FileSystem(anon=True)

def reproject_mask_to_tile(mask_bool, tile_tr, tile_rows, tile_cols):
    dst=np.zeros((tile_rows,tile_cols),dtype="uint8")
    reproject(mask_bool.astype("uint8"),dst,src_transform=tr,src_crs=crs,
              dst_transform=tile_tr,dst_crs="EPSG:3413",resampling=Resampling.nearest)
    return dst.astype(bool)

# accumulate: pooled (year -> list of hotspot pixel velocity values) across tiles
hot_by_year={}; cold_by_year={}; glacier_all_v=[]; hot_all_v=[]; cold_all_v=[]
tile_maps=[]  # for the figure: (x, y, v_median_map, hot_mask, cold_mask, gm_mask)

for it in overlapping:
    s3path=it.assets["zarr"].href.replace("https://its-live-data.s3.amazonaws.com/","its-live-data/")
    store=fs.get_mapper(s3path)
    ds=xr.open_zarr(store,consolidated=True,decode_timedelta=True)

    x=ds.x.values; y=ds.y.values
    res_x=float(x[1]-x[0]); res_y=float(y[1]-y[0])
    tile_tr=Affine(res_x,0,x[0]-res_x/2, 0,res_y,y[0]-res_y/2)
    tile_rows,tile_cols=len(y),len(x)

    gm_t=reproject_mask_to_tile(gm,tile_tr,tile_rows,tile_cols)
    hot_t=reproject_mask_to_tile(hot,tile_tr,tile_rows,tile_cols)
    cold_t=reproject_mask_to_tile(cold,tile_tr,tile_rows,tile_cols)
    print(f"  {it.id}: glacier px on this tile={gm_t.sum()} hot={hot_t.sum()} cold={cold_t.sum()}")
    if gm_t.sum()==0:
        continue

    # crop to glacier bbox (+buffer) BEFORE any heavy compute -- full cube is huge
    ys,xs=np.where(gm_t)
    pad=5
    y0,y1=max(ys.min()-pad,0),min(ys.max()+pad,tile_rows-1)
    x0,x1=max(xs.min()-pad,0),min(xs.max()+pad,tile_cols-1)
    v_crop=ds["v"].isel(y=slice(y0,y1+1),x=slice(x0,x1+1))
    gm_c=gm_t[y0:y1+1,x0:x1+1]; hot_c=hot_t[y0:y1+1,x0:x1+1]; cold_c=cold_t[y0:y1+1,x0:x1+1]

    mid_dates=pd.to_datetime(ds.mid_date.values)
    years=mid_dates.year

    # per-pixel median velocity across all time (a "typical flow speed" map for the figure)
    v_med_map=v_crop.median(dim="mid_date",skipna=True).compute().values
    tile_crop_tr=Affine(res_x,0,x[x0]-res_x/2, 0,res_y,y[y0]-res_y/2)
    tile_maps.append((x[x0:x1+1],y[y0:y1+1],v_med_map,hot_c,cold_c,gm_c,tile_crop_tr))

    glacier_all_v.append(v_med_map[gm_c][np.isfinite(v_med_map[gm_c])])
    hot_all_v.append(v_med_map[hot_c][np.isfinite(v_med_map[hot_c])])
    cold_all_v.append(v_med_map[cold_c][np.isfinite(v_med_map[cold_c])])

    # per-year median velocity within hotspot/coldspot, for the time series
    for yr in sorted(set(years)):
        sel=np.where(years==yr)[0]
        if sel.size==0: continue
        vals=v_crop.isel(mid_date=sel).compute().values   # (n_pairs_this_year, ny, nx)
        hv=vals[:,hot_c]; cv=vals[:,cold_c]
        hv=hv[np.isfinite(hv)]; cv=cv[np.isfinite(cv)]
        if hv.size: hot_by_year.setdefault(yr,[]).append(hv)
        if cv.size: cold_by_year.setdefault(yr,[]).append(cv)
    print(f"    processed {it.id}: {len(years)} image pairs, {len(set(years))} distinct years")

glacier_all_v=np.concatenate(glacier_all_v) if glacier_all_v else np.array([])
hot_all_v=np.concatenate(hot_all_v) if hot_all_v else np.array([])
cold_all_v=np.concatenate(cold_all_v) if cold_all_v else np.array([])

# ── (3) spatial test ──────────────────────────────────────────────────────
print("\n=== SPATIAL TEST: median flow speed [m/yr] ===")
print(f"  whole glacier: n={glacier_all_v.size} median={np.median(glacier_all_v):.1f} "
      f"p90={np.percentile(glacier_all_v,90):.1f}")
print(f"  HOTSPOT:       n={hot_all_v.size} median={np.median(hot_all_v):.1f} "
      f"p90={np.percentile(hot_all_v,90):.1f}")
print(f"  COLDSPOT:      n={cold_all_v.size} median={np.median(cold_all_v):.1f} "
      f"p90={np.percentile(cold_all_v,90):.1f}")

# ── (4) temporal test: hotspot/coldspot median velocity per year ────────────
hot_series=sorted((yr,float(np.median(np.concatenate(v)))) for yr,v in hot_by_year.items())
cold_series=sorted((yr,float(np.median(np.concatenate(v)))) for yr,v in cold_by_year.items())
print("\n=== TEMPORAL TEST: hotspot median velocity by year ===")
for yr,v in hot_series: print(f"  {yr}: {v:.1f} m/yr")

# ── outputs ──────────────────────────────────────────────────────────────
with open(AOUT/"SURGE_DELINEATION_stats.csv","w") as f:
    f.write("region,n_px,median_v_m_yr,p90_v_m_yr\n")
    for name,v in [("whole_glacier",glacier_all_v),("hotspot",hot_all_v),("coldspot",cold_all_v)]:
        if v.size:
            f.write(f"{name},{v.size},{np.median(v):.2f},{np.percentile(v,90):.2f}\n")
    f.write("\nyear,hotspot_median_v_m_yr,coldspot_median_v_m_yr\n")
    cold_d=dict(cold_series)
    for yr,v in hot_series:
        f.write(f"{yr},{v:.2f},{cold_d.get(yr,float('nan')):.2f}\n")
print("\n-> SURGE_DELINEATION_stats.csv")

# merge all tiles onto ONE common grid (the project's own EPSG:32657 grid,
# same as every other figure in this report) instead of plotting each
# ITS_LIVE tile as a separate panel in its own native pixel space -- with
# the glacier straddling two tiles, that previously showed most of the
# glacier in one panel and a small, seemingly disconnected fragment in the
# other, cut off rather than a single coherent map.
v_merged=np.full((rows,cols),np.nan,"float32")
for tx,ty,vmap,hmask,cmask,gmask,tile_crop_tr in tile_maps:
    v_src=np.where(gmask,vmap,np.nan).astype("float32")
    v_dst=np.full((rows,cols),np.nan,"float32")
    reproject(v_src,v_dst,src_transform=tile_crop_tr,src_crs="EPSG:3413",
              dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear)
    fill=np.isnan(v_merged)&np.isfinite(v_dst)
    v_merged[fill]=v_dst[fill]

left,bottom,right,top=rasterio.transform.array_bounds(rows,cols,tr)
ext=[left,right,bottom,top]
fig,ax=plt.subplots(figsize=(10,10))
im=ax.imshow(v_merged,cmap="viridis",vmin=0,
             vmax=np.nanpercentile(v_merged,95) if np.isfinite(v_merged).any() else 1,extent=ext)
ax.contour(hot.astype(float),levels=[0.5],colors="red",extent=ext,linewidths=1.2)
ax.contour(cold.astype(float),levels=[0.5],colors="cyan",extent=ext,linewidths=1.2)
shp.boundary.plot(ax=ax,color='k',linewidth=0.5)
ax.set_xlim(left,right); ax.set_ylim(bottom,top)
fig.colorbar(im,ax=ax,label="median flow speed [m/yr]",fraction=0.046)
ax.set_title(f"ITS_LIVE flow speed (median all-time), {len(tile_maps)} tiles merged\nred=hotspot outline, cyan=coldspot outline")
fig.tight_layout(); fig.savefig(AOUT/"SURGE_DELINEATION_velocity_map.png",dpi=600,bbox_inches="tight")
print("-> SURGE_DELINEATION_velocity_map.png")

fig,ax=plt.subplots(figsize=(12,6))
for s,e,lab in ERUPT:
    ax.axvspan(s.year+ (s.timetuple().tm_yday/365.25), e.year+(e.timetuple().tm_yday/365.25),
               color="orange",alpha=0.15,zorder=0)
if hot_series:
    yrs,vs=zip(*hot_series); ax.plot(yrs,vs,"o-",color="#a8481f",label="hotspot median v")
if cold_series:
    yrs,vs=zip(*cold_series); ax.plot(yrs,vs,"o-",color="#1f6f78",label="coldspot median v")
ax.set_ylabel("median flow speed [m/yr]"); ax.set_xlabel("year")
ax.set_title("Klyuchevskaya hotspot/coldspot flow speed vs. time\n(orange bands = eruption windows)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(AOUT/"SURGE_DELINEATION_timeseries.png",dpi=600,bbox_inches="tight")
print("-> SURGE_DELINEATION_timeseries.png")
