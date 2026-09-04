#!/usr/bin/env python3
"""
STEP 37: Two follow-ups on script 36's geothermal check, both requested after
seeing the +25C Apakhonchich flank-zone anomaly:

  (1) PER-SEASON ROBUSTNESS: script 36's anomaly was computed from all 13
      winters (2013-2026) pooled into one climatology. Is the flank zone
      actually warmer than same-elevation terrain in EVERY individual winter,
      or is the pooled number dominated by one or two unusually warm seasons
      (e.g. active eruptive years depositing warm ash)? Recomputes the same
      elevation-restricted flank-vs-rest comparison separately per winter
      season (Dec-Feb, 8-21 scenes/season).
  (2) UNBIASED GLACIER-WIDE SCAN: script 36 only tested the one *named*
      Apakhonchich vent zone. This drops that a-priori wedge and instead
      computes an elevation-band-relative thermal anomaly at EVERY pixel on
      the grid (not just the flank sector), clusters contiguous anomalous-warm
      areas (scipy.ndimage.label), and reports each cluster's location,
      elevation, bearing/distance from the summit, and -- critically -- what
      fraction of it actually sits on glacier ice, in the melt HOTSPOT, or in
      the COLDSPOT. Also computes a single pixel-wise partial correlation
      (controlling for elevation, same method as script 28) between thermal
      anomaly and thinning rate across the whole glaciated area, to test
      directly whether "warmer than expected for its elevation" predicts
      "melting faster than expected" anywhere on the ice, not just at
      Apakhonchich.

Outputs: GEOTHERMAL_SEASONAL_robustness.csv, GEOTHERMAL_footprint_scan_map.png,
         GEOTHERMAL_footprint_scan_stats.csv
"""
import sys; sys.path.insert(0,"/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/code/ash_analysis"); import _style
import glob, csv, numpy as np, rasterio, rasterio.features, rasterio.transform, geopandas as gpd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from pyproj import Transformer
from rasterio.warp import reproject, Resampling
from scipy import ndimage, stats
import pystac_client, planetary_computer, odc.stac
from s2_util import QA_EXCLUDE_MASK
import warnings; warnings.filterwarnings("ignore")

AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation")
GLINV=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/code/ash_analysis/GLINV_main_massif.gpkg")
BASE=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
RATE_TIF=AOUT/"summer_trend_rate_glacieronly.tif"
BASELINE_TIF=AOUT/"summer_trend_baseline_years.tif"
MIN_BASELINE_YR=10.0
crs="EPSG:32657"

SUMMIT_LONLAT=(160.644089,56.056044)
FLANK_BEARING=(100.0,170.0); FLANK_ELEV=(3400.0,4500.0); FLANK_MAXDIST=8000.0
ANOM_CLUSTER_MIN_PX=8          # ~0.7 ha at 30m -- drop single/double-pixel noise
ANOM_PERCENTILE=99.0           # "anomalously warm" = top 1% of elevation-band-relative anomaly

BBOX=[159.889,55.472,161.359,56.522]
CAT=pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1',
                              modifier=planetary_computer.sign_inplace)
ST_SCALE,ST_OFFSET=0.00341802,149.0

# ---- (0) same rate/baseline/glacier/hotspot/coldspot setup as script 28/36 ----
with rasterio.open(RATE_TIF) as s:
    tr=s.transform; rows,cols=s.shape; rate=s.read(1).astype("float32"); rate[rate==s.nodata]=np.nan
with rasterio.open(BASELINE_TIF) as s:
    baseline_yr=s.read(1).astype("float32"); baseline_yr[baseline_yr==s.nodata]=np.nan
shp=gpd.read_file(GLINV).to_crs(crs)
gm=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
    out_shape=(rows,cols),transform=tr,fill=0,dtype="uint8").astype(bool)
valid=gm&np.isfinite(rate)
longbase=valid&np.isfinite(baseline_yr)&(baseline_yr>=MIN_BASELINE_YR)
p05,p95=np.nanpercentile(rate[longbase],[5,95])
hot=longbase&(rate<=p05); cold=longbase&(rate>=p95)

# 7-scene median elevation mosaic (98.3% grid coverage) instead of a single
# reference scene (63.5% coverage) -- a single scene's own footprint silently
# excluded over a third of the surrounding area from the geothermal scan,
# undermining the "unbiased, whole-area" scan this script does.
with rasterio.open(AOUT/"FULL_GRID_elevation_mosaic.tif") as s:
    elev=s.read(1).astype("float32"); elev[elev==s.nodata]=np.nan

xs=np.arange(cols)*tr.a+tr.c+tr.a/2; ys=np.arange(rows)*tr.e+tr.f+tr.e/2
XX,YY=np.meshgrid(xs,ys)
tfm_fwd=Transformer.from_crs("EPSG:4326",crs,always_xy=True)
tfm_inv=Transformer.from_crs(crs,"EPSG:4326",always_xy=True)
x0,y0=tfm_fwd.transform(*SUMMIT_LONLAT)
dx,dy=XX-x0,YY-y0
bearing=np.degrees(np.arctan2(dx,dy))%360.0
dist=np.hypot(dx,dy)
flank=((bearing>=FLANK_BEARING[0])&(bearing<=FLANK_BEARING[1])
       &(dist<=FLANK_MAXDIST)&(elev>=FLANK_ELEV[0])&(elev<=FLANK_ELEV[1]))
print(f"hotspot px={hot.sum():,}  flank-zone px={flank.sum():,}  glacier px={gm.sum():,}")

# ---- fetch all winter Landsat items once, keep per-item datetime for grouping ----
# platform restricted to landsat-8/9 -- Landsat 7 uses a single 'lwir' thermal band
# (different name AND different bandpass/calibration than 8/9's dual lwir10/lwir11),
# and odc.stac's band-alias resolver can outright fail ("No such band/alias: lwir11")
# when a batch mixes it in. Same fail-closed platform filter s2_util.py already uses
# for the optical Landsat fetch, applied here for the same reason.
it_all=list(CAT.search(collections=['landsat-c2-l2'],bbox=BBOX,
    datetime="2013-01-01/2026-06-01",query={'eo:cloud_cover':{'lt':60}}).items())
it_all=[i for i in it_all if i.datetime.month in (12,1,2)
        and i.properties.get('platform') in ('landsat-8','landsat-9')]
def season_of(dt): return dt.year+1 if dt.month==12 else dt.year
seasons=sorted(set(season_of(i.datetime) for i in it_all))
print(f"winter Landsat 8/9 scenes: {len(it_all)} across {len(seasons)} seasons ({seasons[0]}-{seasons[-1]})")

def load_thermal(items):
    if not items: return None,None
    ds=odc.stac.load(items,bands=["lwir11","qa_pixel"],bbox=BBOX,resolution=30.0,
                     crs=crs,groupby="solar_day",chunks={})
    st_dn=ds["lwir11"]; qa=ds["qa_pixel"]
    st_k=(st_dn.astype("float32")*ST_SCALE+ST_OFFSET).where(((qa&QA_EXCLUDE_MASK)==0)&(st_dn>0))
    med=st_k.median(dim="time",skipna=True); n_obs=(~st_k.isnull()).sum(dim="time")
    out_t=np.full((rows,cols),np.nan,"float32"); out_n=np.full((rows,cols),0,"int32")
    reproject(np.asarray(med.values,"float32"),out_t,src_transform=ds.odc.transform,src_crs=crs,
              dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear)
    reproject(np.asarray(n_obs.values,"int32"),out_n,src_transform=ds.odc.transform,src_crs=crs,
              dst_transform=tr,dst_crs=crs,resampling=Resampling.nearest)
    return out_t,out_n

# ---- (1) per-season robustness: flank-elevation-band vs rest, one winter at a time ----
print("\n=== (1) PER-SEASON ROBUSTNESS: flank zone vs same-elevation-band rest, one winter at a time ===")
season_rows=[]
elevband_all=(elev>=FLANK_ELEV[0])&(elev<=FLANK_ELEV[1])
for yr in seasons:
    items=[i for i in it_all if season_of(i.datetime)==yr]
    t_k,n_obs=load_thermal(items)
    if t_k is None: continue
    t_c=t_k-273.15
    ok=np.isfinite(t_k)
    f_px=ok&flank; r_px=ok&elevband_all&(~flank)
    if f_px.sum()<3 or r_px.sum()<10:
        print(f"  {yr}: insufficient coverage (flank n={f_px.sum()}, rest n={r_px.sum()}) -- skipped")
        season_rows.append((yr,len(items),int(f_px.sum()),np.nan,int(r_px.sum()),np.nan,np.nan))
        continue
    tf,trr=np.nanmedian(t_c[f_px]),np.nanmedian(t_c[r_px])
    print(f"  {yr} ({len(items):2d} scenes): flank n={f_px.sum():3d} medianT={tf:+6.2f}C | "
          f"rest(same elev) n={r_px.sum():4d} medianT={trr:+6.2f}C | anomaly={tf-trr:+.2f}C")
    season_rows.append((yr,len(items),int(f_px.sum()),float(tf),int(r_px.sum()),float(trr),float(tf-trr)))

valid_anoms=[r[6] for r in season_rows if np.isfinite(r[6])]
n_warm=sum(1 for a in valid_anoms if a>0)
print(f"\n{n_warm}/{len(valid_anoms)} seasons with usable data show a WARM flank-zone anomaly "
      f"(range {min(valid_anoms):+.1f} to {max(valid_anoms):+.1f}C)" if valid_anoms else "no seasons had usable data")

with open(AOUT/"ash_analysis/GEOTHERMAL_SEASONAL_robustness.csv","w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["winter_season","n_scenes","flank_n","flank_medianT_C","rest_n","rest_medianT_C","anomaly_C"])
    for r in season_rows: w.writerow(r)
print("-> GEOTHERMAL_SEASONAL_robustness.csv")

# ---- (2) unbiased glacier-wide scan: full climatology, no a-priori flank wedge ----
print("\n=== (2) UNBIASED GLACIER-WIDE THERMAL ANOMALY SCAN (all 13 winters pooled) ===")
temp_k,n_obs=load_thermal(it_all)
temp_c=temp_k-273.15
valid_t=np.isfinite(temp_k)&(n_obs>=3)&np.isfinite(elev)
print(f"climatology valid px (>=3 obs): {valid_t.sum():,}")

bands=np.arange(int(np.nanmin(elev[valid_t])//200*200),int(np.nanmax(elev[valid_t])//200*200)+200,200)
anomaly=np.full((rows,cols),np.nan,"float32")
for b0 in bands:
    b1=b0+200
    inb=valid_t&(elev>=b0)&(elev<b1)
    if inb.sum()<20: continue
    anomaly[inb]=temp_c[inb]-np.nanmedian(temp_c[inb])

thr=np.nanpercentile(anomaly[valid_t],ANOM_PERCENTILE)
print(f"anomalously warm threshold (P{ANOM_PERCENTILE:.0f} of elevation-band-relative anomaly): +{thr:.2f}C")
warm=valid_t&(anomaly>=thr)
lbl,nlbl=ndimage.label(warm)
print(f"{nlbl} connected warm clusters found (before size filtering)")

cluster_rows=[]
for cid in range(1,nlbl+1):
    m=lbl==cid; n=int(m.sum())
    if n<ANOM_CLUSTER_MIN_PX: continue
    cy,cx=ndimage.center_of_mass(m)
    cx_map=xs[int(round(cx))]; cy_map=ys[int(round(cy))]
    lon,lat=tfm_inv.transform(cx_map,cy_map)
    elev_m=float(np.nanmean(elev[m])); anom_m=float(np.nanmean(anomaly[m]))
    bear_m=float(np.nanmean(bearing[m])); dist_m=float(np.nanmean(dist[m]))
    pct_ice=100*(m&gm).sum()/n; pct_hot=100*(m&hot).sum()/n; pct_cold=100*(m&cold).sum()/n
    is_apakhonchich=100*(m&flank).sum()/n>50
    cluster_rows.append(dict(cluster_id=cid,n_px=n,area_ha=round(n*900/1e4,1),lon=round(lon,4),lat=round(lat,4),
        elev_m=round(elev_m,0),bearing_from_summit=round(bear_m,0),dist_from_summit_m=round(dist_m,0),
        mean_anomaly_C=round(anom_m,1),pct_on_ice=round(pct_ice,1),pct_in_hotspot=round(pct_hot,1),
        pct_in_coldspot=round(pct_cold,1),is_apakhonchich_zone=is_apakhonchich))
cluster_rows.sort(key=lambda r:-r["n_px"])
print(f"\n{len(cluster_rows)} clusters >= {ANOM_CLUSTER_MIN_PX}px, sorted by size:")
for r in cluster_rows[:20]:
    tag=" [Apakhonchich]" if r["is_apakhonchich_zone"] else ""
    print(f"  #{r['cluster_id']:3d} n={r['n_px']:5d} ({r['area_ha']:6.1f}ha) elev={r['elev_m']:.0f}m "
          f"bearing={r['bearing_from_summit']:.0f}deg dist={r['dist_from_summit_m']/1000:.1f}km "
          f"anomaly=+{r['mean_anomaly_C']:.1f}C on_ice={r['pct_on_ice']:.0f}% "
          f"in_hotspot={r['pct_in_hotspot']:.0f}% in_coldspot={r['pct_in_coldspot']:.0f}%{tag}")

n_on_ice=sum(1 for r in cluster_rows if r["pct_on_ice"]>0)
n_in_hotspot=sum(1 for r in cluster_rows if r["pct_in_hotspot"]>0)
print(f"\n{n_on_ice}/{len(cluster_rows)} clusters touch glacier ice at all; {n_in_hotspot} touch the melt hotspot")

# ---- pixel-wise partial correlation: anomaly vs thinning rate, controlling for elevation ----
def partial_corr(y,x,control):
    A=np.c_[control,np.ones_like(control)]
    by=np.linalg.lstsq(A,y,rcond=None)[0]; ry=y-A@by
    bx=np.linalg.lstsq(A,x,rcond=None)[0]; rx=x-A@bx
    return stats.pearsonr(rx,ry)

corr_px=gm&valid_t&np.isfinite(rate)&np.isfinite(anomaly)
print(f"\npixel-wise anomaly-vs-rate correlation, on-ice px with both valid: {corr_px.sum():,}")
if corr_px.sum()>50:
    rr,zz,aa=rate[corr_px],elev[corr_px],anomaly[corr_px]
    raw=stats.pearsonr(aa,rr); part=partial_corr(rr,aa,zz)
    print(f"  raw r={raw[0]:+.3f} (p={raw[1]:.1e})  |  partial|elevation r={part[0]:+.3f} (p={part[1]:.1e})")
    print(f"  {'thermal anomaly predicts faster thinning even controlling for elevation' if part[1]<0.01 and part[0]<-0.05 else 'no meaningful glacier-wide relationship between thermal anomaly and thinning rate'}")
else:
    raw=(np.nan,np.nan); part=(np.nan,np.nan)
    print("  too few on-ice pixels with both valid rate and valid thermal anomaly -- skipped")

with open(AOUT/"ash_analysis/GEOTHERMAL_footprint_scan_stats.csv","w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["metric","value"])
    w.writerow(["anomaly_threshold_C",round(float(thr),2)])
    w.writerow(["n_clusters",len(cluster_rows)])
    w.writerow(["n_clusters_on_ice",n_on_ice])
    w.writerow(["n_clusters_in_hotspot",n_in_hotspot])
    w.writerow(["anomaly_vs_rate_raw_r",round(float(raw[0]),3) if np.isfinite(raw[0]) else ""])
    w.writerow(["anomaly_vs_rate_raw_p",raw[1] if np.isfinite(raw[1]) else ""])
    w.writerow(["anomaly_vs_rate_partial_r",round(float(part[0]),3) if np.isfinite(part[0]) else ""])
    w.writerow(["anomaly_vs_rate_partial_p",part[1] if np.isfinite(part[1]) else ""])
    w.writerow([])
    if cluster_rows:
        w.writerow(list(cluster_rows[0].keys()))
        for r in cluster_rows: w.writerow(list(r.values()))
print("-> GEOTHERMAL_footprint_scan_stats.csv")

# ---- figure: anomaly map with clusters + hotspot/coldspot overlay, rate map alongside ----
left,bottom,right,top=rasterio.transform.array_bounds(rows,cols,tr)
ext=[left,right,bottom,top]
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(18,9))

gm_show=np.where(gm,1,np.nan)
ax1.imshow(gm_show,cmap=ListedColormap(["#e8e4d8"]),extent=ext,zorder=1)
am=np.where(valid_t,anomaly,np.nan)
im1=ax1.imshow(am,cmap="RdBu_r",vmin=-15,vmax=15,extent=ext,zorder=2)
outline=np.zeros((rows,cols),bool)
for r in cluster_rows: outline|=(lbl==r["cluster_id"])
if outline.any():
    ax1.contour(outline.astype(float),levels=[0.5],colors="black",linewidths=1.0,extent=ext,origin="upper")
shp.boundary.plot(ax=ax1,color='k',linewidth=0.6,zorder=3)
ax1.plot(x0,y0,marker="^",color="lime",markersize=10,zorder=4,label="summit")
ax1.set_xlim(left,right); ax1.set_ylim(bottom,top); ax1.legend(loc="lower left")
plt.colorbar(im1,ax=ax1,label="elevation-band-relative winter thermal anomaly [C]",fraction=0.04)
ax1.set_title(f"unbiased scan: {len(cluster_rows)} warm clusters (outlined) >=P{ANOM_PERCENTILE:.0f}\n"
              f"{n_on_ice} touch ice, {n_in_hotspot} touch the melt hotspot")

rate_show=np.where(gm,rate,np.nan)
im2=ax2.imshow(rate_show,cmap="RdBu",vmin=-3,vmax=3,extent=ext,zorder=1)
hot_show=np.where(hot,1,np.nan)
ax2.imshow(hot_show,cmap=ListedColormap(["#000000"]),extent=ext,zorder=2)
shp.boundary.plot(ax=ax2,color='k',linewidth=0.6,zorder=3)
if outline.any():
    ax2.contour(outline.astype(float),levels=[0.5],colors="lime",linewidths=1.2,extent=ext,origin="upper")
ax2.set_xlim(left,right); ax2.set_ylim(bottom,top)
plt.colorbar(im2,ax=ax2,label="summer trend rate [m/yr]",fraction=0.04)
ax2.set_title("thinning rate + hotspot (black) with warm-cluster outlines (lime)\nfor spatial cross-reference")

fig.tight_layout(); fig.savefig(AOUT/"ash_analysis/GEOTHERMAL_footprint_scan_map.png",dpi=600,bbox_inches="tight")
print("-> GEOTHERMAL_footprint_scan_map.png")
