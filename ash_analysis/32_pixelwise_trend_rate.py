#!/usr/bin/env python3
import sys; sys.path.insert(0,"/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/ash_analysis"); import _style
"""
STEP 32: Per-pixel LINEAR TREND rate (replaces the rigid early/late two-bucket
median-difference in dH_robust_all.py's summer rate for spatial analyses).

Motivation: the coverage diagnostic (2026-08-04, scratch coverage_diag.py) showed
the "35% of glacier lacks rate data" gap in summer_robust_rate_glacieronly.tif is
NOT a satellite-footprint dead zone. Only 2.1% of glacier pixels genuinely have
zero coverage in EITHER period (early<2018 / late>=2018). The other 23% (mostly
the northern branches) have full POST-2018 coverage but zero PRE-2018 coverage on
this track, and get dropped only because the two-bucket median-difference
estimator requires both an early AND a late value at every pixel.

Fix: coregister every summer (May-Oct) 155_0045 acquisition day individually onto
the single fixed 2024-09-04 full-footprint reference (same reference used
throughout this session -- FULL_FOOTPRINT_REF), on a grid cropped to the glacier
+1km buffer (not the full scene extent, for tractable memory/runtime), then fit a
robust per-pixel linear trend (2-pass outlier-clipped least squares, vectorized
over the whole stack -- no per-pixel python loop) using whatever dates are
actually available at each pixel. Full-record pixels keep ~13yr baseline;
north-branch pixels get a shorter (~7yr, 2018-2025-only) baseline instead of
being dropped entirely.

Outputs (in RESULTS_presentation/):
  summer_trend_rate_glacieronly.tif   -- per-pixel linear-trend rate [m/yr]
  summer_trend_baseline_years.tif     -- actual date-range span used per pixel
  summer_trend_ndates.tif             -- number of inlier dates used per pixel
  TREND_vs_TWOBUCKET_compare.png      -- sanity check against the old estimator
"""
import numpy as np, rasterio, rasterio.features, geopandas as gpd, xdem
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date
import glob, re, warnings
from rasterio.warp import reproject, Resampling
warnings.filterwarnings("ignore")

BASE=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
OUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out")
AOUT=OUT/"RESULTS_presentation"
GLINV_MAIN=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/ash_analysis/GLINV_main_massif.gpkg")
crs="EPSG:32657"; RES=30.0
FULL_FOOTPRINT_REF=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b/"
    "2024/2024-09-04_065957_TDT_SM_A_95512_155_0045_HH./prc07/DEM_FNL_2024-09-04_065957_0-030m00_155_0045_HH..tif")
MIN_DATES=4   # require at least this many inlier dates to report a trend

def find_dem(sd):
    for t in ("DEM_FNL","DEM_VER"):
        g=glob.glob(str(sd/"prc07"/f"{t}_*.tif"))
        if g: return g[0]

def best_scene(files):
    best=None; best_n=-1
    for f in files:
        with rasterio.open(f) as s:
            a=s.read(1); n=int((a!=s.nodata).sum())
        if n>best_n: best_n=n; best=f
    return best

def is_summer(mo): return 5<=mo<=10

byday={}
for yd in sorted(BASE.glob("20*")):
    for sd in sorted(yd.glob("20*")):
        nm=sd.name; m=re.match(r"(\d{4})-(\d{2})-(\d{2})",nm)
        if not m or "155_0045" not in nm: continue
        y,mo,da=map(int,m.groups())
        if not is_summer(mo): continue
        f=find_dem(sd)
        if not f: continue
        byday.setdefault(m.group(0),[]).append(f)
days=sorted(byday)
import os
_lim=os.environ.get("TREND_DEBUG_LIMIT")
if _lim: days=days[:int(_lim)]
print(f"{len(days)} distinct summer acquisition days on 155_0045" + (f" (DEBUG-limited)" if _lim else ""))

shp=gpd.read_file(GLINV_MAIN).to_crs(crs)
b=shp.total_bounds; buf=1000.0
xmin=np.floor((b[0]-buf)/RES)*RES; xmax=np.ceil((b[2]+buf)/RES)*RES
ymin=np.floor((b[1]-buf)/RES)*RES; ymax=np.ceil((b[3]+buf)/RES)*RES
cols=int((xmax-xmin)/RES); rows=int((ymax-ymin)/RES)
tr=rasterio.transform.from_origin(xmin,ymax,RES,RES)
print(f"cropped grid {rows}x{cols} (glacier+1km buffer)  bounds {xmin,xmax,ymin,ymax}")

gm=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
    out_shape=(rows,cols),transform=tr,fill=0,dtype='uint8').astype(bool)
st=~gm

def load_on(f):
    with rasterio.open(f) as s:
        a=s.read(1).astype("float32"); a[(a==s.nodata)|(a<0)|(a>5000)]=np.nan
        o=np.full((rows,cols),np.nan,"float32")
        reproject(a,o,src_transform=s.transform,src_crs=s.crs,dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear)
    o[(o<0)|(o>5000)]=np.nan; return o

ref_f=best_scene(byday["2024-09-04"])
Zref=load_on(ref_f)
rB=xdem.DEM.from_array(Zref,transform=tr,crs=crs,nodata=np.nan)
print(f"reference: {Path(ref_f).stem[:40]}")

# hard reject on implausible coregistration shift -- FULL_COREG_scan.csv (2026-08-04)
# showed a median shift of (-1.50,-0.10)m with MAD-radius 3.22m across the full
# archive, and one clearly failed fit (2012-08-29, shift ~104m, ~33x the MAD
# radius). A per-pixel 3-sigma clip alone can't reliably reject a bad scene when
# few dates are available at a given pixel, so reject at the scene level too.
MAX_SHIFT_M=25.0
t0=date(2024,9,4).toordinal()
tvals=[]; stack=[]; n_ok=0; n_fail=0; n_rejected=0
for i,day in enumerate(days):
    f=best_scene(byday[day])
    try:
        A=load_on(f)
        if f==ref_f:
            Aa=A
        else:
            rA=xdem.DEM.from_array(A,transform=tr,crs=crs,nodata=np.nan)
            nk=xdem.coreg.NuthKaab(); nk.fit(rB,rA,inlier_mask=st,random_state=42)
            dx,dy,dz=nk.to_translations()
            if not (np.isfinite(dx) and np.isfinite(dy) and np.hypot(dx,dy)<=MAX_SHIFT_M):
                print(f"  REJECTED {day}: implausible shift dx={dx:.1f} dy={dy:.1f}")
                n_rejected+=1; continue
            Aal=nk.apply(rA); Aa=Aal.data.filled(np.nan) if hasattr(Aal.data,'filled') else np.asarray(Aal.data,float)
        bias=np.nanmedian((Aa-Zref)[st&np.isfinite(Aa)&np.isfinite(Zref)])
        if not np.isfinite(bias): bias=0.0
        Aa=(Aa-bias).astype("float32")
        d=date.fromisoformat(day)
        tvals.append((d.toordinal()-t0)/365.25); stack.append(Aa); n_ok+=1
    except Exception as e:
        n_fail+=1
        print(f"  FAILED {day}: {e}")
    if (i+1)%20==0: print(f"  {i+1}/{len(days)} processed (ok={n_ok} fail={n_fail} rejected={n_rejected})")

print(f"coregistered {n_ok}/{len(days)} summer days onto fixed 2024-09-04 reference ({n_rejected} rejected, {n_fail} failed)")
Z=np.stack(stack,axis=0); T=np.array(tvals,dtype="float32")
del stack

def fit_slope(Z,T,valid):
    Tb=np.broadcast_to(T[:,None,None],Z.shape)
    n=valid.sum(axis=0)
    tsum=np.where(valid,Tb,0).sum(axis=0)
    zsum=np.where(valid,np.nan_to_num(Z),0).sum(axis=0)
    with np.errstate(invalid="ignore",divide="ignore"):
        tmean=np.where(n>0,tsum/np.maximum(n,1),np.nan)
        zmean=np.where(n>0,zsum/np.maximum(n,1),np.nan)
    dt=np.where(valid,Tb-tmean[None],0.0)
    dz=np.where(valid,np.nan_to_num(Z)-np.nan_to_num(zmean[None]),0.0)
    Stt=(dt*dt*valid).sum(axis=0); Stz=(dt*dz*valid).sum(axis=0)
    with np.errstate(invalid="ignore",divide="ignore"):
        slope=np.where(Stt>0,Stz/np.where(Stt>0,Stt,1),np.nan)
    pred=slope[None]*Tb+zmean[None]
    resid=np.where(valid,Z-pred,np.nan)
    return slope.astype("float32"),zmean.astype("float32"),resid

valid=np.isfinite(Z)
slope1,zmean1,resid1=fit_slope(Z,T,valid)
mad=1.4826*np.nanmedian(np.abs(resid1),axis=0)
mad=np.where(np.isfinite(mad)&(mad>0.3),mad,0.3)
inlier=valid & (np.abs(resid1)<=3*mad[None])
slope2,zmean2,resid2=fit_slope(Z,T,inlier)
n2=inlier.sum(axis=0)

Tb=np.broadcast_to(T[:,None,None],Z.shape)
Tmasked=np.where(inlier,Tb,np.nan)
tmax=np.nanmax(Tmasked,axis=0); tmin=np.nanmin(Tmasked,axis=0)
baseline_yr=(tmax-tmin).astype("float32")

rate=np.where(n2>=MIN_DATES,slope2,np.nan).astype("float32")
print(f"pixels with >={MIN_DATES} inlier dates (rate reported): {np.isfinite(rate).sum():,}")
print(f"  of those, within glacier mask: {(gm&np.isfinite(rate)).sum():,} / {gm.sum():,} ({100*(gm&np.isfinite(rate)).sum()/gm.sum():.1f}%)")

prof=dict(driver="GTiff",height=rows,width=cols,count=1,dtype="float32",crs=crs,
          transform=tr,nodata=-9999.0,compress="lzw")
for arr,nm in [(np.where(gm,rate,np.nan),"summer_trend_rate_glacieronly.tif"),
               (np.where(gm,baseline_yr,np.nan),"summer_trend_baseline_years.tif"),
               (np.where(gm,n2.astype("float32"),np.nan),"summer_trend_ndates.tif"),
               (np.where(gm,zmean2,np.nan),"summer_trend_elevation_mean.tif")]:
    with rasterio.open(AOUT/nm,"w",**prof) as d:
        d.write(np.where(np.isfinite(arr),arr,-9999.0).astype("float32"),1)
    print(f"-> {nm}")

gg=rate[gm&np.isfinite(rate)]
print(f"glacier-wide trend-rate median={np.median(gg):+.3f} m/yr  NMAD={1.4826*np.median(np.abs(gg-np.median(gg))):.3f}  n={gg.size:,}")

# ---- sanity check: compare against the old two-bucket estimator where both exist ----
OLD=AOUT/"summer_robust_rate_glacieronly.tif"
try:
    with rasterio.open(OLD) as s:
        old=s.read(1).astype("float32"); old[old==s.nodata]=np.nan
        oo=np.full((rows,cols),np.nan,"float32")
        reproject(old,oo,src_transform=s.transform,src_crs=s.crs,dst_transform=tr,dst_crs=crs,resampling=Resampling.nearest)
    both=gm&np.isfinite(rate)&np.isfinite(oo)
    print(f"overlap px with both estimators: {both.sum():,}; median diff (trend-old)={np.nanmedian(rate[both]-oo[both]):+.3f} m/yr")
    fig,ax=plt.subplots(figsize=(6,6))
    ax.hexbin(oo[both],rate[both],gridsize=80,cmap="viridis",mincnt=1)
    lim=2.5; ax.plot([-lim,lim],[-lim,lim],'r--',lw=1)
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
    ax.set_xlabel("old two-bucket rate [m/yr]"); ax.set_ylabel("new per-pixel trend rate [m/yr]")
    ax.set_title("Trend-fit vs two-bucket estimator, pixels where both exist")
    fig.tight_layout(); fig.savefig(AOUT/"TREND_vs_TWOBUCKET_compare.png",dpi=150,bbox_inches="tight")
    print("-> TREND_vs_TWOBUCKET_compare.png")
except Exception as e:
    print(f"compare skipped: {e}")

# ---- quick preview map (rate + baseline-years) ----
left,bottom,right,top=rasterio.transform.array_bounds(rows,cols,tr)
ext=[left,right,bottom,top]
fig,axes=plt.subplots(1,2,figsize=(14,7))
im0=axes[0].imshow(np.where(gm,rate,np.nan),cmap="RdBu",vmin=-2,vmax=2,extent=ext)
plt.colorbar(im0,ax=axes[0],shrink=0.7,label="trend rate [m/yr]")
shp.boundary.plot(ax=axes[0],color='k',linewidth=0.5)
axes[0].set_title(f"per-pixel trend rate ({(gm&np.isfinite(rate)).sum()/gm.sum()*100:.1f}% glacier coverage)")
im1=axes[1].imshow(np.where(gm,baseline_yr,np.nan),cmap="viridis",vmin=0,vmax=13,extent=ext)
plt.colorbar(im1,ax=axes[1],shrink=0.7,label="baseline span [yr]")
shp.boundary.plot(ax=axes[1],color='k',linewidth=0.5)
axes[1].set_title("actual date-range span used per pixel")
for ax in axes: ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig(AOUT/"PREVIEW_trend_rate.png",dpi=150,bbox_inches="tight")
print("-> PREVIEW_trend_rate.png")
