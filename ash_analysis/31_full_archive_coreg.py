#!/usr/bin/env python3
import sys; sys.path.insert(0,"/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/ash_analysis"); import _style
"""
STEP 31: Full-archive coregistration stability check.

Step 26 diagnosed the flagship annual pipeline's 13 consecutive-year pairs
(one best scene per year, summer only) and found -- after the best_scene()
fix -- zero outliers. That answers "is the annual pipeline's own series
clean", but it is a narrow sample: 13 points can't reveal within-year
jitter, a seasonal (summer vs winter) bias in the horizontal fit, or a slow
drift across the mission that a once-a-year sample would smear out.

This script instead coregisters EVERY distinct acquisition day on track
155_0045 (all seasons, not just summer) against a single fixed reference
grid, using best_scene() per day to avoid the known secondary-subframe
scenes (same dedup logic as the archive footprint scan). One (dx,dy,dz)
point per day, not per year.
Outputs: FULL_COREG_scan.csv, FULL_COREG_scatter.png (dx vs dy, coloured by
season), FULL_COREG_timeseries.png (dx/dy/dz vs date).
"""
import glob, re, numpy as np, rasterio, rasterio.features, geopandas as gpd, xdem
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import date
from collections import defaultdict
from rasterio.warp import reproject, Resampling
import csv, warnings
warnings.filterwarnings("ignore")

BASE=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation")
GLINV=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/ash_analysis/GLINV_main_massif.gpkg")
crs="EPSG:32657"; RES=30.0

def find(sd):
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

# ---- one best scene per distinct acquisition day (all seasons) ----
byday=defaultdict(list)
for sd in sorted(BASE.glob("20*/20*")):
    nm=sd.name
    if "155_0045" not in nm: continue
    m=re.match(r"(\d{4})-(\d{2})-(\d{2})",nm)
    if not m: continue
    f=find(sd)
    if not f: continue
    byday[m.group(0)].append(f)
days=sorted(byday)
print(f"{len(days)} distinct acquisition days on 155_0045")

# reference grid: the same full-footprint 2024-09-04 scene used elsewhere this session
ref_f=None
for f in byday.get("2024-09-04",[]):
    ref_f=best_scene(byday["2024-09-04"]); break
if ref_f is None:
    ref_f=best_scene(byday[days[len(days)//2]])
with rasterio.open(ref_f) as s: tr=s.transform; rows,cols=s.shape
shp=gpd.read_file(GLINV).to_crs(crs)
gm=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
    out_shape=(rows,cols),transform=tr,fill=0,dtype='uint8').astype(bool)
st=~gm
print(f"reference: {Path(ref_f).stem[:20]}")

def load_on(f):
    with rasterio.open(f) as s:
        a=s.read(1).astype("float32"); a[(a==s.nodata)|(a<0)|(a>5000)]=np.nan
        o=np.full((rows,cols),np.nan,"float32")
        reproject(a,o,src_transform=s.transform,src_crs=s.crs,dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear)
    o[(o<0)|(o>5000)]=np.nan; return o

rB=xdem.DEM.from_array(load_on(ref_f),transform=tr,crs=crs,nodata=np.nan)
rows_out=[]
for i,day in enumerate(days):
    f=best_scene(byday[day])
    if f==ref_f:
        rows_out.append(dict(date=day,dx=0.0,dy=0.0,dz=0.0,month=int(day[5:7]))); continue
    try:
        A=load_on(f)
        rA=xdem.DEM.from_array(A,transform=tr,crs=crs,nodata=np.nan)
        nk=xdem.coreg.NuthKaab(); nk.fit(rA,rB,inlier_mask=st,random_state=42)
        dx,dy,dz=nk.to_translations()
    except Exception as e:
        dx,dy,dz=np.nan,np.nan,np.nan
    rows_out.append(dict(date=day,dx=dx,dy=dy,dz=dz,month=int(day[5:7])))
    if (i+1)%20==0: print(f"  {i+1}/{len(days)} done ({day}: dx={dx:+.2f} dy={dy:+.2f})")

with open(AOUT/"FULL_COREG_scan.csv","w",newline="") as fcsv:
    w=csv.DictWriter(fcsv,fieldnames=["date","dx","dy","dz","month"],lineterminator="\n"); w.writeheader()
    for r in rows_out: w.writerow(r)
print("-> FULL_COREG_scan.csv")

ok=[r for r in rows_out if np.isfinite(r["dx"])]
print(f"{len(ok)}/{len(rows_out)} scenes fit successfully")
dxs=np.array([r["dx"] for r in ok]); dys=np.array([r["dy"] for r in ok])
months=np.array([r["month"] for r in ok])
is_summer=(months==8)|(months==9)
med_dx,med_dy=np.median(dxs),np.median(dys)
mad=1.4826*np.median(np.hypot(dxs-med_dx,dys-med_dy))
print(f"median offset ({med_dx:+.2f},{med_dy:+.2f}) m, MAD-radius={mad:.2f} m")
print(f"summer (n={is_summer.sum()}) median ({np.median(dxs[is_summer]):+.2f},{np.median(dys[is_summer]):+.2f}) m")
print(f"non-summer (n={(~is_summer).sum()}) median ({np.median(dxs[~is_summer]):+.2f},{np.median(dys[~is_summer]):+.2f}) m")

# ---- scatter, coloured by season ----
fig,ax=plt.subplots(figsize=(8,8))
ax.scatter(dxs[~is_summer],dys[~is_summer],s=30,color="#8899aa",alpha=0.7,label=f"other months (n={(~is_summer).sum()})",zorder=3)
ax.scatter(dxs[is_summer],dys[is_summer],s=40,color="#c0463a",alpha=0.85,label=f"Aug-Sep, ie annual-pipeline season (n={is_summer.sum()})",zorder=4)
ax.scatter([med_dx],[med_dy],marker='+',s=200,color='k',zorder=5)
circ=plt.Circle((med_dx,med_dy),2*mad,fill=False,ls='--',color='grey')
ax.add_patch(circ)
ax.set_xlabel("x shift (easting) [m]"); ax.set_ylabel("y shift (northing) [m]")
ax.set_title(f"Full-archive coregistration shift, all {len(ok)} acquisition days on 155_0045\n"
             f"vs the 13-pair annual diagnostic: does season or full sampling change the picture?")
ax.legend(loc='best',fontsize=9); ax.set_aspect('equal')
fig.tight_layout(); fig.savefig(AOUT/"FULL_COREG_scatter.png",dpi=600,bbox_inches="tight")
print("-> FULL_COREG_scatter.png")

# ---- time series ----
dts=[date.fromisoformat(r["date"]) for r in ok]
fig,axes=plt.subplots(3,1,figsize=(14,9),sharex=True)
for ax,key,lab in zip(axes,["dx","dy","dz"],["x shift [m]","y shift [m]","z shift [m]"]):
    vals=[r[key] for r in ok]
    cols=["#c0463a" if m in (8,9) else "#8899aa" for m in months]
    ax.scatter(dts,vals,c=cols,s=18)
    ax.axhline(0,color='k',lw=0.6,ls=':')
    ax.set_ylabel(lab)
axes[0].set_title("Coregistration shift vs time, every acquisition day on 155_0045\nred = Aug-Sep (annual-pipeline season), grey = other months")
axes[-1].xaxis.set_major_locator(mdates.YearLocator())
fig.tight_layout(); fig.savefig(AOUT/"FULL_COREG_timeseries.png",dpi=600,bbox_inches="tight")
print("-> FULL_COREG_timeseries.png")
