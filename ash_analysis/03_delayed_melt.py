#!/usr/bin/env python3
"""
STEP 3: Delayed-melt test for the 2023 PAROXYSM (best-sampled case).
Hypothesis: areas covered by ash after the eruption first GAIN height (deposition),
then LOSE height faster than clean glacier in following seasons (dark material
absorbs solar radiation -> enhanced melt).

Uses the ash mask from step 2 (2023 paroxysm) and tracks elevation change of
ash-covered vs clean glacier pixels across the DENSE post-eruption DEM series
(2024-01 ... 2025-03). Produces a time series comparison.
"""
import glob, re
import numpy as np, rasterio, rasterio.features, geopandas as gpd, xdem
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import date2num
from pathlib import Path
from datetime import date
from rasterio.warp import reproject, Resampling

BASE=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation/ash_analysis")
GLINV=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b/"
           "2025/2025-01-14_065954_TDT_SM_A_97516_155_0045+1-2/prc05_refdem/"
           "GLINV_rgi06_ttl_utm57_N_b_UTM.shp")
crs="EPSG:32657"; RES=30.0
ASH_MASK=AOUT/"ASH_mask_2023_Nov_PAROXYSM.tif"

# post-paroxysm ascending DEMs (2024-01 onward), + a pre-eruption reference
def find(sd):
    for t in ("DEM_FNL","DEM_VER"):
        g=glob.glob(str(sd/"prc07"/f"{t}_*.tif"))
        if g: return g[0]
dems=[]
for sd in sorted(BASE.glob("202*/20*")):
    nm=sd.name; m=re.match(r"(\d{4})-(\d{2})-(\d{2})",nm)
    if not m or "155_0045" not in nm: continue
    d=date(*map(int,m.groups()))
    if d<date(2023,1,1) or d>date(2025,12,31): continue
    f=find(sd)
    if f: dems.append((d,f))
dems=sorted(set(dems))
print(f"DEMs 2023-2025: {[d.isoformat() for d,_ in dems]}")

# grid from a full 2025 scene
ref_f=[f for d,f in dems if d.year==2025][0]
with rasterio.open(ref_f) as s:
    tr=s.transform; rows,cols=s.shape; xb=s.bounds
def rg(f):
    with rasterio.open(f) as s:
        a=s.read(1).astype("float32"); a[(a==s.nodata)|(a<0)|(a>5000)]=np.nan
        d=np.full((rows,cols),np.nan,"float32")
        reproject(a,d,src_transform=s.transform,src_crs=s.crs,dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear)
    d[(d<0)|(d>5000)]=np.nan; return d
shp=gpd.read_file(GLINV).to_crs(crs)
gm=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
    out_shape=(rows,cols),transform=tr,fill=0,dtype="uint8").astype(bool)
st=~gm
# ash mask on grid
with rasterio.open(ASH_MASK) as s:
    am=s.read(1); ash=np.full((rows,cols),0,"float32")
    reproject(am.astype("float32"),ash,src_transform=s.transform,src_crs=s.crs,dst_transform=tr,dst_crs=crs,resampling=Resampling.nearest)
ash=ash>0.5
print(f"ash pixels on grid: {ash.sum():,}")

# reference = pre-eruption DEM (2023-02-10)
pre=[f for d,f in dems if d.year==2023]
ref=rg(pre[0]) if pre else rg(dems[-1][1])
refD=xdem.DEM.from_array(ref,transform=tr,crs=crs,nodata=np.nan)
ash_ts=[]; gl_ts=[]
for d,f in dems:
    dem=rg(f)
    try:
        tD=xdem.DEM.from_array(dem,transform=tr,crs=crs,nodata=np.nan)
        nk=xdem.coreg.NuthKaab(); nk.fit(refD,tD,inlier_mask=st,random_state=42)
        dem=nk.apply(tD); dem=dem.data.filled(np.nan) if hasattr(dem.data,'filled') else np.asarray(dem.data,float)
    except Exception: pass
    dh=dem-ref; dh=dh-np.nanmedian(dh[st&np.isfinite(dh)])
    a=np.nanmedian(dh[ash&np.isfinite(dh)]) if (ash&np.isfinite(dh)).sum()>20 else np.nan
    g=np.nanmedian(dh[gm&~ash&np.isfinite(dh)]) if (gm&~ash&np.isfinite(dh)).sum()>20 else np.nan
    ash_ts.append((d,a)); gl_ts.append((d,g))
    print(f"  {d}: ash dH={a:+.2f}m  clean dH={g:+.2f}m")

fig,ax=plt.subplots(figsize=(13,6))
ax.axvspan(date2num(date(2023,6,22)),date2num(date(2024,1,15)),color='firebrick',alpha=0.25,label='2023-24 eruption')
for ts,lab,col in [(ash_ts,'ash/lava-covered glacier','saddlebrown'),(gl_ts,'clean glacier','#1f77b4')]:
    xs=[date2num(d) for d,v in ts if np.isfinite(v)]; ys=[v for d,v in ts if np.isfinite(v)]
    ax.plot(xs,ys,'o-',color=col,ms=7,lw=1.6,label=lab)
ax.axhline(0,color='k',lw=0.6,ls=':')
ax.set_ylabel("elevation change vs pre–eruption [m]")
ax.set_title("2023–24 paroxysm: elevation change of ash–covered vs clean glacier\n"
             "(deposition = initial gain; albedo–melt = later enhanced loss)")
ax.legend(); ax.grid(alpha=0.3)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate()
fig.savefig(AOUT/"DELAYED_MELT_2023paroxysm.png",dpi=600,bbox_inches="tight")
print("\n-> DELAYED_MELT_2023paroxysm.png")
