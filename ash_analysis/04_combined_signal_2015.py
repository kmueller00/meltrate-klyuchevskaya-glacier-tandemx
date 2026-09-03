#!/usr/bin/env python3
"""
STEP 4: Combined glacier elevation-change signal 2015-onward (Sentinel-2 era),
using BOTH winter and summer DEMs, with the full KVERT/GVP eruption record marked.
Shows the overall picture: does glacier surface elevation track eruptive periods?

Per DEM: coregister to reference on stable terrain, glacier-median elevation
anomaly vs reference. Winter and summer plotted with different markers.
Auto-includes new summer DEMs (2023/24/25) once they process.
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
AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation")
GLINV=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b/"
           "2025/2025-01-14_065954_TDT_SM_A_97516_155_0045+1-2/prc05_refdem/"
           "GLINV_rgi06_ttl_utm57_N_b_UTM.shp")
GLINV=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/ash_analysis/GLINV_main_massif.gpkg")  # OVERRIDE: main massif only (southern tongue-only glaciers excluded)
crs="EPSG:32657"; RES=30.0

# ── full eruption record 2015+ (KVERT/GVP) ────────────────────────────────────
ERUPT=[(date(2015,1,2),date(2015,3,24),"2015 Strombolian",0),
 (date(2015,8,27),date(2015,9,10),"2015 Aug",0),
 (date(2016,4,1),date(2016,5,1),"2016 Apr explosive",0),
 (date(2019,4,1),date(2019,11,1),"2019 eruption",1),
 (date(2020,10,2),date(2021,3,25),"2020-21 cycle",1),
 (date(2022,11,20),date(2022,12,20),"2022 Nov",0),
 (date(2023,6,22),date(2024,1,15),"2023-24 PAROXYSM",1),
 (date(2024,8,1),date(2024,10,15),"2024 Aug-Oct",0)]

def find(sd):
    for t in ("DEM_FNL","DEM_VER"):
        g=glob.glob(str(sd/"prc07"/f"{t}_*.tif"))
        if g: return g[0]

# ascending 155 DEMs 2015+ (single track = clean)
dems=[]
for sd in [Path(p) for p in glob.glob(str(BASE/"20*/20*"))]:
    nm=Path(sd).name; m=re.match(r"(\d{4})-(\d{2})-(\d{2})",nm)
    if not m or "155_0045" not in nm: continue
    d=date(*map(int,m.groups()))
    if d<date(2015,1,1): continue
    f=find(sd)
    if not f: continue
    # quality gate: check median elevation is physical
    with rasterio.open(f) as s:
        a=s.read(1).astype(float); a[(a==s.nodata)|(a<0)|(a>5000)]=np.nan
        v=a[np.isfinite(a)]
    if v.size<3e5 or not (800<np.median(v)<2500): continue
    dems.append((d,f))
dems=sorted(set(dems))
print(f"DEMs 2015+: {len(dems)}")

# grid from a full 2025 scene
ref_f=[f for d,f in dems if d.year==2025 and (d.month in(1,2,3))][0]
with rasterio.open(ref_f) as s: tr=s.transform; rows,cols=s.shape
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
ref=rg(ref_f); refD=xdem.DEM.from_array(ref,transform=tr,crs=crs,nodata=np.nan)

series=[]
for d,f in dems:
    dem=rg(f)
    try:
        tD=xdem.DEM.from_array(dem,transform=tr,crs=crs,nodata=np.nan)
        nk=xdem.coreg.NuthKaab(); nk.fit(refD,tD,inlier_mask=st,random_state=42)
        dem=nk.apply(tD); dem=dem.data.filled(np.nan) if hasattr(dem.data,'filled') else np.asarray(dem.data,float)
    except Exception: pass
    dh=dem-ref; dh=np.where(np.abs(dh)<=50,dh,np.nan)   # hard clip
    b=np.nanmedian(dh[st&np.isfinite(dh)]); dh=dh-(b if np.isfinite(b) else 0)
    gv=dh[gm&np.isfinite(dh)]
    if gv.size<500: continue
    # QUALITY GATE: reject corrupt DEMs / bad coreg via stable-terrain NMAD
    stv=dh[st&np.isfinite(dh)]
    stab_nmad=1.4826*np.nanmedian(np.abs(stv-np.nanmedian(stv))) if stv.size>500 else 999
    anom=np.nanmedian(gv)
    if stab_nmad>6.0 or abs(anom)>15:      # noisy stable terrain or implausible anomaly
        print(f"  {d}: REJECTED (stable NMAD={stab_nmad:.1f}m, anom={anom:+.1f}m)"); continue
    seas="W" if (d.month>=11 or d.month<=4) else "S"
    series.append((d,anom,seas))

# MERGE sub-scenes: one value per (date) = median of that date's scenes
from collections import defaultdict
bydate=defaultdict(list)
for d,a,se in series: bydate[d].append((a,se))
series=[(d, float(np.median([a for a,_ in v])), v[0][1]) for d,v in bydate.items()]
series=sorted(series)
print(f"\nmerged to {len(series)} date-points (sub-scenes combined)")
for d,a,se in series: print(f"  {d} [{se}]: {a:+.2f} m")

fig,ax=plt.subplots(figsize=(15,6))
for s,e,lab,major in ERUPT:
    ax.axvspan(date2num(s),date2num(e),color=('firebrick' if major else 'orange'),
               alpha=0.25 if major else 0.14,zorder=0)
    ax.text(date2num(s),0.98,lab,rotation=90,va='top',ha='right',fontsize=7.5,
            color=('firebrick' if major else 'darkorange'),transform=ax.get_xaxis_transform())
W=sorted([(date2num(d),a) for d,a,se in series if se=="W"])
S=sorted([(date2num(d),a) for d,a,se in series if se=="S"])
# SEPARATE season trend lines (rolling median, window=5) -- do NOT connect across
# seasons: winter and summer sit on different data due to X-band snow penetration
# (winter phase-centre is BELOW the surface -> winter reads lower; bias ~ measured
#  median -0.4 m here, but year-to-year variable, so seasons are not intercomparable).
def rollmed(pts,w=5):
    if len(pts)<3: return None
    x,y=zip(*pts); x=np.array(x); y=np.array(y); out=[]
    for i in range(len(x)):
        lo=max(0,i-w//2); hi=min(len(x),i+w//2+1); out.append(np.median(y[lo:hi]))
    return x,np.array(out)
if W: ax.scatter(*zip(*W),s=62,c='#1f77b4',marker='o',edgecolor='k',lw=0.4,zorder=3,label='winter DEM (radar-penetration biased low)')
if S: ax.scatter(*zip(*S),s=70,c='#e0662a',marker='^',edgecolor='k',lw=0.4,zorder=3,label='summer DEM (surface)')
for pts,col in [(S,'#e0662a'),(W,'#1f77b4')]:
    rm=rollmed(pts)
    if rm is not None: ax.plot(rm[0],rm[1],'-',color=col,lw=2,alpha=0.9,zorder=2)
ax.axhline(0,color='k',lw=0.6,ls=':')
ax.set_ylabel("glacier–median elevation anomaly [m]  (vs 2025 ref)")
ax.set_title("Klyuchevskaya glacier surface elevation 2015–2025 (ascending 155, Sentinel–2 era)\n"
             "summer = true surface; winter biased low by X–band snow penetration — compare within–season only")
ax.legend(loc='lower left',fontsize=9); ax.grid(alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_xlim(date2num(date(2015,1,1)),date2num(date(2026,3,1)))
fig.tight_layout(); fig.savefig(AOUT/"COMBINED_signal_2015-2025.png",dpi=600,bbox_inches="tight")
print("\n-> COMBINED_signal_2015-2025.png")
