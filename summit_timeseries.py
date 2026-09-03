#!/usr/bin/env python3
"""
SUMMIT-ZONE elevation time series vs Klyuchevskaya eruptions.
Tests whether eruptive deposition leaves a detectable signal in glacier-surface
elevation at the summit/upper-cone (where lava/tephra accumulates).

Method per DEM:
  - read on a common grid, coregister (Nuth&Kaab) to a reference DEM on STABLE terrain
  - summit zone = glacier pixels above SUMMIT_ELEV (upper cone / crater area)
  - record median summit-zone elevation (relative to reference) + date
Plotted separately for ASCENDING (155) and DESCENDING (011) tracks, with
KVERT/GVP-verified eruption spans overlaid.
"""
import numpy as np, rasterio, rasterio.features, geopandas as gpd, xdem
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import date2num
from pathlib import Path
from datetime import date
import glob, re
from rasterio.warp import reproject, Resampling

BASE=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
OUT =Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out")
GLINV=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b/"
           "2025/2025-01-14_065954_TDT_SM_A_97516_155_0045+1-2/prc05_refdem/"
           "GLINV_rgi06_ttl_utm57_N_b_UTM.shp")
RES=30.0; crs="EPSG:32657"
SUMMIT_ELEV=3000.0    # m — glacier pixels above this = upper cone / summit zone
MIN_VALID=1.5e6

def find_dem(sd):
    for t in ("DEM_FNL","DEM_VER"):
        g=glob.glob(str(sd/"prc07"/f"{t}_*.tif"))
        if g: return g[0]

def collect(orbit):  # '155' (asc) or '011' (desc)
    out=[]
    for yd in sorted(BASE.glob("20*")):
        for sd in sorted(yd.glob("20*")):
            nm=sd.name; m=re.match(r"(\d{4})-(\d{2})-(\d{2})",nm)
            if not m: continue
            y,mo,da=map(int,m.groups())
            if not (mo>=11 or mo<=4): continue       # winter only (consistent snow)
            if f"{orbit}_0045" not in nm: continue
            f=find_dem(sd)
            if not f: continue
            with rasterio.open(f) as s:
                a=s.read(1); v=np.sum((a!=s.nodata)&np.isfinite(a)&(a>-1000)&(a<9000))
            if v>MIN_VALID: out.append((date(y,mo,da),f))
    # dedupe by date (slc-T/R can double)
    seen={}; [seen.setdefault(d,f) for d,f in out]
    return sorted(seen.items())

def grid(scenes):
    bs=[]
    for _,f in scenes:
        with rasterio.open(f) as s: bs.append(s.bounds)
    xmin=max(b.left for b in bs);xmax=min(b.right for b in bs)
    ymin=max(b.bottom for b in bs);ymax=min(b.top for b in bs)
    xmin=np.ceil(xmin/RES)*RES;xmax=np.floor(xmax/RES)*RES
    ymin=np.ceil(ymin/RES)*RES;ymax=np.floor(ymax/RES)*RES
    return (int((xmax-xmin)/RES),int((ymax-ymin)/RES),
            rasterio.transform.from_origin(xmin,ymax,RES,RES))

def rg(f,cols,rows,tr):
    with rasterio.open(f) as s:
        src=s.read(1).astype("float32"); src[(src==s.nodata)|(src<-1000)|(src>9000)]=np.nan
        dst=np.full((rows,cols),np.nan,"float32")
        reproject(src,dst,src_transform=s.transform,src_crs=s.crs,dst_transform=tr,
                  dst_crs=crs,resampling=Resampling.bilinear,src_nodata=np.nan,dst_nodata=np.nan)
    dst[(dst<-1000)|(dst>9000)]=np.nan
    return dst

# eruptions (KVERT/GVP verified) -- (start,end,label,major)
ERUPT=[(date(2010,2,27),date(2010,3,9),"2010",0),
 (date(2012,10,15),date(2012,11,30),"2012",0),
 (date(2013,1,25),date(2013,11,30),"2013 MAJOR",1),
 (date(2015,1,2),date(2015,8,27),"2015",0),
 (date(2019,10,25),date(2019,10,27),"2019",0),
 (date(2020,12,9),date(2020,12,20),"2020",0),
 (date(2022,11,20),date(2022,12,15),"2022",0),
 (date(2023,6,22),date(2024,1,15),"2023-24 MAJOR+parox",1),
 (date(2025,7,30),date(2025,9,1),"2025",0)]

def process(orbit,label):
    sc=collect(orbit)
    if len(sc)<2: print(f"  {label}: only {len(sc)} DEMs, skip"); return None
    cols,rows,tr=grid(sc)
    shp=gpd.read_file(GLINV).to_crs(crs)
    gm=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
        out_shape=(rows,cols),transform=tr,fill=0,dtype="uint8").astype(bool)
    st=~gm
    ref=rg(sc[-1][1],cols,rows,tr)   # reference = latest DEM (most complete)
    refD=xdem.DEM.from_array(ref,transform=tr,crs=crs,nodata=np.nan)
    res=[]
    for d,f in sc:
        dem=rg(f,cols,rows,tr)
        try:
            tD=xdem.DEM.from_array(dem,transform=tr,crs=crs,nodata=np.nan)
            nk=xdem.coreg.NuthKaab(); nk.fit(refD,tD,inlier_mask=st,random_state=42)
            dem=nk.apply(tD); dem=dem.data.filled(np.nan) if hasattr(dem.data,'filled') else np.asarray(dem.data,float)
        except Exception: pass
        # de-bias on stable terrain
        bias=np.nanmedian((dem-ref)[st&np.isfinite(dem-ref)])
        dem=dem-bias
        # summit zone = glacier & elev>SUMMIT_ELEV (use ref elevation to define zone consistently)
        summit = gm & (ref>SUMMIT_ELEV) & np.isfinite(dem)
        if summit.sum()<50: res.append((d,np.nan,0)); continue
        # elevation anomaly = summit median (dem) - summit median (ref)  [m]
        anom=np.nanmedian(dem[summit])-np.nanmedian(ref[summit])
        res.append((d,anom,int(summit.sum())))
    return res

print(f"Summit zone = glacier pixels above {SUMMIT_ELEV:.0f} m")
asc=process("155","ASCENDING")
desc=process("011","DESCENDING")

fig,axes=plt.subplots(2,1,figsize=(14,9),sharex=True)
for ax,res,ttl,col in [(axes[0],asc,"ASCENDING (155_0045)","#1f77b4"),
                        (axes[1],desc,"DESCENDING (011_0045)","#d62728")]:
    # eruption bars
    for s,e,lab,major in ERUPT:
        ax.axvspan(date2num(s),date2num(e),color=('firebrick' if major else 'orange'),
                   alpha=0.28 if major else 0.15,zorder=0)
        if major: ax.text(date2num(s),0.97,lab,rotation=90,va='top',ha='right',
                          fontsize=7,color='firebrick',transform=ax.get_xaxis_transform())
    if res:
        ds=[date2num(d) for d,a,n in res if np.isfinite(a)]
        ys=[a for d,a,n in res if np.isfinite(a)]
        ax.plot(ds,ys,'o-',color=col,ms=6,lw=1.4,zorder=3)
        ax.axhline(0,color='k',lw=0.6,ls=':')
    ax.set_ylabel("Summit-zone elevation\nanomaly [m]")
    ax.set_title(f"{ttl}   (n={len([1 for d,a,n in res if np.isfinite(a)]) if res else 0} winter DEMs)")
    ax.grid(alpha=0.3)
axes[1].xaxis.set_major_locator(mdates.YearLocator())
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[1].set_xlim(date2num(date(2011,6,1)),date2num(date(2026,3,1)))
fig.suptitle(f"Klyuchevskaya summit-glacier elevation vs eruptions\n"
             f"(summit = glacier >{SUMMIT_ELEV:.0f} m; red=major eruption, orange=minor)",fontsize=12)
fig.tight_layout()
fig.savefig(OUT/"SUMMIT_timeseries_vs_eruptions.png",dpi=140,bbox_inches="tight")
print("\n-> SUMMIT_timeseries_vs_eruptions.png")
# print the data
for res,lab in [(asc,'ASC'),(desc,'DESC')]:
    if res:
        print(f"\n{lab}:")
        for d,a,n in res:
            print(f"  {d}: anomaly={a:+.2f} m  (n={n} summit px)" if np.isfinite(a) else f"  {d}: (no summit data)")
