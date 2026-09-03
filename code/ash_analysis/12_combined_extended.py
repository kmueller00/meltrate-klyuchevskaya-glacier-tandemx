#!/usr/bin/env python3
"""
STEP 12: EXTENDED combined glacier elevation signal 2012-2025.
Two non-overlapping TanDEM-X eras (different tracks, cannot cross-coregister):
  - DESCENDING 011 : 2012-2014 (early segment)
  - ASCENDING  155 : 2015-2025 (main record)
Each era is internally coregistered to its own reference and expressed as a
glacier-median elevation anomaly. Plotted on one axis with a break note so the
long-term picture reaches back to 2012. Eruptions marked.
"""
import glob, re, numpy as np, rasterio, rasterio.features, geopandas as gpd, xdem
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import date2num
from pathlib import Path
from datetime import date
from collections import defaultdict
from rasterio.warp import reproject, Resampling

BASE=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation")
GLINV=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b/"
           "2025/2025-01-14_065954_TDT_SM_A_97516_155_0045+1-2/prc05_refdem/"
           "GLINV_rgi06_ttl_utm57_N_b_UTM.shp")
crs="EPSG:32657"; RES=30.0

ERUPT=[(date(2013,1,25),date(2013,11,30),"2013 MAJOR",1),
 (date(2015,1,2),date(2015,3,24),"2015",0),(date(2015,8,27),date(2015,9,10),"2015 Aug",0),
 (date(2016,4,1),date(2016,5,1),"2016",0),(date(2019,4,1),date(2019,11,1),"2019",1),
 (date(2020,10,2),date(2021,3,25),"2020-21",1),(date(2022,11,20),date(2022,12,20),"2022",0),
 (date(2023,6,22),date(2024,1,15),"2023-24 PAROXYSM",1),(date(2024,8,1),date(2024,10,15),"2024",0)]

def find(sd):
    for t in ("DEM_FNL","DEM_VER"):
        g=glob.glob(str(sd/"prc07"/f"{t}_*.tif"))
        if g: return g[0]

def collect(track,y0,y1):
    dems=[]
    for sd in [Path(p) for p in glob.glob(str(BASE/"20*/20*"))]:
        nm=sd.name; m=re.match(r"(\d{4})-(\d{2})-(\d{2})",nm)
        if not m or f"{track}_0045" not in nm: continue
        d=date(*map(int,m.groups()))
        if not (y0<=d.year<=y1): continue
        f=find(sd)
        if not f: continue
        with rasterio.open(f) as s:
            a=s.read(1).astype(float); a[(a==s.nodata)|(a<0)|(a>5000)]=np.nan; v=a[np.isfinite(a)]
        if v.size<3e5 or not (800<np.median(v)<2500): continue
        dems.append((d,f))
    return sorted(set(dems))

def era_series(dems):
    if len(dems)<2: return []
    ref_f=dems[-1][1]
    with rasterio.open(ref_f) as s: tr=s.transform; rows,cols=s.shape
    def rg(f):
        with rasterio.open(f) as s:
            a=s.read(1).astype("float32"); a[(a==s.nodata)|(a<0)|(a>5000)]=np.nan
            d=np.full((rows,cols),np.nan,"float32")
            reproject(a,d,src_transform=s.transform,src_crs=s.crs,dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear)
        d[(d<0)|(d>5000)]=np.nan; return d
    shp=gpd.read_file(GLINV).to_crs(crs)
    gm=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
        out_shape=(rows,cols),transform=tr,fill=0,dtype="uint8").astype(bool); st=~gm
    ref=rg(ref_f); refD=xdem.DEM.from_array(ref,transform=tr,crs=crs,nodata=np.nan)
    out=[]
    for d,f in dems:
        dem=rg(f)
        try:
            tD=xdem.DEM.from_array(dem,transform=tr,crs=crs,nodata=np.nan)
            nk=xdem.coreg.NuthKaab(); nk.fit(refD,tD,inlier_mask=st,random_state=42)
            dem=nk.apply(tD); dem=dem.data.filled(np.nan) if hasattr(dem.data,'filled') else np.asarray(dem.data,float)
        except Exception: pass
        dh=dem-ref; dh=np.where(np.abs(dh)<=50,dh,np.nan)
        b=np.nanmedian(dh[st&np.isfinite(dh)]); dh=dh-(b if np.isfinite(b) else 0)
        gv=dh[gm&np.isfinite(dh)]
        if gv.size<500: continue
        stv=dh[st&np.isfinite(dh)]
        nmad=1.4826*np.nanmedian(np.abs(stv-np.nanmedian(stv))) if stv.size>500 else 999
        anom=np.nanmedian(gv)
        if nmad>6 or abs(anom)>15: continue
        seas="W" if (d.month>=11 or d.month<=4) else "S"
        out.append((d,anom,seas))
    # merge per date
    bd=defaultdict(list)
    for d,a,se in out: bd[d].append((a,se))
    return sorted([(d,float(np.median([a for a,_ in v])),v[0][1]) for d,v in bd.items()])

print("DESCENDING 011 (2012-2014)..."); desc=era_series(collect("011",2012,2014))
print(f"  {len(desc)} points"); print("ASCENDING 155 (2015-2025)..."); asc=era_series(collect("155",2015,2025))
print(f"  {len(asc)} points")

# each era referenced to its own last DEM -> offset era means so they align at the seam (2014/2015)
# express both as anomaly vs each era's mean, then stitch by matching the trend (report separately, don't fake-join)
fig,ax=plt.subplots(figsize=(16,6))
for s,e,lab,major in ERUPT:
    ax.axvspan(date2num(s),date2num(e),color=('firebrick' if major else 'orange'),alpha=0.25 if major else 0.13,zorder=0)
    ax.text(date2num(s),0.98,lab,rotation=90,va='top',ha='right',fontsize=7,
            color=('firebrick' if major else 'darkorange'),transform=ax.get_xaxis_transform())
for series,col,mk,name in [(desc,'#6a3d9a','D','DESCENDING 011 (2012-14)'),
                            (asc,'#1f77b4','o','ASCENDING 155 (2015-25)')]:
    if not series: continue
    W=[(date2num(d),a) for d,a,se in series if se=="W"]; S=[(date2num(d),a) for d,a,se in series if se=="S"]
    pts=sorted([(date2num(d),a) for d,a,se in series])
    ax.plot(*zip(*pts),'-',color=col,lw=1,alpha=0.5,zorder=2)
    if W: ax.scatter(*zip(*W),s=55,c=col,marker=mk,edgecolor='k',lw=0.4,zorder=3)
    if S: ax.scatter(*zip(*S),s=60,c=col,marker='^',edgecolor='k',lw=0.4,zorder=3)
ax.axvline(date2num(date(2014,10,1)),color='grey',ls='--',lw=1)
ax.text(date2num(date(2014,10,1)),0.02,' track change\n 011→155',transform=ax.get_xaxis_transform(),fontsize=7,color='grey',va='bottom')
ax.axhline(0,color='k',lw=0.6,ls=':')
ax.set_ylabel("glacier–median elevation anomaly [m]\n(each era vs its own last DEM)")
ax.set_title("Klyuchevskaya glacier surface elevation 2012–2025 — EXTENDED\n"
             "purple ◆=descending(2012–14, winter)  blue ●=ascending winter  ▲=summer; red=major eruption")
ax.grid(alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_xlim(date2num(date(2012,1,1)),date2num(date(2026,3,1)))
fig.tight_layout(); fig.savefig(AOUT/"COMBINED_signal_2012-2025_extended.png",dpi=600,bbox_inches="tight")
print("-> COMBINED_signal_2012-2025_extended.png")
