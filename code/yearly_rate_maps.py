#!/usr/bin/env python3
"""
Per-year glacier elevation-change RATE maps (m/yr), ascending 155.
Each map = (year_B median DEM - year_A median DEM)/dt, coregistered (Nuth&Kaab)
on stable terrain, glacier-masked, outlier-clipped. Eruption intervals annotated.
A multi-panel figure shows consecutive year-pairs; the 2023->2024 panel brackets
the major 2023-24 paroxysm.
"""
import numpy as np, rasterio, rasterio.features, geopandas as gpd, xdem
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date
import glob, re
from rasterio.warp import reproject, Resampling
from collections import defaultdict

BASE=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
OUT =Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation")
GLINV=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b/"
           "2025/2025-01-14_065954_TDT_SM_A_97516_155_0045+1-2/prc05_refdem/"
           "GLINV_rgi06_ttl_utm57_N_b_UTM.shp")
RES=30.0; crs="EPSG:32657"; CLIP=8.0   # +/- m/yr display/clip

# eruption presence per interval (which year-pairs bracket an eruption)
ERUPT_YEARS={2019,2020,2022,2023,2024}  # eruptions occurred in these years (KVERT/GVP)

def find(sd):
    for t in ("DEM_FNL","DEM_VER"):
        g=glob.glob(str(sd/"prc07"/f"{t}_*.tif"))
        if g: return g[0]

# collect full-coverage ascending winter DEMs by year
byyear=defaultdict(list)
for yd in sorted(BASE.glob("20*")):
    for sd in sorted(yd.glob("20*")):
        nm=sd.name; m=re.match(r"(\d{4})-(\d{2})-(\d{2})",nm)
        if not m or "155_0045" not in nm: continue
        y,mo,d=map(int,m.groups())
        if not (mo>=11 or mo<=4): continue
        f=find(sd)
        if not f: continue
        with rasterio.open(f) as s:
            a=s.read(1).astype(float); a[(a==s.nodata)]=np.nan
            h=(s.bounds.top-s.bounds.bottom)/1000
        # CLEAN to physical elevation range (0-5000m); removes garbage edge pixels
        a[(a<0)|(a>5000)]=np.nan
        v=a[np.isfinite(a)]
        if v.size<5e5: continue                    # too few valid pixels
        if not (800<np.median(v)<2500): continue   # implausible median = bad DEM
        if h>100: byyear[y].append(f)
years=sorted(byyear)
print("years (cleaned to physical range):",{y:len(byyear[y]) for y in years})

# common grid over ALL
allf=[f for y in years for f in byyear[y]]
bs=[]
for f in allf:
    with rasterio.open(f) as s: bs.append(s.bounds)
xmin=max(b.left for b in bs);xmax=min(b.right for b in bs)
ymin=max(b.bottom for b in bs);ymax=min(b.top for b in bs)
xmin=np.ceil(xmin/RES)*RES;xmax=np.floor(xmax/RES)*RES
ymin=np.ceil(ymin/RES)*RES;ymax=np.floor(ymax/RES)*RES
cols=int((xmax-xmin)/RES);rows=int((ymax-ymin)/RES)
tr=rasterio.transform.from_origin(xmin,ymax,RES,RES)
def rg(f):
    with rasterio.open(f) as s:
        src=s.read(1).astype("float32");src[(src==s.nodata)|(src<0)|(src>5000)]=np.nan
        dst=np.full((rows,cols),np.nan,"float32")
        reproject(src,dst,src_transform=s.transform,src_crs=s.crs,dst_transform=tr,dst_crs=crs,
                  resampling=Resampling.bilinear,src_nodata=np.nan,dst_nodata=np.nan)
    dst[(dst<0)|(dst>5000)]=np.nan; return dst
shp=gpd.read_file(GLINV).to_crs(crs)
gm=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
    out_shape=(rows,cols),transform=tr,fill=0,dtype="uint8").astype(bool)
st=~gm

# yearly MEDIAN dem
ymed={}
for y in years:
    ymed[y]=np.nanmedian(np.stack([rg(f) for f in byyear[y]]),axis=0)

# consecutive year-pairs
pairs=[(years[i],years[i+1]) for i in range(len(years)-1)]
n=len(pairs); ncol=3; nrow=int(np.ceil(n/ncol))
fig,axes=plt.subplots(nrow,ncol,figsize=(5*ncol,5*nrow))
axes=np.array(axes).ravel()
maps={}
for k,(ya,yb) in enumerate(pairs):
    A=ymed[ya].copy(); B=ymed[yb].copy()
    # coreg B->A on stable terrain
    try:
        rA=xdem.DEM.from_array(A,transform=tr,crs=crs,nodata=np.nan)
        rB=xdem.DEM.from_array(B,transform=tr,crs=crs,nodata=np.nan)
        nk=xdem.coreg.NuthKaab(); nk.fit(rA,rB,inlier_mask=st,random_state=42)
        B=nk.apply(rB); B=B.data.filled(np.nan) if hasattr(B.data,'filled') else np.asarray(B.data,float)
    except Exception: pass
    dh=B-A; b=np.nanmedian(dh[st&np.isfinite(dh)]); dh=dh-(b if np.isfinite(b) else 0)
    rate=dh/(yb-ya)
    # hard physical clip first (removes any residual garbage), then 3*NMAD
    rate=np.where(np.abs(rate)<=CLIP, rate, np.nan)
    g=rate[gm&np.isfinite(rate)]; med=np.median(g) if g.size else np.nan
    nmad=1.4826*np.median(np.abs(g-med)) if g.size else np.nan
    rate=np.where(np.abs(rate-med)<=3*nmad,rate,np.nan)  # clip
    rate_g=np.where(gm,rate,np.nan)
    maps[(ya,yb)]=(rate_g,med)
    # save tif
    prof=dict(driver="GTiff",height=rows,width=cols,count=1,dtype="float32",crs=crs,
              transform=tr,nodata=-9999.0,compress="lzw")
    with rasterio.open(OUT/f"YEARLY_rate_{ya}_{yb}.tif","w",**prof) as dd:
        dd.write(np.where(np.isfinite(rate_g),rate_g,-9999.0).astype("float32"),1)
    ax=axes[k]
    im=ax.imshow(rate_g,cmap="RdBu",vmin=-CLIP,vmax=CLIP,extent=[xmin,xmax,ymin,ymax])
    erupt = (ya in ERUPT_YEARS) or (yb in ERUPT_YEARS)
    title=f"{ya}→{yb}   {med:+.2f} m/yr"
    ax.set_title(title + ("  \U0001F30B ERUPTION" if erupt else ""),
                 fontsize=11, color=('darkred' if erupt else 'k'),
                 fontweight=('bold' if erupt else 'normal'))
    if erupt:
        for s in ax.spines.values(): s.set_color('red'); s.set_linewidth(2.5)
    ax.set_xticks([]); ax.set_yticks([])
for k in range(n,len(axes)): axes[k].axis('off')
cb=fig.colorbar(im,ax=axes.tolist(),shrink=0.6,label="elevation-change rate [m yr$^{-1}$]")
fig.suptitle("Klyuchevskaya glaciers — per-interval elevation-change rate (ascending 155)\n"
             "red frame + \U0001F30B = interval brackets an eruption (KVERT/GVP)",fontsize=13)
fig.savefig(OUT/"YEARLY_rate_maps_panel.png",dpi=140,bbox_inches="tight")
print("\n-> YEARLY_rate_maps_panel.png  + per-pair YEARLY_rate_<a>_<b>.tif")
for (ya,yb),(_,med) in maps.items():
    e="ERUPTION" if (ya in ERUPT_YEARS or yb in ERUPT_YEARS) else ""
    print(f"  {ya}->{yb}: {med:+.3f} m/yr  {e}")
