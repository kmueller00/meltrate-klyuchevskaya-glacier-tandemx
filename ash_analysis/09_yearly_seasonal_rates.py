#!/usr/bin/env python3
"""
STEP 9: Yearly elevation-change RATE (m/yr), season-matched.
  - SUMMER->SUMMER : consecutive late-summer (Aug-Sep) medians  -> ablation-referenced rate
  - WINTER->WINTER : consecutive late-winter (Feb-Mar) medians   -> accumulation-referenced rate
Season-matching removes the seasonal snow cycle so the rate reflects true annual
mass change. Each map is glacier-masked, Nuth-Kaab coregistered, outlier-clipped.
Also emits a per-year glacier-median rate table for both seasons.
"""
import glob, re, numpy as np, rasterio, rasterio.features, geopandas as gpd, xdem
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date
from collections import defaultdict
from rasterio.warp import reproject, Resampling

BASE=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation")
GLINV=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b/"
           "2025/2025-01-14_065954_TDT_SM_A_97516_155_0045+1-2/prc05_refdem/"
           "GLINV_rgi06_ttl_utm57_N_b_UTM.shp")
GLINV=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/ash_analysis/GLINV_main_massif.gpkg")  # OVERRIDE: main massif only (southern tongue-only glaciers excluded)
crs="EPSG:32657"; RES=30.0; CLIP=8.0

def find(sd):
    for t in ("DEM_FNL","DEM_VER"):
        g=glob.glob(str(sd/"prc07"/f"{t}_*.tif"))
        if g: return g[0]

# collect DEMs by (season, year); SUMMER=Aug-Sep, WINTER=Feb-Mar
def collect(season):
    months=(8,9) if season=="summer" else (2,3)
    byyear=defaultdict(list)
    for yd in sorted(BASE.glob("20*")):
        for sd in sorted(yd.glob("20*")):
            nm=sd.name; m=re.match(r"(\d{4})-(\d{2})-(\d{2})",nm)
            if not m or "155_0045" not in nm: continue
            y,mo,d=map(int,m.groups())
            if mo not in months: continue
            f=find(sd)
            if not f: continue
            with rasterio.open(f) as s:
                a=s.read(1).astype(float); a[(a==s.nodata)|(a<0)|(a>5000)]=np.nan
                v=a[np.isfinite(a)]
            if v.size<3e5 or not (800<np.median(v)<2500): continue
            byyear[y].append(f)
    return byyear

RES_GRID=30.0
def build(season):
    byyear=collect(season); years=sorted(byyear)
    if len(years)<2: print(f"{season}: <2 years"); return None,None,None,None
    allf=[f for y in years for f in byyear[y]]
    bs=[]
    for f in allf:
        with rasterio.open(f) as s: bs.append(s.bounds)
    xmin=max(b.left for b in bs); xmax=min(b.right for b in bs)
    ymin=max(b.bottom for b in bs); ymax=min(b.top for b in bs)
    xmin=np.ceil(xmin/RES_GRID)*RES_GRID; xmax=np.floor(xmax/RES_GRID)*RES_GRID
    ymin=np.ceil(ymin/RES_GRID)*RES_GRID; ymax=np.floor(ymax/RES_GRID)*RES_GRID
    cols=int((xmax-xmin)/RES_GRID); rows=int((ymax-ymin)/RES_GRID)
    tr=rasterio.transform.from_origin(xmin,ymax,RES_GRID,RES_GRID)
    def rg(f):
        with rasterio.open(f) as s:
            a=s.read(1).astype("float32"); a[(a==s.nodata)|(a<0)|(a>5000)]=np.nan
            d=np.full((rows,cols),np.nan,"float32")
            reproject(a,d,src_transform=s.transform,src_crs=s.crs,dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear)
        d[(d<0)|(d>5000)]=np.nan; return d
    shp=gpd.read_file(GLINV).to_crs(crs)
    gm=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
        out_shape=(rows,cols),transform=tr,fill=0,dtype="uint8").astype(bool); st=~gm
    # yearly median dem
    ymed={y:np.nanmedian(np.stack([rg(f) for f in byyear[y]]),axis=0) for y in years}
    pairs=[(years[i],years[i+1]) for i in range(len(years)-1)]
    maps={}
    for ya,yb in pairs:
        A=ymed[ya].copy(); B=ymed[yb].copy()
        try:
            rA=xdem.DEM.from_array(A,transform=tr,crs=crs,nodata=np.nan)
            rB=xdem.DEM.from_array(B,transform=tr,crs=crs,nodata=np.nan)
            nk=xdem.coreg.NuthKaab(); nk.fit(rA,rB,inlier_mask=st,random_state=42)
            B=nk.apply(rB); B=B.data.filled(np.nan) if hasattr(B.data,'filled') else np.asarray(B.data,float)
        except Exception: pass
        dh=B-A; b=np.nanmedian(dh[st&np.isfinite(dh)]); dh=dh-(b if np.isfinite(b) else 0)
        rate=dh/(yb-ya)
        rate=np.where(np.abs(rate)<=CLIP,rate,np.nan)
        g=rate[gm&np.isfinite(rate)]; med=np.median(g) if g.size else np.nan
        nmad=1.4826*np.median(np.abs(g-med)) if g.size else np.nan
        rate=np.where(np.abs(rate-med)<=3*nmad,rate,np.nan)
        maps[(ya,yb)]=(np.where(gm,rate,np.nan),med,nmad)
    return maps,tr,(xmin,xmax,ymin,ymax),gm

for season in ("summer","winter"):
    maps,tr,ext,gm=build(season)
    if not maps: continue
    xmin,xmax,ymin,ymax=ext
    pairs=sorted(maps)
    n=len(pairs); ncol=4; nrow=int(np.ceil(n/ncol))
    fig,axes=plt.subplots(nrow,ncol,figsize=(4.2*ncol,4.2*nrow))
    axes=np.array(axes).ravel()
    for k,(ya,yb) in enumerate(pairs):
        rate,med,nmad=maps[(ya,yb)]
        ax=axes[k]
        im=ax.imshow(rate,cmap="RdBu",vmin=-CLIP,vmax=CLIP,extent=[xmin,xmax,ymin,ymax])
        ax.set_title(f"{ya}→{yb}   {med:+.2f} m/yr",fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        prof=dict(driver="GTiff",height=rate.shape[0],width=rate.shape[1],count=1,dtype="float32",
                  crs=crs,transform=tr,nodata=-9999.0,compress="lzw")
        with rasterio.open(AOUT/f"RATE_{season}_{ya}_{yb}.tif","w",**prof) as dd:
            dd.write(np.where(np.isfinite(rate),rate,-9999.0).astype("float32"),1)
    for k in range(n,len(axes)): axes[k].axis('off')
    cb=fig.colorbar(im,ax=axes.tolist(),shrink=0.6,label="elevation-change rate [m yr$^{-1}$]")
    fig.suptitle(f"Klyuchevskaya glaciers — {season.upper()}–to–{season.upper()} elevation–change rate (ascending 155)\n"
                 f"season–matched consecutive–year pairs; blue=gain red=loss",fontsize=13)
    fig.savefig(AOUT/f"RATE_{season}_panel.png",dpi=600,bbox_inches="tight")
    print(f"\n=== {season.upper()} yearly rates (glacier median) ===")
    for (ya,yb),(_,med,nmad) in sorted(maps.items()):
        print(f"  {ya}->{yb}: {med:+.3f} m/yr  (spatial NMAD {nmad:.2f})")
    print(f"-> RATE_{season}_panel.png + per-pair tifs")
