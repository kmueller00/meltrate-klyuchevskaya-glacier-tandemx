#!/usr/bin/env python3
"""
STEP 10: Bootstrap significance of the ash-vs-clean elevation anomaly per eruption.

For each eruption's summer-to-summer bracket we have per-pixel dH on ash vs clean
glacier. The statistic is  D = median(dH_ash) - median(dH_clean).
We bootstrap D (resample pixels with replacement, N=2000) to get a 95% CI and a
two-sided p-value (fraction of bootstrap D on the opposite side of 0). Because
neighbouring 30 m pixels are spatially correlated, we ALSO report a block/effective
-N bootstrap (coarsen ash & clean to ~5-px blocks) so the CI isn't over-narrowed by
pseudo-replication. Emits a forest-plot of D +/- CI per eruption.
"""
import sys; sys.path.insert(0,"/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/code/ash_analysis")
import glob, numpy as np, rasterio, rasterio.features, geopandas as gpd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datetime import date
from pathlib import Path
from rasterio.warp import reproject, Resampling
from scipy import ndimage
from s2_util import ndsi_bright

BASE=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation/ash_analysis")
GLINV=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/code/ash_analysis/GLINV_main_massif.gpkg")  # main massif only (southern tongue-only glaciers excluded)
crs="EPSG:32657"; RES=30.0
NB=2000; rng=np.random.default_rng(42)

BRACKETS=[("2015 Aug-Sep","2015-08-23","2016-08-31",date(2015,8,27),"landsat",150),
          ("2019 Oct","2019-09-27","2020-09-02",date(2019,10,25),"s2",60),
          ("2020 Dec","2020-09-24","2021-08-31",date(2020,12,9),"s2",60),
          ("2022 Nov","2022-10-01","2023-08-05",date(2022,11,20),"s2",60),
          ("2023 PAROXYSM","2023-09-29","2024-09-04",date(2023,11,1),"s2",60)]

def dem_path(dstr):
    for sd in glob.glob(f"{BASE}/{dstr[:4]}/{dstr}*155_0045*"):
        for t in ("DEM_FNL","DEM_VER"):
            g=glob.glob(f"{sd}/prc07/{t}_*.tif")
            if g: return g[0]

def boot(a,c,block=None):
    """bootstrap median(a)-median(c); block=coarsen factor for spatial correlation."""
    if block:
        a=a[::block]; c=c[::block]     # thin to reduce pseudo-replication (already flattened)
    D=np.empty(NB)
    na,nc=len(a),len(c)
    for i in range(NB):
        D[i]=np.median(a[rng.integers(0,na,na)])-np.median(c[rng.integers(0,nc,nc)])
    d0=np.median(a)-np.median(c)
    lo,hi=np.percentile(D,[2.5,97.5])
    # two-sided p: fraction of bootstrap on opposite side of observed sign, x2
    p=2*min((D<=0).mean(),(D>=0).mean())
    return d0,lo,hi,p

shp=gpd.read_file(GLINV).to_crs(crs)
rows_out=[]
for lab,pre,post,ED,src,wd in BRACKETS:
    fA=dem_path(post); fB=dem_path(pre)
    if not(fA and fB): print(f"{lab}: missing DEM"); continue
    with rasterio.open(fA) as s:
        A=s.read(1).astype(float); A[(A==s.nodata)|(A<0)|(A>5000)]=np.nan
        tr=s.transform; rows,cols=s.shape
    with rasterio.open(fB) as s:
        b=s.read(1).astype(float); b[(b==s.nodata)|(b<0)|(b>5000)]=np.nan
        B=np.full((rows,cols),np.nan,'float32')
        reproject(b.astype('float32'),B,src_transform=s.transform,src_crs=s.crs,
                  dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear)
    dh=A-B; dh=np.where(np.abs(dh)<=120,dh,np.nan)
    gm=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
        out_shape=(rows,cols),transform=tr,fill=0,dtype='uint8').astype(bool)
    stv=dh[(~gm)&np.isfinite(dh)]
    if stv.size>1000: dh=dh-np.nanmedian(stv)
    ndA,brA=ndsi_bright(post,rows,cols,tr,crs,erupt_date=ED,side="post",gm=gm,label=f"{lab} post",source=src,wide_days=wd)
    ndB,_  =ndsi_bright(pre, rows,cols,tr,crs,erupt_date=ED,side="pre", gm=gm,label=f"{lab} pre", source=src,wide_days=wd)
    if ndA is None or ndB is None: print(f"{lab}: S2 fail"); continue
    dn=ndA-ndB; br_thr=np.nanpercentile(brA[gm&np.isfinite(brA)],40)
    ash=gm&np.isfinite(dn)&(dn<-0.2)&(ndA<0.3)&(brA<br_thr)
    ash=ndimage.binary_opening(ash); ash=ndimage.binary_closing(ash)
    da=dh[ash&np.isfinite(dh)]; dc=dh[gm&~ash&np.isfinite(dh)]
    if da.size<50: print(f"{lab}: too few ash px"); continue
    d0,lo,hi,p=boot(da,dc)
    # block bootstrap (thin ~5x -> ~150m decorrelation) for honest CI
    d0b,lob,hib,pb=boot(da,dc,block=5)
    sig="***" if pb<0.001 else "**" if pb<0.01 else "*" if pb<0.05 else "n.s."
    print(f"{lab}: D={d0:+.2f}m  95%CI[{lo:+.2f},{hi:+.2f}] p={p:.4f} | "
          f"block: CI[{lob:+.2f},{hib:+.2f}] p={pb:.4f} {sig}  (ash n={da.size:,})")
    rows_out.append((lab,d0b,lob,hib,pb,sig,da.size))

# forest plot (block-bootstrap CI = the honest one)
if rows_out:
    fig,ax=plt.subplots(figsize=(9,0.9*len(rows_out)+2))
    y=np.arange(len(rows_out))[::-1]
    for yi,(lab,d0,lo,hi,p,sig,n) in zip(y,rows_out):
        col='#2c7bb6' if d0>0 else '#d7191c'
        ax.plot([lo,hi],[yi,yi],'-',color=col,lw=2.5)
        ax.plot(d0,yi,'o',color=col,ms=10)
        ax.text(hi+0.1,yi,f"{d0:+.2f} m  {sig}\n(n={n:,})",va='center',fontsize=9)
    ax.axvline(0,color='k',lw=1,ls='--')
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows_out])
    ax.set_xlabel("ash − clean median dH  [m]   (block–bootstrap 95% CI)")
    ax.set_title("Ash–relative deposition anomaly per eruption — significance\n"
                 "positive = deposition/retention; CI excludes 0 → significant\n"
                 "*** p<0.001  ** p<0.01  * p<0.05")
    ax.margins(x=0.25); fig.tight_layout()
    fig.savefig(AOUT/"BOOTSTRAP_significance.png",dpi=600,bbox_inches="tight")
    print("-> BOOTSTRAP_significance.png")
