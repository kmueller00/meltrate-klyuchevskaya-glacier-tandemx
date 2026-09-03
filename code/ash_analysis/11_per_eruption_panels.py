#!/usr/bin/env python3
"""
STEP 11: Per-eruption depiction. One 3-panel figure per eruption:
  [1] Sentinel-2 NDSI post-eruption melt season (snow vs ash) + ash contour
  [2] DEM dH over the summer-to-summer bracket + ash contour
  [3] boxplot ash vs clean dH, with the bootstrap D and significance annotated
Uses the SAME snow-free summer brackets and ash logic as steps 8/10.
"""
import sys; sys.path.insert(0,"/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/code/ash_analysis")
import glob, numpy as np, rasterio, rasterio.features, geopandas as gpd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datetime import date, timedelta
from pathlib import Path
from rasterio.warp import reproject, Resampling
from scipy import ndimage
from s2_util import ndsi_bright as _ndsi_bright_shared

BASE=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation/ash_analysis")
GLINV=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/code/ash_analysis/GLINV_main_massif.gpkg")  # main massif only (southern tongue-only glaciers excluded)
crs="EPSG:32657"; RES=30.0
rng=np.random.default_rng(7)

# (label, pre-summer, post-summer, eruption description, eruption date)
ERUPS=[("2019 Oct Strombolian","2019-09-27","2020-09-02","minor Strombolian, 25 Oct 2019",date(2019,10,25)),
       ("2020 Dec eruption",   "2020-09-24","2021-08-31","effusive, Dec 2020",date(2020,12,9)),
       ("2022 Nov eruption",   "2022-10-01","2023-08-05","explosive, 20 Nov 2022",date(2022,11,20)),
       ("2023-24 PAROXYSM",    "2023-09-29","2024-09-04","paroxysm, Nov 2023 (largest)",date(2023,11,1))]

def dem_path(dstr):
    for sd in glob.glob(f"{BASE}/{dstr[:4]}/{dstr}*155_0045*"):
        for t in ("DEM_FNL","DEM_VER"):
            g=glob.glob(f"{sd}/prc07/{t}_*.tif")
            if g: return g[0]

def boot_block(a,c,nb=1500,block=5):
    a=a[::block]; c=c[::block]; D=np.empty(nb)
    for i in range(nb):
        D[i]=np.median(a[rng.integers(0,len(a),len(a))])-np.median(c[rng.integers(0,len(c),len(c))])
    d0=np.median(a)-np.median(c); lo,hi=np.percentile(D,[2.5,97.5])
    p=2*min((D<=0).mean(),(D>=0).mean()); return d0,lo,hi,p

shp=gpd.read_file(GLINV).to_crs(crs)
for lab,pre,post,desc,ED in ERUPS:
    fA=dem_path(post); fB=dem_path(pre)
    if not(fA and fB): print(f"{lab}: missing DEM"); continue
    with rasterio.open(fA) as s:
        A=s.read(1).astype(float); A[(A==s.nodata)|(A<0)|(A>5000)]=np.nan
        tr=s.transform; rows,cols=s.shape; xb=s.bounds
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
    ndA,brA=_ndsi_bright_shared(post,rows,cols,tr,crs,erupt_date=ED,side="post",gm=gm,label=f"{lab} post")
    ndB,_  =_ndsi_bright_shared(pre, rows,cols,tr,crs,erupt_date=ED,side="pre", gm=gm,label=f"{lab} pre")
    if ndA is None or ndB is None: print(f"{lab}: S2 fail"); continue
    dn=ndA-ndB; br_thr=np.nanpercentile(brA[gm&np.isfinite(brA)],40)
    ash=gm&np.isfinite(dn)&(dn<-0.2)&(ndA<0.3)&(brA<br_thr)
    ash=ndimage.binary_opening(ash); ash=ndimage.binary_closing(ash)
    da=dh[ash&np.isfinite(dh)]; dc=dh[gm&~ash&np.isfinite(dh)]
    ma=np.nanmedian(da) if da.size>20 else np.nan; mc=np.nanmedian(dc) if dc.size>20 else np.nan
    if da.size>=50:
        d0,lo,hi,p=boot_block(da,dc)
        sig="***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."
    else: d0,lo,hi,p,sig=np.nan,np.nan,np.nan,np.nan,"—"
    ext=[xb.left,xb.right,xb.bottom,xb.top]
    fig=plt.figure(figsize=(17,5.5))
    gs=fig.add_gridspec(1,4,width_ratios=[1.1,1.1,0.06,0.85],wspace=0.35)
    ax0=fig.add_subplot(gs[0]); ax1=fig.add_subplot(gs[1]); cax=fig.add_subplot(gs[2]); ax2=fig.add_subplot(gs[3])
    ax0.imshow(ndA,cmap='RdBu',vmin=-0.5,vmax=1,extent=ext)
    if ash.any(): ax0.contour(np.flipud(ash.astype(float)),levels=[.5],colors='lime',linewidths=1.2,extent=ext)
    ax0.set_title(f"S2 NDSI post–eruption ({post[:7]})\nblue=snow  red=ash/rock  (green=ash mask)")
    gvv=dh[gm&np.isfinite(dh)]; vlim=max(3,np.nanpercentile(np.abs(gvv),95)) if gvv.size else 5
    im=ax1.imshow(np.where(gm,dh,np.nan),cmap='RdBu_r',vmin=-vlim,vmax=vlim,extent=ext)
    if ash.any(): ax1.contour(np.flipud(ash.astype(float)),levels=[.5],colors='lime',linewidths=1.2,extent=ext)
    ax1.set_title(f"DEM dH {pre}→{post}\nred=gain(deposition) blue=loss(melt)")
    fig.colorbar(im,cax=cax,label="dH [m]")
    if da.size>20 and dc.size>20:
        ax2.boxplot([dc[np.isfinite(dc)],da[np.isfinite(da)]],labels=['clean','ash'],
                    showfliers=False,widths=0.6,medianprops=dict(color='k',lw=2))
        ax2.axhline(0,color='grey',ls=':'); ax2.set_ylabel("dH [m]")
        ax2.yaxis.set_label_position("right"); ax2.yaxis.tick_right()
        ax2.set_title(f"ash {ma:+.1f}  clean {mc:+.1f} m\nanomaly {d0:+.2f} m  {sig}")
    for a in (ax0,ax1): a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{lab}  —  {desc}",fontsize=14,fontweight='bold')
    safe=lab.replace(' ','_').replace('/','-').replace('—','-')
    fig.savefig(AOUT/f"ERUPTION_{safe}.png",dpi=600,bbox_inches="tight"); plt.close(fig)
    print(f"{lab}: ash {ma:+.2f} clean {mc:+.2f} anomaly {d0:+.2f} {sig} (n={da.size:,}) -> ERUPTION_{safe}.png")
