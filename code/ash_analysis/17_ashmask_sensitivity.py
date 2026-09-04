#!/usr/bin/env python3
"""
STEP 17: Ash-mask sensitivity — is the mask too small?
The production mask is conservative (NDSI drop < -0.2 AND NDSI_after < 0.3 AND
darker than the 40th brightness percentile, then morphological opening which
kills small patches). Here we compare three settings on the 2023 paroxysm and
2022 brackets and report how the ash AREA and the ash-vs-clean ANOMALY respond:
  strict   : production settings (as published so far)
  moderate : dNDSI<-0.15, NDSI_a<0.40, bright<p50, closing only (no opening)
  loose    : dNDSI<-0.10, NDSI_a<0.50, bright<p60, no morphology
If the anomaly is stable across settings, the finding is threshold-robust and
the larger 'moderate' mask is defensible as the new default.
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
GLINV=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/code/ash_analysis/GLINV_main_massif.gpkg")
crs="EPSG:32657"; RES=30.0
BRACKETS=[("2023 PAROXYSM","2023-09-29","2024-09-04",date(2023,11,1)),
          ("2022 Nov","2022-10-01","2023-08-05",date(2022,11,20))]
SETTINGS=[("strict",  -0.20,0.30,40,"open+close"),
          ("moderate",-0.15,0.40,50,"close"),
          ("loose",   -0.10,0.50,60,"none")]

def dem_path(dstr):
    for sd in glob.glob(f"{BASE}/{dstr[:4]}/{dstr}*155_0045*"):
        for t in ("DEM_FNL","DEM_VER"):
            g=glob.glob(f"{sd}/prc07/{t}_*.tif")
            if g: return g[0]

shp=gpd.read_file(GLINV).to_crs(crs)
allrows=[]
for lab,pre,post,ED in BRACKETS:
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
    ndA,brA=ndsi_bright(post,rows,cols,tr,crs,erupt_date=ED,side="post",gm=gm,label=f"{lab} post")
    ndB,_  =ndsi_bright(pre, rows,cols,tr,crs,erupt_date=ED,side="pre", gm=gm,label=f"{lab} pre")
    if ndA is None or ndB is None: print(f"{lab}: S2 fail"); continue
    dn=ndA-ndB
    masks={}
    for nm,thr_dn,thr_na,pct,morph in SETTINGS:
        br_thr=np.nanpercentile(brA[gm&np.isfinite(brA)],pct)
        m=gm&np.isfinite(dn)&(dn<thr_dn)&(ndA<thr_na)&(brA<br_thr)
        if morph=="open+close": m=ndimage.binary_closing(ndimage.binary_opening(m))
        elif morph=="close": m=ndimage.binary_closing(m)
        masks[nm]=m
        da=dh[m&np.isfinite(dh)]; dc=dh[gm&~m&np.isfinite(dh)]
        ma=np.nanmedian(da) if da.size>20 else np.nan
        mc=np.nanmedian(dc) if dc.size>20 else np.nan
        km2=m.sum()*RES*RES/1e6
        print(f"{lab} [{nm}]: {m.sum():,} px = {km2:.1f} km2 ({100*m.sum()/gm.sum():.1f}% glacier) "
              f"ash={ma:+.2f} clean={mc:+.2f} anomaly={ma-mc:+.2f} m")
        allrows.append((lab,nm,m.sum(),km2,ma-mc))
    # figure per bracket: one map panel PER threshold setting (previously all
    # three were overlaid as contours on one panel; the strict/moderate
    # contours were effectively invisible against the loose one -- separate
    # panels make each setting's actual extent unambiguous) + anomaly bars
    fig=plt.figure(figsize=(20,6)); gs=fig.add_gridspec(1,4,width_ratios=[1,1,1,0.8])
    ext=[xb.left,xb.right,xb.bottom,xb.top]
    mask_cols=["#39a339","#c9a227","#b05ab0"]
    for i,((nm,_,_,_,_),col) in enumerate(zip(SETTINGS,mask_cols)):
        axi=fig.add_subplot(gs[i])
        axi.imshow(ndA,cmap='RdBu',vmin=-0.5,vmax=1,extent=ext)
        if masks[nm].any():
            axi.contourf(np.flipud(masks[nm].astype(float)),levels=[.5,1.5],colors=[col],
                         alpha=0.55,extent=ext)
            axi.contour(np.flipud(masks[nm].astype(float)),levels=[.5],colors=col,linewidths=1.3,extent=ext)
        r=[x for x in allrows if x[0]==lab and x[1]==nm][0]
        axi.set_title(f"{nm}\n{r[3]:.0f} km², anomaly {r[4]:+.2f} m",color=col,fontweight="bold")
        axi.set_xticks([]); axi.set_yticks([])
    ax1=fig.add_subplot(gs[3])
    sub=[r for r in allrows if r[0]==lab]
    ax1.bar(range(3),[r[4] for r in sub],color=mask_cols)
    ax1.set_xticks(range(3)); ax1.set_xticklabels([r[1]+f"\n{r[3]:.0f} km²" for r in sub],fontsize=9)
    ax1.axhline(0,color='k',lw=0.7); ax1.set_ylabel("ash − clean anomaly [m]")
    for i,r in enumerate(sub): ax1.text(i,r[4]+(0.05 if r[4]>=0 else -0.05),f"{r[4]:+.2f}",
        ha='center',va='bottom' if r[4]>=0 else 'top',fontweight='bold',fontsize=9)
    ax1.set_title("anomaly stability across thresholds")
    fig.suptitle(f"Ash–mask sensitivity — {lab}",fontsize=13)
    safe=lab.replace(' ','_')
    fig.savefig(AOUT/f"ASHMASK_sensitivity_{safe}.png",dpi=600,bbox_inches="tight"); plt.close(fig)
    print(f"-> ASHMASK_sensitivity_{safe}.png")
