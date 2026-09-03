#!/usr/bin/env python3
"""
STEP 2 (v2): Map ash/lava on glacier from Sentinel-2 (MOSAICKED over all tiles),
test DEM height change on ash vs clean glacier. Robust DEM cleaning applied.
"""
import json, glob, re
import numpy as np, rasterio, rasterio.features, geopandas as gpd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date, timedelta
import pystac_client, planetary_computer, odc.stac
from rasterio.warp import reproject, Resampling

BASE=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation/ash_analysis")
GLINV=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b/"
           "2025/2025-01-14_065954_TDT_SM_A_97516_155_0045+1-2/prc05_refdem/"
           "GLINV_rgi06_ttl_utm57_N_b_UTM.shp")
crs="EPSG:32657"; RES=30.0; BBOX=[159.889,55.472,161.359,56.522]
matches=json.load(open(AOUT/"eruption_matches.json"))
CAT=pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1',
                              modifier=planetary_computer.sign_inplace)

def clean_dem(a,nd):
    a=a.astype(float); a[(a==nd)|(a<0)|(a>5000)]=np.nan; return a

def dem_on_grid(dstr,track,rows,cols,tr):
    for sd in glob.glob(f"{BASE}/{dstr[:4]}/{dstr}*{track}_0045*"):
        for t in ("DEM_FNL","DEM_VER"):
            g=glob.glob(f"{sd}/prc07/{t}_*.tif")
            if g:
                with rasterio.open(g[0]) as s:
                    a=clean_dem(s.read(1),s.nodata)
                    d=np.full((rows,cols),np.nan,"float32")
                    reproject(a.astype("float32"),d,src_transform=s.transform,src_crs=s.crs,
                              dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear)
                d[(d<0)|(d>5000)]=np.nan; return d
    return None

def s2_mosaic_ndsi(around_date, rows, cols, tr):
    """Mosaic all low-cloud S2 within +/-15 days of a date -> NDSI + green on grid."""
    d0=date.fromisoformat(around_date)
    dt=f"{(d0-timedelta(days=15)).isoformat()}/{(d0+timedelta(days=15)).isoformat()}"
    s=CAT.search(collections=['sentinel-2-l2a'],bbox=BBOX,datetime=dt,query={'eo:cloud_cover':{'lt':15}})
    items=list(s.items())
    if not items: return None,None
    # load a mosaic (odc picks least-cloudy per pixel via groupby='solar_day' + median)
    ds=odc.stac.load(items,bands=["B03","B11","SCL"],bbox=BBOX,resolution=RES,crs=crs,
                     groupby="solar_day",chunks={})
    g=ds["B03"].astype("float32"); sw=ds["B11"].astype("float32"); scl=ds["SCL"]
    ndsi=(g-sw)/(g+sw+1e-6)
    valid=~scl.isin([0,1,3,8,9,10])   # drop nodata/cloud/shadow
    ndsi=ndsi.where(valid); g=g.where(valid)
    # temporal median (cloud-free composite)
    ndsi_m=ndsi.median(dim="time",skipna=True).values
    g_m=g.median(dim="time",skipna=True).values
    # reproject to DEM grid
    def togrid(arr):
        out=np.full((rows,cols),np.nan,"float32")
        reproject(np.asarray(arr,dtype="float32"),out,src_transform=ds.odc.transform,src_crs=crs,
                  dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear)
        return out
    return togrid(ndsi_m), togrid(g_m)

results=[]
for m in matches:
    if not (m['s2_before'] and m['s2_after'] and m['dem_before'] and m['dem_after']): continue
    lab=m['eruption']; print(f"\n=== {lab} ===")
    # grid from the AFTER dem
    dfA=None
    for sd in glob.glob(f"{BASE}/{m['dem_after'][:4]}/{m['dem_after']}*{m['dem_after_track']}_0045*"):
        for t in ("DEM_FNL","DEM_VER"):
            g=glob.glob(f"{sd}/prc07/{t}_*.tif")
            if g: dfA=g[0]; break
    if not dfA: print("  no after-DEM"); continue
    with rasterio.open(dfA) as s:
        demA=clean_dem(s.read(1),s.nodata); tr=s.transform; rows,cols=s.shape; xb=s.bounds
    demB=dem_on_grid(m['dem_before'],m['dem_before_track'],rows,cols,tr)
    if demB is None: print("  no before-DEM"); continue
    shp=gpd.read_file(GLINV).to_crs(crs)
    gm=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
        out_shape=(rows,cols),transform=tr,fill=0,dtype="uint8").astype(bool)
    ndsiB,_=s2_mosaic_ndsi(m['s2_before'],rows,cols,tr)
    ndsiA,grA=s2_mosaic_ndsi(m['s2_after'],rows,cols,tr)
    if ndsiA is None or ndsiB is None: print("  S2 mosaic failed"); continue
    print(f"  S2 valid px: before={np.isfinite(ndsiB).sum():,} after={np.isfinite(ndsiA).sum():,}")
    # ash = glacier pixel that darkened (NDSI dropped) & is now non-snow
    dndsi=ndsiA-ndsiB
    ash=gm&np.isfinite(dndsi)&(dndsi<-0.25)&(ndsiA<0.4)
    # dH, robust debias + clip
    dh=demA-demB
    # HARD physical clip: eruptive deposition can be large, so allow +/-120 m
    # (removes only corrupt DEM pixels, not genuine ash/lava deposition at summit)
    dh=np.where(np.abs(dh)<=120, dh, np.nan)
    # de-bias on STABLE (off-glacier) terrain only
    stv=dh[(~gm)&np.isfinite(dh)]
    if stv.size>1000: dh=dh-np.nanmedian(stv)
    # robust NMAD clip is computed on the STABLE terrain (measures DEM noise),
    # NOT the glacier — otherwise genuine deposition on the glacier gets erased.
    if stv.size>1000:
        st_nmad=1.4826*np.nanmedian(np.abs(stv-np.nanmedian(stv)))
        # keep glacier pixels within a generous window; clip only clear DEM blunders
        keep = (~gm) | (np.abs(dh) <= max(60.0, 8*st_nmad))
        dh=np.where(keep, dh, np.nan)
    gv=dh[gm&np.isfinite(dh)]
    dh_ash=dh[ash&np.isfinite(dh)]; dh_cl=dh[gm&~ash&np.isfinite(dh)]
    mash=np.nanmedian(dh_ash) if dh_ash.size>20 else np.nan
    mcl =np.nanmedian(dh_cl) if dh_cl.size>20 else np.nan
    print(f"  ash px={ash.sum():,} ({100*ash.sum()/max(1,gm.sum()):.1f}% glacier)")
    print(f"  dH ash={mash:+.2f}m  dH clean={mcl:+.2f}m  -> ash {'HIGHER' if mash>mcl else 'lower'}")
    results.append((lab,ash.sum(),mash,mcl))
    # figure
    fig,ax=plt.subplots(1,3,figsize=(18,6))
    ax[0].imshow(ndsiA,cmap='RdBu',vmin=-0.5,vmax=1,extent=[xb.left,xb.right,xb.bottom,xb.top])
    ax[0].set_title(f"S2 NDSI after (~{m['s2_after']})\nblue=snow, red=ash/rock")
    ax[1].imshow(np.where(gm,0.5,np.nan),cmap='Greys',vmin=0,vmax=1,extent=[xb.left,xb.right,xb.bottom,xb.top])
    ax[1].imshow(np.where(ash,1,np.nan),cmap='autumn',vmin=0,vmax=1,extent=[xb.left,xb.right,xb.bottom,xb.top])
    ax[1].set_title(f"ash/lava on glacier (orange) — {ash.sum():,} px")
    vlim=max(2,np.nanpercentile(np.abs(gv),95)) if gv.size else 10
    im=ax[2].imshow(np.where(gm,dh,np.nan),cmap='RdBu',vmin=-vlim,vmax=vlim,extent=[xb.left,xb.right,xb.bottom,xb.top])
    ax[2].set_title(f"DEM dH {m['dem_before']}→{m['dem_after']}\nblue=gain,red=loss  (ash:{mash:+.1f} clean:{mcl:+.1f} m)")
    plt.colorbar(im,ax=ax[2],shrink=0.7,label="dH [m]")
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{lab} — Sentinel–2 ash mapping vs DEM height change",fontsize=14)
    safe=lab.replace(' ','_').replace('/','-')
    fig.savefig(AOUT/f"ASH_{safe}.png",dpi=130,bbox_inches="tight"); plt.close(fig)
    prof=dict(driver="GTiff",height=rows,width=cols,count=1,dtype="float32",crs=crs,transform=tr,nodata=-9999.0,compress="lzw")
    with rasterio.open(AOUT/f"ASH_mask_{safe}.tif","w",**prof) as d:
        d.write(np.where(ash,1.0,-9999.0).astype("float32"),1)
    print(f"  -> ASH_{safe}.png")
print("\n=== SUMMARY: dH on ash vs clean glacier ===")
for lab,n,ma,mc in results:
    print(f"  {lab}: ash dH={ma:+.2f}m vs clean={mc:+.2f}m ({n:,} ash px)")
