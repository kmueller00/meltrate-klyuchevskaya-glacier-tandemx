#!/usr/bin/env python3
"""
STEP 5: Refined ash detection + clean ash-on-DEM overlay for the deposition test.

Ash detection (improved): a glacier pixel is ash/lava-covered if AFTER the eruption
it shows BOTH (a) a strong NDSI drop vs before (snow -> dark), AND (b) low absolute
NDSI + low visible brightness (dark surface). Morphological cleanup removes speckle.

Overlay: ash extent (contour) drawn on the DEM dH map, so you can SEE whether ash
areas coincide with height GAIN (deposition). Also prints a stats table + boxplot.
"""
import json, glob, re
import numpy as np, rasterio, rasterio.features, geopandas as gpd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage
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

def clean(a,nd): a=a.astype(float); a[(a==nd)|(a<0)|(a>5000)]=np.nan; return a
def dem_grid(dstr,track,rows,cols,tr):
    for sd in glob.glob(f"{BASE}/{dstr[:4]}/{dstr}*{track}_0045*"):
        for t in ("DEM_FNL","DEM_VER"):
            g=glob.glob(f"{sd}/prc07/{t}_*.tif")
            if g:
                with rasterio.open(g[0]) as s:
                    a=clean(s.read(1),s.nodata); d=np.full((rows,cols),np.nan,"float32")
                    reproject(a.astype("float32"),d,src_transform=s.transform,src_crs=s.crs,
                              dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear)
                d[(d<0)|(d>5000)]=np.nan; return d

def s2(around, rows, cols, tr):
    d0=date.fromisoformat(around)
    dt=f"{(d0-timedelta(days=15)).isoformat()}/{(d0+timedelta(days=15)).isoformat()}"
    items=list(CAT.search(collections=['sentinel-2-l2a'],bbox=BBOX,datetime=dt,
                          query={'eo:cloud_cover':{'lt':20}}).items())
    if not items: return None
    ds=odc.stac.load(items,bands=["B03","B04","B11","SCL"],bbox=BBOX,resolution=RES,crs=crs,
                     groupby="solar_day",chunks={})
    valid=~ds["SCL"].isin([0,1,3,8,9,10])
    g=ds["B03"].astype("float32").where(valid); r=ds["B04"].astype("float32").where(valid)
    sw=ds["B11"].astype("float32").where(valid)
    ndsi=((g-sw)/(g+sw+1e-6)).median("time",skipna=True).values
    bright=((g+r)/2).median("time",skipna=True).values
    def tg(x):
        o=np.full((rows,cols),np.nan,"float32")
        reproject(np.asarray(x,"float32"),o,src_transform=ds.odc.transform,src_crs=crs,
                  dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear); return o
    return tg(ndsi),tg(bright)

shp=gpd.read_file(GLINV).to_crs(crs)
stats=[]
for m in matches:
    if not (m['s2_before'] and m['s2_after'] and m['dem_before'] and m['dem_after']): continue
    lab=m['eruption']; print(f"\n=== {lab} ===")
    dfA=None
    for sd in glob.glob(f"{BASE}/{m['dem_after'][:4]}/{m['dem_after']}*{m['dem_after_track']}_0045*"):
        for t in ("DEM_FNL","DEM_VER"):
            g=glob.glob(f"{sd}/prc07/{t}_*.tif")
            if g: dfA=g[0]; break
    if not dfA: continue
    with rasterio.open(dfA) as s:
        demA=clean(s.read(1),s.nodata); tr=s.transform; rows,cols=s.shape; xb=s.bounds
    demB=dem_grid(m['dem_before'],m['dem_before_track'],rows,cols,tr)
    if demB is None: continue
    gm=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
        out_shape=(rows,cols),transform=tr,fill=0,dtype="uint8").astype(bool)
    sb=s2(m['s2_before'],rows,cols,tr); sa=s2(m['s2_after'],rows,cols,tr)
    if not sb or not sa: print("  S2 fail"); continue
    ndsiB,_=sb; ndsiA,brightA=sa
    # REFINED ash: NDSI dropped strongly AND now dark (low NDSI) AND darker than typical snow brightness
    dndsi=ndsiA-ndsiB
    br_thresh=np.nanpercentile(brightA[gm&np.isfinite(brightA)],40) if (gm&np.isfinite(brightA)).any() else 3000
    ash = gm & np.isfinite(dndsi) & (dndsi<-0.2) & (ndsiA<0.3) & (brightA<br_thresh)
    # morphological cleanup: remove isolated speckle, fill small holes
    ash=ndimage.binary_opening(ash,iterations=1)
    ash=ndimage.binary_closing(ash,iterations=1)
    # dH robust
    dh=demA-demB; dh=np.where(np.abs(dh)<=50,dh,np.nan)
    stv=dh[(~gm)&np.isfinite(dh)]
    if stv.size>1000: dh=dh-np.nanmedian(stv)
    dh_ash=dh[ash&np.isfinite(dh)]; dh_cl=dh[gm&~ash&np.isfinite(dh)]
    ma=np.nanmedian(dh_ash) if dh_ash.size>20 else np.nan
    mc=np.nanmedian(dh_cl) if dh_cl.size>20 else np.nan
    stats.append((lab,ash.sum(),ma,mc,dh_ash,dh_cl))
    print(f"  ash px={ash.sum():,}  dH ash={ma:+.2f}m  clean={mc:+.2f}m  diff={ma-mc:+.2f}m")
    # OVERLAY figure: dH map with ash contour
    fig,ax=plt.subplots(1,2,figsize=(15,7))
    gv=dh[gm&np.isfinite(dh)]; vlim=max(2,np.nanpercentile(np.abs(gv),95)) if gv.size else 10
    im=ax[0].imshow(np.where(gm,dh,np.nan),cmap='RdBu',vmin=-vlim,vmax=vlim,extent=[xb.left,xb.right,xb.bottom,xb.top])
    if ash.sum()>10:
        ax[0].contour(np.flipud(ash.astype(float)),levels=[0.5],colors='lime',linewidths=1.5,
                      extent=[xb.left,xb.right,xb.bottom,xb.top])
    ax[0].set_title(f"DEM dH {m['dem_before']}→{m['dem_after']}\ngreen=ash outline (S2)  blue=gain,red=loss")
    plt.colorbar(im,ax=ax[0],shrink=0.7,label="dH [m]")
    ax[0].set_xticks([]);ax[0].set_yticks([])
    # boxplot ash vs clean
    data=[dh_cl[np.isfinite(dh_cl)],dh_ash[np.isfinite(dh_ash)]]
    bp=ax[1].boxplot(data,tick_labels=['clean glacier','ash-covered'],showfliers=False,patch_artist=True)
    for p,c in zip(bp['boxes'],['#1f77b4','saddlebrown']): p.set_facecolor(c);p.set_alpha(0.6)
    ax[1].axhline(0,color='k',lw=0.6,ls=':'); ax[1].set_ylabel("dH [m]")
    ax[1].set_title(f"elevation change: ash vs clean\n(ash {ma:+.1f}m vs clean {mc:+.1f}m)")
    fig.suptitle(f"{lab} — ash extent vs DEM height change",fontsize=14)
    safe=lab.replace(' ','_').replace('/','-')
    fig.savefig(AOUT/f"OVERLAY_{safe}.png",dpi=130,bbox_inches="tight"); plt.close(fig)
    print(f"  -> OVERLAY_{safe}.png")
print("\n=== SUMMARY: deposition test (ash should be HIGHER than clean) ===")
for lab,n,ma,mc,_,_ in stats:
    sig="✓ DEPOSITION" if (np.isfinite(ma) and np.isfinite(mc) and ma>mc+1) else ""
    print(f"  {lab}: ash {ma:+.2f}m vs clean {mc:+.2f}m  (n={n:,})  {sig}")
