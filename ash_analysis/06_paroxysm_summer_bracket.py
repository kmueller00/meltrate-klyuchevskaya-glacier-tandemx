#!/usr/bin/env python3
"""
STEP 6: The 2023-24 PAROXYSM, summer-to-summer bracket.

The tight winter bracket (Oct2023->Jan2024) buries the deposition signal under
seasonal snow (both epochs snow-covered). The correct design for isolating
eruptive deposition + subsequent ash-driven melt is SUMMER->SUMMER: compare the
last snow-free late-summer DEM BEFORE the Nov-2023 paroxysm with the matching
late-summer DEM AFTER it. Both epochs are ablation-season (minimal snow at the
summit), so dH reflects deposition/lava (gain) and ash-albedo melt (loss),
not winter accumulation.

  pre : 2023-09-29 (late summer, before Nov-2023 paroxysm)
  post: 2024-09-04 (late summer, ~10 months after)   -> 1-year snow-free bracket

Sentinel-2 ash mask taken from the AFTER melt-season (2024) mosaic vs BEFORE
(2023), so "ash" = glacier that darkened & lost snow cover across the eruption.
"""
import glob, numpy as np, rasterio, rasterio.features, geopandas as gpd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datetime import date, timedelta
from pathlib import Path
from rasterio.warp import reproject, Resampling
import pystac_client, planetary_computer, odc.stac
from scipy import ndimage

BASE=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation/ash_analysis")
GLINV=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b/"
           "2025/2025-01-14_065954_TDT_SM_A_97516_155_0045+1-2/prc05_refdem/"
           "GLINV_rgi06_ttl_utm57_N_b_UTM.shp")
GLINV=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/ash_analysis/GLINV_main_massif.gpkg")  # OVERRIDE: main massif only (southern tongue-only glaciers excluded)
crs="EPSG:32657"; RES=30.0; BBOX=[159.889,55.472,161.359,56.522]
PRE="2023-09-29"; POST="2024-09-04"   # snow-free summer bracket around Nov-2023 paroxysm
CAT=pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1',
                              modifier=planetary_computer.sign_inplace)

def dem(dstr):
    for sd in glob.glob(f"{BASE}/{dstr[:4]}/{dstr}*155_0045*"):
        for t in ("DEM_FNL","DEM_VER"):
            g=glob.glob(f"{sd}/prc07/{t}_*.tif")
            if g: return g[0]

def ndsi_and_dark(around, rows, cols, tr):
    d0=date.fromisoformat(around)
    dt=f"{(d0-timedelta(days=20)).isoformat()}/{(d0+timedelta(days=20)).isoformat()}"
    it=list(CAT.search(collections=['sentinel-2-l2a'],bbox=BBOX,datetime=dt,
                       query={'eo:cloud_cover':{'lt':20}}).items())
    if not it: return None,None
    ds=odc.stac.load(it,bands=["B03","B04","B11","SCL"],bbox=BBOX,resolution=RES,crs=crs,
                     groupby="solar_day",chunks={})
    g=ds["B03"].astype('float32'); r=ds["B04"].astype('float32'); sw=ds["B11"].astype('float32'); scl=ds["SCL"]
    valid=~scl.isin([0,1,3,8,9,10])
    nd=((g-sw)/(g+sw+1e-6)).where(valid)
    bright=g.where(valid)                       # green reflectance ~ brightness (snow bright, ash dark)
    ndm=nd.median(dim="time",skipna=True).values
    brm=bright.median(dim="time",skipna=True).values
    def tg(a):
        o=np.full((rows,cols),np.nan,'float32')
        reproject(np.asarray(a,'float32'),o,src_transform=ds.odc.transform,src_crs=crs,
                  dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear); return o
    return tg(ndm),tg(brm)

# --- load summer DEM pair on the AFTER grid ---
fA=dem(POST); fB=dem(PRE)
assert fA and fB, f"missing DEM {PRE} or {POST}"
with rasterio.open(fA) as s:
    A=s.read(1).astype(float); A[(A==s.nodata)|(A<0)|(A>5000)]=np.nan
    tr=s.transform; rows,cols=s.shape; xb=s.bounds
with rasterio.open(fB) as s:
    b=s.read(1).astype(float); b[(b==s.nodata)|(b<0)|(b>5000)]=np.nan
    B=np.full((rows,cols),np.nan,'float32')
    reproject(b.astype('float32'),B,src_transform=s.transform,src_crs=s.crs,
              dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear)
dh=A-B
dh=np.where(np.abs(dh)<=120,dh,np.nan)          # physical clip only (keep deposition)

shp=gpd.read_file(GLINV).to_crs(crs)
gm=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
    out_shape=(rows,cols),transform=tr,fill=0,dtype='uint8').astype(bool)
# de-bias on stable terrain
stv=dh[(~gm)&np.isfinite(dh)]
if stv.size>1000: dh=dh-np.nanmedian(stv)

# --- ash mask: darkened + snow lost between summer 2023 and summer 2024 ---
ndA,brA=ndsi_and_dark(POST,rows,cols,tr)
ndB,brB=ndsi_and_dark(PRE,rows,cols,tr)
assert ndA is not None and ndB is not None, "S2 mosaic failed"
dndsi=ndA-ndB
br_thr=np.nanpercentile(brA[gm&np.isfinite(brA)],40)   # darker-than-median glacier
ash=gm&np.isfinite(dndsi)&(dndsi<-0.2)&(ndA<0.3)&(brA<br_thr)
ash=ndimage.binary_opening(ash,iterations=1)
ash=ndimage.binary_closing(ash,iterations=1)

dh_ash=dh[ash&np.isfinite(dh)]; dh_cl=dh[gm&~ash&np.isfinite(dh)]
mash=np.nanmedian(dh_ash) if dh_ash.size>20 else np.nan
mcl =np.nanmedian(dh_cl) if dh_cl.size>20 else np.nan
print(f"PAROXYSM summer bracket {PRE} -> {POST}")
print(f"  glacier valid dH px: {np.isfinite(dh[gm]).sum():,}")
print(f"  ash px={ash.sum():,} ({100*ash.sum()/max(1,gm.sum()):.1f}% glacier), with valid dH={ (ash&np.isfinite(dh)).sum():,}")
print(f"  dH ASH  = {mash:+.2f} m   (n={dh_ash.size:,})")
print(f"  dH CLEAN= {mcl:+.2f} m   (n={dh_cl.size:,})")
print(f"  --> ash-relative signal = {mash-mcl:+.2f} m")

# --- figure: dH map + ash contour + boxplot ---
fig=plt.figure(figsize=(15,6))
gs=fig.add_gridspec(1,3,width_ratios=[1.2,1.2,0.9])
ax0=fig.add_subplot(gs[0]); ax1=fig.add_subplot(gs[1]); ax2=fig.add_subplot(gs[2])
ext=[xb.left,xb.right,xb.bottom,xb.top]
ax0.imshow(ndA,cmap='RdBu',vmin=-0.5,vmax=1,extent=ext); ax0.set_title(f"S2 NDSI summer 2024\nblue=snow  red=ash/rock")
if ash.any(): ax0.contour(np.flipud(ash.astype(float)),levels=[0.5],colors='lime',linewidths=1.2,extent=ext)
vlim=max(3,np.nanpercentile(np.abs(dh[gm&np.isfinite(dh)]),95))
im=ax1.imshow(np.where(gm,dh,np.nan),cmap='RdBu_r',vmin=-vlim,vmax=vlim,extent=ext)
if ash.any(): ax1.contour(np.flipud(ash.astype(float)),levels=[0.5],colors='lime',linewidths=1.2,extent=ext)
ax1.set_title(f"DEM dH {PRE}→{POST}\nred=gain(deposition)  blue=loss(melt)")
plt.colorbar(im,ax=ax1,shrink=0.7,label="dH [m]")
bp=[dh_cl[np.isfinite(dh_cl)],dh_ash[np.isfinite(dh_ash)]]
ax2.boxplot(bp,labels=['clean\nglacier','ash-\ncovered'],showfliers=False,widths=0.6,
            medianprops=dict(color='k',lw=2),boxprops=dict(lw=1.3))
ax2.axhline(0,color='grey',ls=':'); ax2.set_ylabel("dH [m]")
ax2.set_title(f"ash {mash:+.1f} m  vs  clean {mcl:+.1f} m")
for a in (ax0,ax1): a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"2023–24 Klyuchevskaya PAROXYSM — summer–to–summer deposition/melt\n"
             f"ash–covered glacier vs clean glacier elevation change",fontsize=13)
fig.savefig(AOUT/"PAROXYSM_summer_bracket.png",dpi=600,bbox_inches="tight")
print(f"  -> PAROXYSM_summer_bracket.png")
