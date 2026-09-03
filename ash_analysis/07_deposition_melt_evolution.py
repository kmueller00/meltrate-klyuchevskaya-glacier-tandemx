#!/usr/bin/env python3
"""
STEP 7: 2023-24 PAROXYSM deposition -> albedo-melt TIME EVOLUTION.

Central hypothesis: eruptive deposition first ADDS mass / retains height on the
ash-covered zones (year 1), then the darkened low-albedo surface drives ENHANCED
melt so those same zones LOSE more height than clean ice in later years.

We fix the ash mask ONCE from the immediate post-paroxysm melt season (summer
2024, darkened + snow-lost vs summer 2023), then track dH of that SAME footprint
across three snow-free late-summer brackets:

  year 1 : 2023-09-29 -> 2024-09-04   (deposition year)
  year 2 : 2024-09-04 -> 2025-09-13   (albedo-melt year)
  cumul  : 2023-09-29 -> 2025-09-13   (net 2-year)

Reports ash vs clean dH for each, and the ash-relative anomaly (ash - clean).
If the hypothesis holds: anomaly is positive (retention) in year 1 and turns
negative (excess melt) in year 2.
"""
import sys; sys.path.insert(0,"/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/ash_analysis")
import glob, numpy as np, rasterio, rasterio.features, geopandas as gpd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datetime import date
from pathlib import Path
from rasterio.warp import reproject, Resampling
from scipy import ndimage
from s2_util import ndsi_bright as _ndsi_bright_shared

BASE=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation/ash_analysis")
GLINV=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/ash_analysis/GLINV_main_massif.gpkg")  # main massif only (southern tongue-only glaciers excluded)
crs="EPSG:32657"; RES=30.0
S23="2023-09-29"; S24="2024-09-04"; S25="2025-09-13"    # snow-free late-summer epochs
PAROXYSM_DATE=date(2023,11,1)

def dem_path(dstr):
    for sd in glob.glob(f"{BASE}/{dstr[:4]}/{dstr}*155_0045*"):
        for t in ("DEM_FNL","DEM_VER"):
            g=glob.glob(f"{sd}/prc07/{t}_*.tif")
            if g: return g[0]

# fix a reference grid from the earliest epoch DEM
fREF=dem_path(S23); assert fREF, "no 2023-09-29 DEM"
with rasterio.open(fREF) as s:
    tr=s.transform; rows,cols=s.shape; xb=s.bounds

def load_on_grid(dstr):
    f=dem_path(dstr); assert f, f"no DEM {dstr}"
    with rasterio.open(f) as s:
        a=s.read(1).astype(float); a[(a==s.nodata)|(a<0)|(a>5000)]=np.nan
        o=np.full((rows,cols),np.nan,'float32')
        reproject(a.astype('float32'),o,src_transform=s.transform,src_crs=s.crs,
                  dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear)
    o[(o<0)|(o>5000)]=np.nan; return o

# glacier mask
shp=gpd.read_file(GLINV).to_crs(crs)
gm=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
    out_shape=(rows,cols),transform=tr,fill=0,dtype='uint8').astype(bool)

# --- FIXED ash mask from immediate post-paroxysm darkening (summer24 vs summer23) ---
ndA,brA=_ndsi_bright_shared(S24,rows,cols,tr,crs,erupt_date=PAROXYSM_DATE,side="post",gm=gm,label="paroxysm post")
ndB,_  =_ndsi_bright_shared(S23,rows,cols,tr,crs,erupt_date=PAROXYSM_DATE,side="pre", gm=gm,label="paroxysm pre")
assert ndA is not None and ndB is not None, "S2 failed"
dndsi=ndA-ndB
br_thr=np.nanpercentile(brA[gm&np.isfinite(brA)],40)
ash=gm&np.isfinite(dndsi)&(dndsi<-0.2)&(ndA<0.3)&(brA<br_thr)
ash=ndimage.binary_opening(ash,iterations=1); ash=ndimage.binary_closing(ash,iterations=1)
print(f"FIXED ash mask (post-paroxysm 2024): {ash.sum():,} px ({100*ash.sum()/gm.sum():.1f}% glacier)")

def dh_stats(dstr_a,dstr_b):
    A=load_on_grid(dstr_b); B=load_on_grid(dstr_a)
    dh=A-B; dh=np.where(np.abs(dh)<=120,dh,np.nan)
    stv=dh[(~gm)&np.isfinite(dh)]
    if stv.size>1000: dh=dh-np.nanmedian(stv)
    da=dh[ash&np.isfinite(dh)]; dc=dh[gm&~ash&np.isfinite(dh)]
    ma=np.nanmedian(da) if da.size>20 else np.nan
    mc=np.nanmedian(dc) if dc.size>20 else np.nan
    return ma,mc,da,dc,dh

brackets=[("Year 1 (deposition)\n2023->2024",S23,S24),
          ("Year 2 (albedo-melt)\n2024->2025",S24,S25),
          ("Net 2-year\n2023->2025",S23,S25)]
rows_out=[]
for lab,a,b in brackets:
    ma,mc,da,dc,dh=dh_stats(a,b)
    rows_out.append((lab,ma,mc,da,dc))
    print(f"{lab.splitlines()[0]}: ash={ma:+.2f}m clean={mc:+.2f}m  anomaly(ash-clean)={ma-mc:+.2f}m "
          f"(ash n={da.size:,})")

# --- figure: 3 boxplots + anomaly bars ---
fig,axes=plt.subplots(1,4,figsize=(19,5.5),gridspec_kw={'width_ratios':[1,1,1,1.1]})
for ax,(lab,ma,mc,da,dc) in zip(axes[:3],rows_out):
    ax.boxplot([dc[np.isfinite(dc)],da[np.isfinite(da)]],labels=['clean','ash'],
               showfliers=False,widths=0.6,medianprops=dict(color='k',lw=2))
    ax.axhline(0,color='grey',ls=':'); ax.set_title(f"{lab}\nash {ma:+.1f}  clean {mc:+.1f} m",fontsize=10)
    ax.set_ylabel("dH [m]")
# anomaly panel
labs=[r[0].splitlines()[0] for r in rows_out]
anom=[r[1]-r[2] for r in rows_out]
cols_bar=['#2c7bb6' if v>0 else '#d7191c' for v in anom]
axes[3].bar(range(3),anom,color=cols_bar)
axes[3].axhline(0,color='k',lw=0.8)
axes[3].set_xticks(range(3)); axes[3].set_xticklabels(['Yr1','Yr2','2yr'])
axes[3].set_ylabel("ash – clean dH [m]")
axes[3].set_title("Ash–relative anomaly\n(+ = retention, – = excess melt)",fontsize=10)
for i,v in enumerate(anom): axes[3].text(i,v+(0.1 if v>=0 else -0.1),f"{v:+.1f}",ha='center',
                                         va='bottom' if v>=0 else 'top',fontsize=10,fontweight='bold')
fig.suptitle("Klyuchevskaya 2023–24 paroxysm: deposition → albedo–melt evolution on ash–covered glacier\n"
             "(same ash footprint tracked across snow–free late–summer DEMs)",fontsize=13)
fig.tight_layout()
fig.savefig(AOUT/"DEPOSITION_MELT_evolution.png",dpi=600,bbox_inches="tight")
print("-> DEPOSITION_MELT_evolution.png")
