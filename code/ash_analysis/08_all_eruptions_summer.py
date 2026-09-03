#!/usr/bin/env python3
"""
STEP 8: Ash-deposition/melt for ALL eruptions using SUMMER-to-SUMMER brackets.

The winter/tight brackets fail: winter S2 can't see ash under snow, and tight DEM
pairs carry little signal. The robust design (validated on the 2023 paroxysm) is
snow-free late-summer -> next late-summer, with the ash mask from the post-eruption
melt-season S2 (darkened + snow-lost vs pre-eruption summer).

For each eruption we pick the last late-summer DEM BEFORE and the first late-summer
DEM in the melt season AFTER, and measure dH(ash) vs dH(clean).

S2 exists from 2016; the 2015 eruption uses Landsat 8 OLI instead (source="landsat"
in BRACKETS). CAVEAT: Landsat 8's 16-day revisit + this site's persistent volcanic
cloud cover left the standard +/-60d wide-window fallback at only 2.9%/25.1%
glacier coverage for the pre/post sides -- far thinner than any Sentinel-2-based
eruption here. Widened to wide_days=150 for this row only (see BRACKETS), which
brings coverage to 63.9%/78.8% and raises the ash/clean sample from ~1.7% to ~50%
of the glacier. This means the 2015 composite pulls in imagery up to ~5 months from
the anchor date rather than the ~2 months used everywhere else -- a real, wider
temporal smear than the other eruptions get, noted here and in the report rather
than silently matching the tighter default.
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

# eruption -> (pre-summer DEM, post-summer DEM, has_ash_source, eruption date, source, wide_days)
BRACKETS=[
    ("2015 Aug-Sep",   "2015-08-23","2016-08-31", True,  date(2015,8,27),  "landsat", 150),  # pre-S2; wide_days widened from default 60 -- see module docstring caveat
    ("2019 Oct",       "2019-09-27","2020-09-02", True,  date(2019,10,25), "s2",      60),
    ("2020 Dec",       "2020-09-24","2021-08-31", True,  date(2020,12,9),  "s2",      60),
    ("2022 Nov",       "2022-10-01","2023-08-05", True,  date(2022,11,20), "s2",      60),  # 09-20 DEM corrupt (garbage tails) -> use 10-01
    ("2023 PAROXYSM",  "2023-09-29","2024-09-04", True,  date(2023,11,1),  "s2",      60),
]

def dem_path(dstr):
    for sd in glob.glob(f"{BASE}/{dstr[:4]}/{dstr}*155_0045*"):
        for t in ("DEM_FNL","DEM_VER"):
            g=glob.glob(f"{sd}/prc07/{t}_*.tif")
            if g: return g[0]

shp=gpd.read_file(GLINV).to_crs(crs)
results=[]
for lab,pre,post,hasS2,ED,src,wd in BRACKETS:
    fA=dem_path(post); fB=dem_path(pre)
    if not (fA and fB):
        print(f"{lab}: missing DEM ({pre} or {post})"); continue
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
    if hasS2:
        ndA,brA=ndsi_bright(post,rows,cols,tr,crs,erupt_date=ED,side="post",gm=gm,label=f"{lab} post",source=src,wide_days=wd)
        ndB,_  =ndsi_bright(pre, rows,cols,tr,crs,erupt_date=ED,side="pre", gm=gm,label=f"{lab} pre", source=src,wide_days=wd)
        if ndA is None or ndB is None:
            print(f"{lab}: {src} failed -> DEM-only"); hasS2=False
    if hasS2:
        dn=ndA-ndB; br_thr=np.nanpercentile(brA[gm&np.isfinite(brA)],40)
        ash=gm&np.isfinite(dn)&(dn<-0.2)&(ndA<0.3)&(brA<br_thr)
        ash=ndimage.binary_opening(ash); ash=ndimage.binary_closing(ash)
        da=dh[ash&np.isfinite(dh)]; dc=dh[gm&~ash&np.isfinite(dh)]
        ma=np.nanmedian(da) if da.size>20 else np.nan
        mc=np.nanmedian(dc) if dc.size>20 else np.nan
        print(f"{lab} [{pre}->{post}]: ash={ma:+.2f}m clean={mc:+.2f}m anomaly={ma-mc:+.2f}m "
              f"(ash {ash.sum():,}px, {100*ash.sum()/gm.sum():.1f}%)")
        results.append((lab,ma,mc,ash.sum(),da,dc))
    else:
        gv=dh[gm&np.isfinite(dh)]; mg=np.nanmedian(gv)
        print(f"{lab} [{pre}->{post}] DEM-only: glacier median dH={mg:+.2f}m (no ash mask, pre-S2)")
        results.append((lab,np.nan,mg,0,None,gv))

# summary bar: ash-relative anomaly per eruption (S2 ones)
s2res=[r for r in results if r[3]>0]
if s2res:
    fig,ax=plt.subplots(figsize=(9,5))
    labs=[r[0] for r in s2res]; anom=[r[1]-r[2] for r in s2res]
    cols_bar=['#2c7bb6' if v>0 else '#d7191c' for v in anom]
    ax.bar(range(len(anom)),anom,color=cols_bar)
    ax.axhline(0,color='k',lw=0.8)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs,rotation=20,ha='right')
    ax.set_ylabel("ash – clean dH  [m]")
    ax.set_title("Ash–relative elevation anomaly, summer–to–summer bracket\n"
                 "(+ = deposition/retention, − = ash–driven excess melt)")
    for i,v in enumerate(anom): ax.text(i,v+(0.05 if v>=0 else -0.05),f"{v:+.2f}",
        ha='center',va='bottom' if v>=0 else 'top',fontweight='bold')
    fig.tight_layout(); fig.savefig(AOUT/"ALL_ERUPTIONS_summer_anomaly.png",dpi=600,bbox_inches="tight")
    print("-> ALL_ERUPTIONS_summer_anomaly.png")
