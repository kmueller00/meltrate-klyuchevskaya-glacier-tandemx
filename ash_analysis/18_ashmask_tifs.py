#!/usr/bin/env python3
"""STEP 18: Export the ash masks as GeoTIFFs for visual QC in a GIS.
For each eruption bracket, saves strict (production) and moderate masks:
1 = ash, 0 = clean glacier, nodata(255) = off-glacier. EPSG:32657, 30 m."""
import sys; sys.path.insert(0,"/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/ash_analysis")
import glob, numpy as np, rasterio, rasterio.features, geopandas as gpd
from datetime import date
from pathlib import Path
from scipy import ndimage
from s2_util import ndsi_bright

BASE=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation/ash_analysis")
GLINV=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/ash_analysis/GLINV_main_massif.gpkg")
crs="EPSG:32657"; RES=30.0
BRACKETS=[("2019_Oct","2019-09-27","2020-09-02",date(2019,10,25)),
          ("2020_Dec","2020-09-24","2021-08-31",date(2020,12,9)),
          ("2022_Nov","2022-10-01","2023-08-05",date(2022,11,20)),
          ("2023_PAROXYSM","2023-09-29","2024-09-04",date(2023,11,1))]
SET={"strict":(-0.20,0.30,40,"open+close"),"moderate":(-0.15,0.40,50,"close")}

def dem_path(dstr):
    for sd in glob.glob(f"{BASE}/{dstr[:4]}/{dstr}*155_0045*"):
        for t in ("DEM_FNL","DEM_VER"):
            g=glob.glob(f"{sd}/prc07/{t}_*.tif")
            if g: return g[0]

shp=gpd.read_file(GLINV).to_crs(crs)
for lab,pre,post,ED in BRACKETS:
    fA=dem_path(post)
    if not fA: print(f"{lab}: no DEM"); continue
    with rasterio.open(fA) as s: tr=s.transform; rows,cols=s.shape
    gm=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
        out_shape=(rows,cols),transform=tr,fill=0,dtype='uint8').astype(bool)
    ndA,brA=ndsi_bright(post,rows,cols,tr,crs,erupt_date=ED,side="post",gm=gm,label=f"{lab} post")
    ndB,_  =ndsi_bright(pre, rows,cols,tr,crs,erupt_date=ED,side="pre", gm=gm,label=f"{lab} pre")
    if ndA is None or ndB is None: print(f"{lab}: S2 fail"); continue
    dn=ndA-ndB
    for nm,(t_dn,t_na,pct,morph) in SET.items():
        br_thr=np.nanpercentile(brA[gm&np.isfinite(brA)],pct)
        m=gm&np.isfinite(dn)&(dn<t_dn)&(ndA<t_na)&(brA<br_thr)
        if morph=="open+close": m=ndimage.binary_closing(ndimage.binary_opening(m))
        else: m=ndimage.binary_closing(m)
        out=np.where(gm,m.astype('uint8'),255).astype('uint8')
        prof=dict(driver="GTiff",height=rows,width=cols,count=1,dtype="uint8",crs=crs,
                  transform=tr,nodata=255,compress="lzw")
        fn=AOUT/f"ASHMASK_{lab}_{nm}.tif"
        with rasterio.open(fn,"w",**prof) as d: d.write(out,1)
        print(f"{lab} {nm}: {m.sum():,} px ({m.sum()*0.0009:.1f} km2) -> {fn.name}")
