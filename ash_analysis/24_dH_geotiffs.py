#!/usr/bin/env python3
import sys; sys.path.insert(0,"/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/ash_analysis"); import _style
"""STEP 24: Export the per-eruption dH (elevation-change) rasters as GeoTIFFs.
One float32 GeoTIFF per eruption bracket, glacier-masked, de-biased, on the same
grid as the ASHMASK/S2 tifs (pixel-perfect stack in QGIS). nodata=-9999."""
import glob, numpy as np, rasterio, rasterio.features, geopandas as gpd
from pathlib import Path
from rasterio.warp import reproject, Resampling

BASE=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation/ash_analysis")
GLINV=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/ash_analysis/GLINV_main_massif.gpkg")
crs="EPSG:32657"

BRACKETS=[("2019_Oct","2019-09-27","2020-09-02"),("2020_Dec","2020-09-24","2021-08-31"),
          ("2022_Nov","2022-10-01","2023-08-05"),("2023_PAROXYSM","2023-09-29","2024-09-04")]

def dem_path(dstr):
    for sd in glob.glob(f"{BASE}/{dstr[:4]}/{dstr}*155_0045*"):
        for t in ("DEM_FNL","DEM_VER"):
            g=glob.glob(f"{sd}/prc07/{t}_*.tif")
            if g: return g[0]

shp=gpd.read_file(GLINV).to_crs(crs)
for lab,pre,post in BRACKETS:
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
    out=np.where(gm & np.isfinite(dh), dh, -9999.0).astype('float32')
    prof=dict(driver="GTiff",height=rows,width=cols,count=1,dtype="float32",crs=crs,
              transform=tr,nodata=-9999.0,compress="lzw")
    fn=AOUT/f"DH_{lab}.tif"
    with rasterio.open(fn,"w",**prof) as d: d.write(out,1)
    valid=(out!=-9999.0).sum()
    print(f"{lab}: dH raster, glacier-masked, {valid:,} valid px -> {fn.name}")
