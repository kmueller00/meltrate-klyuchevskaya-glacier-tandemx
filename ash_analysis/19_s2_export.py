#!/usr/bin/env python3
"""STEP 19 (v2): Cloud-free Sentinel-2 exports on the ASHMASK grid.
Per bracket: pre RGB, post RGB (uint8) + post NDSI (float32).
- windows NEVER cross the eruption date (pre strictly before, post strictly after)
- two-layer composite: NEAR window median first, gaps filled from WIDE window
- prints glacier-area valid fraction so cloud-free coverage is verified, not assumed
"""
import glob, numpy as np, rasterio, rasterio.features, geopandas as gpd
from datetime import date, timedelta
from pathlib import Path
from rasterio.warp import reproject, Resampling
import pystac_client, planetary_computer, odc.stac

BASE=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation/ash_analysis")
GLINV=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/ash_analysis/GLINV_main_massif.gpkg")
crs="EPSG:32657"; RES=30.0; BBOX=[159.889,55.472,161.359,56.522]
CAT=pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1',
                              modifier=planetary_computer.sign_inplace)
# label, preDEM, postDEM, eruption date
BR=[("2019_Oct","2019-09-27","2020-09-02",date(2019,10,25)),
    ("2020_Dec","2020-09-24","2021-08-31",date(2020,12,9)),
    ("2022_Nov","2022-10-01","2023-08-05",date(2022,11,20)),
    ("2023_PAROXYSM","2023-09-29","2024-09-04",date(2023,11,1))]

def dem_path(dstr):
    for sd in glob.glob(f"{BASE}/{dstr[:4]}/{dstr}*155_0045*"):
        for t in ("DEM_FNL","DEM_VER"):
            g=glob.glob(f"{sd}/prc07/{t}_*.tif")
            if g: return g[0]

def fetch(d0,d1,rows,cols,tr):
    if d0>=d1: return None
    it=list(CAT.search(collections=['sentinel-2-l2a'],bbox=BBOX,
        datetime=f"{d0.isoformat()}/{d1.isoformat()}",
        query={'eo:cloud_cover':{'lt':40}}).items())
    if not it: return None
    ds=odc.stac.load(it,bands=["B02","B03","B04","B11","SCL"],bbox=BBOX,resolution=RES,
                     crs=crs,groupby="solar_day",chunks={})
    valid=~ds["SCL"].isin([0,1,3,8,9,10])
    out={}
    for b in ("B02","B03","B04","B11"):
        m=ds[b].astype('float32').where(valid).median(dim="time",skipna=True).values
        o=np.full((rows,cols),np.nan,'float32')
        reproject(np.asarray(m,'float32'),o,src_transform=ds.odc.transform,src_crs=crs,
                  dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear)
        out[b]=o
    return out

def compose(near,wide):
    if near is None: return wide
    if wide is None: return near
    return {k:np.where(np.isfinite(near[k]),near[k],wide[k]) for k in near}

def w_rgb(fn,b,tr,rows,cols):
    rgb=np.stack([np.clip(b[k]/3000*255,0,255) for k in ("B04","B03","B02")])
    rgb=np.where(np.isfinite(rgb),rgb,0).astype('uint8')
    prof=dict(driver="GTiff",height=rows,width=cols,count=3,dtype="uint8",crs=crs,
              transform=tr,compress="lzw",photometric="RGB")
    with rasterio.open(fn,"w",**prof) as d: d.write(rgb)

shp=gpd.read_file(GLINV).to_crs(crs)
for lab,pre,post,ED in BR:
    fA=dem_path(post)
    if not fA: print(f"{lab}: no DEM"); continue
    with rasterio.open(fA) as s: tr=s.transform; rows,cols=s.shape
    gm=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
        out_shape=(rows,cols),transform=tr,fill=0,dtype='uint8').astype(bool)
    pd0=date.fromisoformat(pre); po0=date.fromisoformat(post)
    epochs={
      # pre: capped 1 day before eruption; near 30 d, wide 90 d back
      "pre":  ((pd0-timedelta(days=30), min(pd0+timedelta(days=20),ED-timedelta(days=1))),
               (pd0-timedelta(days=90), min(pd0+timedelta(days=20),ED-timedelta(days=1)))),
      # post: entirely after eruption (post anchor is next melt season); near ±25, wide ±60
      "post": ((max(po0-timedelta(days=25),ED+timedelta(days=1)), po0+timedelta(days=25)),
               (max(po0-timedelta(days=60),ED+timedelta(days=1)), po0+timedelta(days=60)))}
    for tag,(nw,ww) in epochs.items():
        near=fetch(*nw,rows,cols,tr); wide=fetch(*ww,rows,cols,tr)
        vf_n=np.isfinite(near["B03"])[gm].mean()*100 if near else 0
        b=compose(near,wide)
        if b is None: print(f"{lab} {tag}: NO DATA"); continue
        vf=np.isfinite(b["B03"])[gm].mean()*100
        w_rgb(AOUT/f"S2_{lab}_{tag}_RGB.tif",b,tr,rows,cols)
        print(f"{lab} {tag}: glacier valid {vf_n:.1f}% (near) -> {vf:.1f}% (filled)  "
              f"[{nw[0]}..{nw[1]}] -> S2_{lab}_{tag}_RGB.tif")
        if tag=="post":
            nd=(b["B03"]-b["B11"])/(b["B03"]+b["B11"]+1e-6)
            prof=dict(driver="GTiff",height=rows,width=cols,count=1,dtype="float32",crs=crs,
                      transform=tr,nodata=-9999.0,compress="lzw")
            with rasterio.open(AOUT/f"S2_{lab}_post_NDSI.tif","w",**prof) as d:
                d.write(np.where(np.isfinite(nd),nd,-9999.0).astype('float32'),1)
            print(f"  -> S2_{lab}_post_NDSI.tif")
