#!/usr/bin/env python3
"""
Builds FULL_GRID_elevation_mosaic.tif: a 7-scene median elevation composite
covering 98.3% of the analysis grid (vs 63.5% from any single reference
scene). Consumed by scripts 36 and 37, which need elevation over bare/
unglaciated terrain (the summit, the Apakhonchich flank) that a
glacier-masked composite (script 32's summer_trend_elevation_mean.tif)
deliberately excludes.

Scene selection: the 7 largest-valid-pixel-footprint summer (May-Oct) scenes
across the whole archive (by best_scene() per date), which happen to span
2013-2025 and include the two-subframe 155_0045+1-2 mosaic dates that cover
the most area of any acquisition on this track. Combined via per-pixel
median (not mean) so a single scene's edge artifacts don't bias the result.

Re-run this only if the archive changes materially (a much larger-footprint
scene lands) -- the output is stable otherwise and does not need refreshing
alongside routine reprocessing.
"""
import glob, re, numpy as np, rasterio
from collections import defaultdict
from pathlib import Path
from rasterio.warp import reproject, Resampling
import warnings; warnings.filterwarnings("ignore")

AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation")
BASE=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
RATE_TIF=AOUT/"summer_trend_rate_glacieronly.tif"
N_SCENES=7

with rasterio.open(RATE_TIF) as s:
    tr=s.transform; rows,cols=s.shape; crs=s.crs

files=glob.glob(str(BASE/"*/*155_0045*/prc07/DEM_FNL_*.tif"))
by_date=defaultdict(list)
for f in files:
    m=re.search(r"(\d{4})-(\d{2})-(\d{2})",f)
    if not m: continue
    y,mo,d=m.groups()
    if 5<=int(mo)<=10: by_date[(y,mo,d)].append(f)

candidates=[]
for fl in by_date.values():
    best=None; bn=-1
    for f in fl:
        with rasterio.open(f) as s:
            a=s.read(1); n=int((a!=s.nodata).sum())
        if n>bn: bn=n; best=f
    candidates.append((bn,best))
candidates.sort(key=lambda x:-x[0])
top=[f for _,f in candidates[:N_SCENES]]
print(f"compositing {len(top)} largest-footprint scenes:")
for f in top: print(" ", f)

stack=[]
for f in top:
    with rasterio.open(f) as s:
        o=np.full((rows,cols),np.nan,"float32")
        a=s.read(1).astype("float32"); a[(a==s.nodata)|(a<0)|(a>5000)]=np.nan
        reproject(a,o,src_transform=s.transform,src_crs=s.crs,dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear)
    stack.append(o)
final=np.nanmedian(np.stack(stack),axis=0)
n_valid=int(np.isfinite(final).sum())
print(f"final coverage: {n_valid:,} / {rows*cols:,} = {100*n_valid/(rows*cols):.1f}%")

prof=dict(driver="GTiff",height=rows,width=cols,count=1,dtype="float32",crs=crs,transform=tr,nodata=-9999,compress="lzw")
out=np.where(np.isfinite(final),final,-9999).astype("float32")
with rasterio.open(AOUT/"FULL_GRID_elevation_mosaic.tif","w",**prof) as d:
    d.write(out,1)
print("-> FULL_GRID_elevation_mosaic.tif")
