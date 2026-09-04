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
the most area of any acquisition on this track.

DEBIASING (added after visible seams were spotted in downstream figures):
the 7 scenes span 12 years, and the glacier itself has genuinely thinned
over that time (that's the whole point of this project) -- naively
median-compositing raw, un-debiased elevations produces a visible step
wherever the SET of contributing scenes changes across the grid, since
each combination has a different true multi-year elevation. Fixed by
debiasing every scene to the single largest-footprint scene first: for
each other scene, compute its median offset on STABLE (off-glacier, per
GLINV) terrain within the overlap, then subtract that constant offset
before compositing -- the same stable-terrain debiasing convention already
used throughout this pipeline (dH_robust_all.py, script 32, etc.), applied
here to remove real inter-annual change rather than to measure it. Combined
via per-pixel median (not mean) so a single scene's edge artifacts don't
bias the result.

Re-run this only if the archive changes materially (a much larger-footprint
scene lands) -- the output is stable otherwise and does not need refreshing
alongside routine reprocessing.
"""
import glob, re, numpy as np, rasterio, rasterio.features, geopandas as gpd
from collections import defaultdict
from pathlib import Path
from rasterio.warp import reproject, Resampling
import warnings; warnings.filterwarnings("ignore")

AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation")
BASE=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
GLINV=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/code/ash_analysis/GLINV_main_massif.gpkg")
RATE_TIF=AOUT/"summer_trend_rate_glacieronly.tif"
N_SCENES=7
crs="EPSG:32657"

with rasterio.open(RATE_TIF) as s:
    tr=s.transform; rows,cols=s.shape

shp=gpd.read_file(GLINV).to_crs(crs)
gm=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
    out_shape=(rows,cols),transform=tr,fill=0,dtype="uint8").astype(bool)
st=~gm   # stable (off-glacier) terrain, for debiasing only -- not assumed elevation-static on-glacier

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
print(f"compositing {len(top)} largest-footprint scenes (reference = largest):")
for f in top: print(" ", f)

def load_on_grid(f):
    with rasterio.open(f) as s:
        o=np.full((rows,cols),np.nan,"float32")
        a=s.read(1).astype("float32"); a[(a==s.nodata)|(a<0)|(a>5000)]=np.nan
        reproject(a,o,src_transform=s.transform,src_crs=s.crs,dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear)
    return o

ref=load_on_grid(top[0])
stack=[ref]
for f in top[1:]:
    a=load_on_grid(f)
    ok=st&np.isfinite(a)&np.isfinite(ref)
    bias=float(np.nanmedian((a-ref)[ok])) if ok.sum()>500 else 0.0
    print(f"  debias {f.split('/')[-1][:20]}: stable-terrain offset {bias:+.2f}m (n={ok.sum():,}), removed")
    stack.append(a-bias)
final=np.nanmedian(np.stack(stack),axis=0)
n_valid=int(np.isfinite(final).sum())
print(f"final coverage: {n_valid:,} / {rows*cols:,} = {100*n_valid/(rows*cols):.1f}%")

prof=dict(driver="GTiff",height=rows,width=cols,count=1,dtype="float32",crs=crs,transform=tr,nodata=-9999,compress="lzw")
out=np.where(np.isfinite(final),final,-9999).astype("float32")
with rasterio.open(AOUT/"FULL_GRID_elevation_mosaic.tif","w",**prof) as d:
    d.write(out,1)
print("-> FULL_GRID_elevation_mosaic.tif")
