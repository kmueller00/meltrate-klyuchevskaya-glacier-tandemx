#!/usr/bin/env python3
"""
STEP 39: Full-archive acquisition inventory -- month x year heatmap and
per-year totals, quality-gated the same way script 15's glacier-median
series is (readable pixel count + plausible median elevation), but WITHOUT
that script's shoulder-month (May/Oct) exclusion or season-outlier
rejection -- this is meant as an honest "what raw, usable data do we
actually have" picture, not the curated melt-trend subset. Scans every
DEM_FNL/DEM_VER folder on track 155_0045 directly.

Outputs: ACQUISITION_heatmap.png, ACQUISITION_inventory.csv
"""
import glob, re, csv, numpy as np, rasterio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from datetime import date
import warnings; warnings.filterwarnings("ignore")

AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation")
BASE=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")

def find_dem(sd):
    for t in ("DEM_FNL","DEM_VER"):
        g=glob.glob(str(sd/"prc07"/f"{t}_*.tif"))
        if g: return g[0]
    return None

dates=set()
for sd in [Path(p) for p in glob.glob(str(BASE/"20*/20*"))]:
    m=re.match(r"(\d{4})-(\d{2})-(\d{2})",sd.name)
    if not m or "155_0045" not in sd.name: continue
    d=date(*map(int,m.groups()))
    f=find_dem(sd)
    if not f: continue
    try:
        with rasterio.open(f) as s:
            a=s.read(1).astype("float32"); a[(a==s.nodata)|(a<0)|(a>5000)]=np.nan
            v=a[np.isfinite(a)]
    except Exception:
        continue
    if v.size<3e5 or not (800<np.nanmedian(v)<2500): continue
    dates.add(d)   # set: multiple subframes on the same date count once

years=sorted(set(d.year for d in dates))
counts=defaultdict(int)   # (year,month)->n
for d in dates: counts[(d.year,d.month)]+=1
grid=np.array([[counts[(y,m)] for m in range(1,13)] for y in years])
per_year=grid.sum(axis=1)

print(f"total usable quality-gated dates 2012-2025 (track 155_0045): {len(dates)}")
for y,row,tot in zip(years,grid,per_year):
    print(f"  {y}: {tot:2d}  "+" ".join(f"{v:2d}" for v in row))

with open(AOUT/"ACQUISITION_inventory.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["year"]+[f"month_{m:02d}" for m in range(1,13)]+["total"])
    for y,row,tot in zip(years,grid,per_year): w.writerow([y]+list(row)+[tot])
print("-> ACQUISITION_inventory.csv")

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,7),gridspec_kw={"width_ratios":[2,1]})
im=ax1.imshow(grid,cmap="YlOrRd",aspect="auto",vmin=0)
ax1.set_xticks(range(12)); ax1.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
ax1.set_yticks(range(len(years))); ax1.set_yticklabels(years)
for i in range(len(years)):
    for j in range(12):
        if grid[i,j]>0:
            ax1.text(j,i,str(grid[i,j]),ha="center",va="center",
                     color="white" if grid[i,j]>grid.max()*0.6 else "black",fontsize=9)
ax1.set_title("Quality-gated acquisitions per month, per year\n(track 155_0045; either subframe counts once per date)")
plt.colorbar(im,ax=ax1,label="# dates",fraction=0.046)

ax2.barh(years,per_year,color="#c0463a")
for y,tot in zip(years,per_year): ax2.text(tot+0.3,y,str(tot),va="center",fontsize=9)
ax2.set_yticks(years); ax2.invert_yaxis()
ax2.set_xlabel("# dates"); ax2.set_title("Total per year")
ax2.grid(alpha=0.3,axis="x")

fig.suptitle("Klyuchevskaya TanDEM-X archive: acquisition inventory",fontsize=13)
fig.tight_layout()
fig.savefig(AOUT/"ACQUISITION_heatmap.png",dpi=300,bbox_inches="tight")
print("-> ACQUISITION_heatmap.png")
