#!/usr/bin/env python3
import sys; sys.path.insert(0,"/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/code/ash_analysis"); import _style
"""
STEP 26: Co-registration OFFSET diagnostics.

User's observation: the Nuth & Kaab shift (dx, dy) applied to align each DEM pair
to the reference seems to cluster around a similar value for MOST pairs, but a
few pairs show a notably different offset. This script:

  (1) Re-runs Nuth-Kaab coregistration for every consecutive summer-pair used in
      the annual-rate series, but this time LOGS the fitted (dx, dy, dz) shift
      instead of silently applying and discarding it.
  (2) Also fits coregistration TWICE per pair, with two different stable-terrain
      masks (see step 27 for the slope-filtered mask), to see if slope selection
      itself changes the recovered offset.
  (3) Plots dx vs dy per pair (labelled by year), highlighting outliers from the
      main cluster (>2 MAD from the median offset) -- if most pairs cluster near
      one (dx,dy) and a few don't, those are the ones to look at for a reference-
      scene problem (mis-registered orbit state vector, wrong baseline, etc).
  (4) Reports whether the OUTLIER pairs share a common reference epoch (i.e. is
      it the reference DEM that is offset, or the individual non-reference DEM).
"""
import glob, re, numpy as np, rasterio, rasterio.features, geopandas as gpd, xdem
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import csv, warnings
warnings.filterwarnings("ignore")

BASE=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation")
GLINV=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/code/ash_analysis/GLINV_main_massif.gpkg")
crs="EPSG:32657"; RES=30.0

def find(sd):
    for t in ("DEM_FNL","DEM_VER"):
        g=glob.glob(str(sd/"prc07"/f"{t}_*.tif"))
        if g: return g[0]

def best_scene(files):
    """Pick the scene with the LARGEST valid-pixel footprint, not an arbitrary
    index -- a naive index pick can land on a partial/secondary sub-frame from
    the same pass (seen 2014/2016/2020/2025: a smaller, offset frame a few
    seconds apart from the main scene), which silently corrupts the grid used
    for Nuth-Kaab coregistration and inflates the recovered shift by km."""
    import rasterio as _rio
    best=None; best_n=-1
    for f in files:
        with _rio.open(f) as s:
            a=s.read(1); n=int((a!=s.nodata).sum())
        if n>best_n: best_n=n; best=f
    return best


# collect summer (Aug-Sep) DEMs by year, same selection as step 20
byyear=defaultdict(list)
for sd in sorted(BASE.glob("20*/20*")):
    nm=sd.name; m=re.match(r"(\d{4})-(\d{2})-(\d{2})",nm)
    if not m or "155_0045" not in nm: continue
    y,mo,d=map(int,m.groups())
    if mo not in (8,9): continue
    f=find(sd)
    if not f: continue
    with rasterio.open(f) as s:
        a=s.read(1).astype(float); a[(a==s.nodata)|(a<0)|(a>5000)]=np.nan; v=a[np.isfinite(a)]
    if v.size<3e5 or not (800<np.median(v)<2500): continue
    byyear[y].append(f)
years=sorted(byyear)

def load_on(f,tr,rows,cols):
    with rasterio.open(f) as s:
        a=s.read(1).astype("float32"); a[(a==s.nodata)|(a<0)|(a>5000)]=np.nan
        o=np.full((rows,cols),np.nan,"float32")
        from rasterio.warp import reproject, Resampling
        reproject(a,o,src_transform=s.transform,src_crs=s.crs,dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear)
    o[(o<0)|(o>5000)]=np.nan; return o

shp=gpd.read_file(GLINV).to_crs(crs)
rows_out=[]
ref_f=best_scene(byyear[years[len(years)//2]])  # largest-footprint mid-series scene as the common grid reference
with rasterio.open(ref_f) as s: tr=s.transform; rows,cols=s.shape
gm=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
    out_shape=(rows,cols),transform=tr,fill=0,dtype='uint8').astype(bool)
st=~gm

# slope of the reference DEM (needed to build slope-filtered stable masks, step27-style)
Zref=load_on(ref_f,tr,rows,cols)
gy,gx=np.gradient(Zref,RES)
slope_deg=np.degrees(np.arctan(np.hypot(gx,gy)))

pairs=[(years[i],years[i+1]) for i in range(len(years)-1)]
for ya,yb in pairs:
    fA=best_scene(byyear[ya]); fB=best_scene(byyear[yb])
    A=load_on(fA,tr,rows,cols); B=load_on(fB,tr,rows,cols)
    rA=xdem.DEM.from_array(A,transform=tr,crs=crs,nodata=np.nan)
    rB=xdem.DEM.from_array(B,transform=tr,crs=crs,nodata=np.nan)
    # (a) ALL stable terrain, no slope filter (what the pipeline has been doing)
    try:
        nk=xdem.coreg.NuthKaab(); nk.fit(rA,rB,inlier_mask=st,random_state=42)
        dx,dy,dz=nk.to_translations()
    except Exception as e:
        dx,dy,dz=np.nan,np.nan,np.nan
        print(f"{ya}->{yb}: coreg FAILED ({type(e).__name__})")
    # (b) slope-filtered stable terrain (2-40 deg -- see step 27 for the literature basis)
    st_slope=st & (slope_deg>=2) & (slope_deg<=40)
    try:
        nk2=xdem.coreg.NuthKaab(); nk2.fit(rA,rB,inlier_mask=st_slope,random_state=42)
        dx2,dy2,dz2=nk2.to_translations()
    except Exception as e:
        dx2,dy2,dz2=np.nan,np.nan,np.nan
    print(f"{ya}->{yb}: ALL-stable dx={dx:+.2f} dy={dy:+.2f} dz={dz:+.2f} m | "
          f"slope-filtered dx={dx2:+.2f} dy={dy2:+.2f} dz={dz2:+.2f} m  "
          f"(ref={Path(fA).stem[:10]}, tgt={Path(fB).stem[:10]})")
    rows_out.append(dict(year_a=ya,year_b=yb,dx_all=dx,dy_all=dy,dz_all=dz,
                          dx_filt=dx2,dy_filt=dy2,dz_filt=dz2,
                          ref_scene=Path(fA).stem[:10],tgt_scene=Path(fB).stem[:10]))

with open(AOUT/"COREG_offsets.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows_out[0].keys()),lineterminator="\n")
    w.writeheader()
    for r in rows_out: w.writerow(r)
print("-> COREG_offsets.csv")

# ---- outlier flagging: >2 MAD from median (dx,dy) using the ALL-stable series ----
dxs=np.array([r['dx_all'] for r in rows_out]); dys=np.array([r['dy_all'] for r in rows_out])
ok=np.isfinite(dxs)&np.isfinite(dys)
med_dx,med_dy=np.median(dxs[ok]),np.median(dys[ok])
mad=1.4826*np.median(np.hypot(dxs[ok]-med_dx,dys[ok]-med_dy))
dist=np.hypot(dxs-med_dx,dys-med_dy)
outlier=dist>2*mad
print(f"\nmedian offset (dx,dy) = ({med_dx:+.2f}, {med_dy:+.2f}) m, MAD-radius={mad:.2f} m")
print("outlier pairs (>2 MAD from the cluster):")
ref_counts=defaultdict(int); tgt_counts=defaultdict(int)
for r,o in zip(rows_out,outlier):
    if o:
        print(f"  {r['year_a']}->{r['year_b']}: dx={r['dx_all']:+.2f} dy={r['dy_all']:+.2f}  "
              f"ref={r['ref_scene']} tgt={r['tgt_scene']}")
        ref_counts[r['ref_scene']]+=1; tgt_counts[r['tgt_scene']]+=1
if ref_counts:
    print("  reference scenes appearing in outlier pairs:", dict(ref_counts))
    print("  target scenes appearing in outlier pairs:", dict(tgt_counts))

# ---- figure: dx vs dy scatter, outliers highlighted ----
fig,ax=plt.subplots(figsize=(7,7))
ax.scatter(dxs[~outlier],dys[~outlier],s=90,color="#2f7cb2",edgecolor='k',lw=0.5,zorder=3,label="within cluster")
ax.scatter(dxs[outlier],dys[outlier],s=110,color="#c0463a",marker='D',edgecolor='k',lw=0.6,zorder=4,label="outlier (>2 MAD)")
for r,x,y,o in zip(rows_out,dxs,dys,outlier):
    ax.annotate(f"{r['year_a']}→{r['year_b']}",(x,y),fontsize=8,xytext=(4,4),textcoords='offset points',
                color="#c0463a" if o else "#444")
ax.scatter([med_dx],[med_dy],marker='+',s=200,color='k',zorder=5)
circ=plt.Circle((med_dx,med_dy),2*mad,fill=False,ls='--',color='grey')
ax.add_patch(circ)
ax.set_xlabel("x shift (easting) [m]"); ax.set_ylabel("y shift (northing) [m]")
ax.set_title("Nuth–Kääb co-registration shift per DEM pair\n"
             "clustered offsets suggest a shared reference-scene geolocation bias;\n"
             "outliers (red) point to that specific pair's scene, not the pipeline")
ax.legend(loc='best',fontsize=9); ax.set_aspect('equal')
fig.tight_layout(); fig.savefig(AOUT/"COREG_offset_diagnostics.png",dpi=600,bbox_inches="tight")
print("-> COREG_offset_diagnostics.png")
