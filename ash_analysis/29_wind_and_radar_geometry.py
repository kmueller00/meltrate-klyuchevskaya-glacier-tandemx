#!/usr/bin/env python3
import sys; sys.path.insert(0,"/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/ash_analysis"); import _style
"""
STEP 29: Two alternative explanations for the hotspot/coldspot pattern, tested
directly against data rather than argued from theory.

(A) RADAR GEOMETRY: could the fastest-thinning "hotspot" (steep, high-elevation,
    SE-facing) actually be a layover/shadow/low-coherence artifact of the
    single ascending track (heading 349.6 deg, right-looking, incidence
    ~39.4 deg -- confirmed from the .mli.par of the 2024-09-04 155_0045
    scene), rather than real elevation change? GAMMA's own per-scene
    ls_map (layover-shadow classification), loc_inc (local incidence angle)
    and int_coh (interferometric coherence) products -- computed from the
    SAME DEM/geometry used to build our analysis grid -- let us check this
    directly instead of guessing from slope/aspect alone.
(B) WIND-DRIVEN ASH: is the ash footprint itself directionally biased (i.e.
    preferentially deposited on one side of the edifice by prevailing wind
    at eruption time), rather than falling symmetrically and simply settling
    by elevation? Tested as ash-cover fraction by aspect octant WITHIN each
    200m elevation band (elevation-controlled), so a real wind signature
    isn't hidden by the elevation/ash collinearity already found in step 28.

Outputs: RADAR_GEOMETRY_hotcold.csv, ASH_BY_ASPECT_ELEVATION.csv,
         RADAR_GEOMETRY_map.png
"""
import numpy as np, rasterio, rasterio.features, geopandas as gpd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from rasterio.warp import reproject, Resampling
import glob, csv, warnings
warnings.filterwarnings("ignore")

AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation")
GLINV=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/ash_analysis/GLINV_main_massif.gpkg")
RATE_TIF=AOUT/"summer_robust_rate_glacieronly.tif"
ASH_TIFS=[AOUT/"ash_analysis/ASHMASK_2019_Oct_moderate.tif",
          AOUT/"ash_analysis/ASHMASK_2020_Dec_moderate.tif",
          AOUT/"ash_analysis/ASHMASK_2022_Nov_moderate.tif",
          AOUT/"ash_analysis/ASHMASK_2023_PAROXYSM_moderate.tif"]
crs="EPSG:32657"; RES=30.0
SAR_SCENE="/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b/2024/2024-09-04_065957_TDT_SM_A_95512_155_0045_HH./prc06/2024-09-04_065957_TDT_SM_A_95512_155_0045_HH."

with rasterio.open(RATE_TIF) as s:
    rate=s.read(1).astype("float32"); rate[rate==s.nodata]=np.nan
    tr=s.transform; rows,cols=s.shape

shp=gpd.read_file(GLINV).to_crs(crs)
gm=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
    out_shape=(rows,cols),transform=tr,fill=0,dtype='uint8').astype(bool)

fL0=glob.glob("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b/2024/2024-09-04*155_0045*/prc07/DEM_FNL_*.tif")[0]
def load_on(f,resamp=Resampling.bilinear,srcnodata=None):
    with rasterio.open(f) as s:
        a=s.read(1).astype("float32")
        if srcnodata is not None: a[a==srcnodata]=np.nan
        elif s.nodata is not None: a[a==s.nodata]=np.nan
        o=np.full((rows,cols),np.nan,"float32")
        reproject(a,o,src_transform=s.transform,src_crs=s.crs,dst_transform=tr,dst_crs=crs,resampling=resamp)
    return o
Z=load_on(fL0); Z[(Z<0)|(Z>5000)]=np.nan
gy,gx=np.gradient(Z,RES)
slope_deg=np.degrees(np.arctan(np.hypot(gx,gy)))
aspect_deg=(np.degrees(np.arctan2(gx,-gy))+360)%360

ash_any=np.zeros((rows,cols),bool)
for at in ASH_TIFS:
    if not at.exists(): continue
    with rasterio.open(at) as s:
        a=s.read(1)
        o=np.zeros((rows,cols),"float32")
        reproject((a==1).astype("float32"),o,src_transform=s.transform,src_crs=s.crs,
                  dst_transform=tr,dst_crs=crs,resampling=Resampling.average)
    ash_any |= (o>0.3)

valid=gm&np.isfinite(rate)&np.isfinite(Z)&np.isfinite(slope_deg)
p05,p95=np.nanpercentile(rate[valid],[5,95])
hot=valid&(rate<=p05); cold=valid&(rate>=p95)
print(f"valid={valid.sum():,} hot={hot.sum():,} cold={cold.sum():,}")

# ---------- (A) radar geometry ----------
ls_map=load_on(SAR_SCENE+".ls_map.tif",resamp=Resampling.nearest,srcnodata=0).astype("float32")
loc_inc=load_on(SAR_SCENE+".loc_inc.tif"); loc_inc[loc_inc<=0]=np.nan
loc_inc=np.degrees(loc_inc)   # GAMMA loc_inc is in radians (confirmed: raw ~0.69 matches .mli.par incidence_angle=39.44deg)
int_coh=load_on(SAR_SCENE+".int_coh.tif"); int_coh[int_coh<=0]=np.nan

vals,counts=np.unique(ls_map[np.isfinite(ls_map)],return_counts=True)
dominant=vals[np.argmax(counts)]
print("ls_map value counts (whole reprojected scene):", dict(zip(vals.tolist(),counts.tolist())),
      f"| dominant/'normal' code = {dominant:.0f}")
# GAMMA gc_map2 default ls_map bit encoding (confirmed from local GAMMA_SOFTWARE-20210701
# DIFF/html/gc_map2.html DESCRIPTION): 0=not tested, 1=tested/normal, 4=layover, 16=shadow,
# OR-combined -> 5=tested+layover, 17=tested+shadow, 20=layover-in-shadow.
is_layover = np.isfinite(ls_map) & (ls_map.astype(int) & 4 > 0)
is_shadow  = np.isfinite(ls_map) & (ls_map.astype(int) & 16 > 0)
flagged = is_layover | is_shadow

def geom_stats(mask,name):
    n=mask.sum()
    tested=mask&np.isfinite(ls_map)
    flag_pct=100*flagged[tested].mean() if tested.sum()>0 else np.nan
    lay_pct=100*is_layover[tested].mean() if tested.sum()>0 else np.nan
    sha_pct=100*is_shadow[tested].mean() if tested.sum()>0 else np.nan
    coh_mean=np.nanmean(int_coh[mask])
    inc_mean=np.nanmean(loc_inc[mask])
    print(f"  {name:28s} n={n:>8,}  layover={lay_pct:5.2f}%  shadow={sha_pct:5.2f}%  "
          f"total_flagged={flag_pct:5.2f}%  mean_coh={coh_mean:.3f}  mean_loc_inc={inc_mean:5.1f}deg")
    return dict(set=name,n_px=int(n),layover_pct=lay_pct,shadow_pct=sha_pct,
                ls_map_flagged_pct=flag_pct,mean_coherence=coh_mean,mean_loc_inc_deg=inc_mean)

rows_geom=[geom_stats(valid,"whole glacier"),geom_stats(hot,"HOTSPOT (fastest thinning)"),
           geom_stats(cold,"COLDSPOT (slowest/thickening)")]
with open(AOUT/"RADAR_GEOMETRY_hotcold.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows_geom[0].keys()),lineterminator="\n"); w.writeheader()
    for r in rows_geom: w.writerow(r)
print("-> RADAR_GEOMETRY_hotcold.csv")

fig,axes=plt.subplots(1,3,figsize=(17,7))
titles=["ls_map (nearest-code categorical:\nblue=normal, other=flagged)","local incidence angle [deg]","interferometric coherence"]
for ax,arr,ttl,cmap,vr in zip(axes,[np.where(valid,flagged.astype(float),np.nan),
                                     np.where(valid,loc_inc,np.nan),
                                     np.where(valid,int_coh,np.nan)],
                                 titles,["viridis","plasma","cividis"],
                                 [(0,1),(0,60),(0,1)]):
    im=ax.imshow(arr,cmap=cmap,vmin=vr[0],vmax=vr[1]); ax.set_title(ttl,fontsize=11)
    plt.colorbar(im,ax=ax,shrink=0.6)
    hs=np.where(hot,1,np.nan); cs=np.where(cold,1,np.nan)
    ax.contour(hot.astype(float),levels=[.5],colors='black',linewidths=1.1)
    ax.contour(cold.astype(float),levels=[.5],colors='lime',linewidths=1.1)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Radar-geometry check: hotspot (black outline) vs coldspot (green outline)\n"
             "against GAMMA's own layover-shadow, incidence-angle and coherence products",fontsize=12)
fig.tight_layout(); fig.savefig(AOUT/"RADAR_GEOMETRY_map.png",dpi=600,bbox_inches="tight")
print("-> RADAR_GEOMETRY_map.png")

# ---------- (B) wind-driven ash test: ash-cover by aspect octant, elevation-controlled ----------
oct_names=["N","NE","E","SE","S","SW","W","NW"]
bands=np.arange(1300,4600,200)
rows_out=[]
for b0 in bands:
    b1=b0+200; sel=valid&(Z>=b0)&(Z<b1)
    n=sel.sum()
    if n<200: continue
    octs=((aspect_deg[sel]+22.5)//45 %8).astype(int)
    ash_sel=ash_any[sel]
    row=dict(elev_mid=b0+100,n_px=int(n))
    for k,name in enumerate(oct_names):
        m=octs==k
        row[f"ash_pct_{name}"]=100*ash_sel[m].mean() if m.sum()>30 else np.nan
        row[f"n_{name}"]=int(m.sum())
    rows_out.append(row)
with open(AOUT/"ASH_BY_ASPECT_ELEVATION.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows_out[0].keys()),lineterminator="\n"); w.writeheader()
    for r in rows_out: w.writerow(r)
print("-> ASH_BY_ASPECT_ELEVATION.csv")
print("\nash-cover [%] by aspect octant within each elevation band (elevation-controlled wind test):")
hdr="elev  "+"  ".join(f"{o:>5s}" for o in oct_names)
print(hdr)
for r in rows_out:
    print(f"{r['elev_mid']:5.0f} "+" ".join(f"{r[f'ash_pct_{o}']:6.1f}" if np.isfinite(r[f'ash_pct_{o}']) else "   nan" for o in oct_names))

# overall (elevation-marginalized) octant means, weighted by n
tot=np.zeros(8); wsum=np.zeros(8)
for r in rows_out:
    for k,o in enumerate(oct_names):
        v=r[f'ash_pct_{o}']; n=r[f'n_{o}']
        if np.isfinite(v) and n>30: tot[k]+=v*n; wsum[k]+=n
overall=tot/np.maximum(wsum,1)
print("\nelevation-weighted mean ash-cover% per aspect octant (whole glacier):")
for o,v in zip(oct_names,overall): print(f"  {o:>2s}: {v:5.1f}%  (n={int(wsum[oct_names.index(o)]):,})")
