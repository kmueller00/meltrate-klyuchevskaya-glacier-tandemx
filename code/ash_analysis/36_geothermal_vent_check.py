#!/usr/bin/env python3
"""
STEP 36: Geothermal delineation of the melt HOTSPOT -- Apakhonchich flank-vent system.

Script 34 ruled out surge (hotspot flows no faster than glacier-wide background,
ITS_LIVE). The other alternative the advisor raised was geothermal heating:
Klyuchevskaya's documented Apakhonchich flank-vent system sits on the same SE
flank as the hotspot -- a real, named feature (GVP eruption reports: Feb 1987
fracture on the SE flank at 3,400-3,900 m; 1990 report of a new vent at 4,500 m
on the upper SE flank/NE slope of the Apakhonchich valley, with a subsidiary
vent at 3,970 m feeding lava flows). Until now this was a literature-only lead
(README, script 34 docstring both flag it as "not yet spatially confirmed") --
this script actually checks it against real data:

  (1) SPATIAL OVERLAP: rasterize an "Apakhonchich flank zone" as a bearing
      wedge from the summit (56.056044N 160.644089E, GVP) toward the SE
      (100-170 deg from north) intersected with the documented vent elevation
      band (3,400-4,500 m, read off the same reference DEM used throughout
      this project) and a generous 8 km summit-distance cap. Checks what
      fraction of the HOTSPOT (script 28's rate<=P5, long-baseline definition,
      reused verbatim) falls inside it, and vice versa.
  (2) THERMAL SIGNAL: MODVOLC/MIROVA aren't reachable through a scriptable API
      without a separate account signup, so this uses Landsat 8/9 Collection-2
      Level-2 surface-temperature (band lwir11, scale/offset verified live
      against the STAC item, not assumed) instead -- same Planetary Computer
      access already built for the ash masks. Composited over DEC-FEB (peak
      snow cover, maximum thermal contrast against any persistent heat source),
      per-pixel cloud/shadow/fill masked via qa_pixel. A real geothermal
      anomaly should show elevated surface temperature that survives an
      elevation control (temperature falls with elevation everywhere on the
      cone; comparing raw temps between the vent zone at 4000m+ and the
      glacier ablation zone at 2000m would just recover the lapse rate, not
      a heat source) -- so flank-zone vs rest-of-grid is compared per 200 m
      elevation band, not in aggregate.

Outputs: GEOTHERMAL_VENT_check_map.png, GEOTHERMAL_VENT_check_stats.csv
"""
import sys; sys.path.insert(0,"/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/code/ash_analysis"); import _style
import glob, csv, numpy as np, rasterio, rasterio.features, rasterio.transform, geopandas as gpd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from pyproj import Transformer
from rasterio.warp import reproject, Resampling
import pystac_client, planetary_computer, odc.stac
from s2_util import QA_EXCLUDE_MASK
import warnings; warnings.filterwarnings("ignore")

AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation")
GLINV=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/code/ash_analysis/GLINV_main_massif.gpkg")
BASE=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
RATE_TIF=AOUT/"summer_trend_rate_glacieronly.tif"
BASELINE_TIF=AOUT/"summer_trend_baseline_years.tif"
MIN_BASELINE_YR=10.0                  # same restriction as script 28
crs="EPSG:32657"

SUMMIT_LONLAT=(160.644089,56.056044)  # GVP: 56.056044N 160.644089E
FLANK_BEARING=(100.0,170.0)           # deg from N, "SE flank" sector (documented vents + Apakhonchich valley)
FLANK_ELEV=(3400.0,4500.0)            # m, union of 1987 fracture (3400-3900) + 1990 vents (3970, 4500)
FLANK_MAXDIST=8000.0                  # m from summit, generous upper-flank cap

BBOX=[159.889,55.472,161.359,56.522]
CAT=pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1',
                              modifier=planetary_computer.sign_inplace)
ST_SCALE,ST_OFFSET=0.00341802,149.0   # landsat-c2-l2 lwir11 raster:bands, verified live against STAC item

# ---- (0) load rate/baseline grid, recompute hotspot exactly as script 28 ----
with rasterio.open(RATE_TIF) as s:
    tr=s.transform; rows,cols=s.shape; rate=s.read(1).astype("float32")
    rate[rate==s.nodata]=np.nan
with rasterio.open(BASELINE_TIF) as s:
    baseline_yr=s.read(1).astype("float32"); baseline_yr[baseline_yr==s.nodata]=np.nan

shp=gpd.read_file(GLINV).to_crs(crs)
gm=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
    out_shape=(rows,cols),transform=tr,fill=0,dtype="uint8").astype(bool)
valid=gm&np.isfinite(rate)
longbase=valid&np.isfinite(baseline_yr)&(baseline_yr>=MIN_BASELINE_YR)
p05,p95=np.nanpercentile(rate[longbase],[5,95])
hot=longbase&(rate<=p05); cold=longbase&(rate>=p95)
print(f"hotspot: {hot.sum():,} px  coldspot: {cold.sum():,} px  (P5={p05:.2f} P95={p95:.2f} m/yr)")

# ---- (1) reference elevation grid (same 2024-09-04 scene used throughout) ----
ref_f=glob.glob(str(BASE/"2024/2024-09-04*155_0045*/prc07/DEM_FNL_*.tif"))[0]
with rasterio.open(ref_f) as s:
    elev=np.full((rows,cols),np.nan,"float32")
    a=s.read(1).astype("float32"); a[(a==s.nodata)|(a<0)|(a>5000)]=np.nan
    reproject(a,elev,src_transform=s.transform,src_crs=s.crs,dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear)
print(f"reference grid elevation range where valid: {np.nanmin(elev):.0f}-{np.nanmax(elev):.0f}m "
      f"(hotspot mean elev: {np.nanmean(elev[hot]):.0f}m)")

# ---- (2) Apakhonchich flank-zone wedge ----
xs=np.arange(cols)*tr.a+tr.c+tr.a/2; ys=np.arange(rows)*tr.e+tr.f+tr.e/2
XX,YY=np.meshgrid(xs,ys)
tfm=Transformer.from_crs("EPSG:4326",crs,always_xy=True)
x0,y0=tfm.transform(*SUMMIT_LONLAT)
dx,dy=XX-x0,YY-y0
bearing=np.degrees(np.arctan2(dx,dy))%360.0
dist=np.hypot(dx,dy)
flank=((bearing>=FLANK_BEARING[0])&(bearing<=FLANK_BEARING[1])
       &(dist<=FLANK_MAXDIST)&(elev>=FLANK_ELEV[0])&(elev<=FLANK_ELEV[1]))
print(f"Apakhonchich flank zone (bearing {FLANK_BEARING}, elev {FLANK_ELEV}, dist<={FLANK_MAXDIST/1000:.0f}km): "
      f"{flank.sum():,} px on grid, {(flank&gm).sum():,} of those on glacier ice")

overlap_hf=hot&flank
pct_hotspot_in_flank=100*overlap_hf.sum()/max(hot.sum(),1)
pct_flank_is_hotspot=100*overlap_hf.sum()/max(flank.sum(),1)
print(f"SPATIAL: {overlap_hf.sum():,} px overlap -- {pct_hotspot_in_flank:.1f}% of hotspot area falls inside the "
      f"flank zone; {pct_flank_is_hotspot:.1f}% of the flank zone is hotspot")
print(f"  hotspot mean elev {np.nanmean(elev[hot]):.0f}m vs flank-zone band {FLANK_ELEV[0]:.0f}-{FLANK_ELEV[1]:.0f}m "
      f"({'OVERLAPS' if np.nanmean(elev[hot])>=FLANK_ELEV[0] else 'BELOW vent elevation -- separate zones'})")

# ---- (3) winter thermal composite (Landsat 8/9 lwir11, DEC-FEB, all years, per-pixel QA-masked) ----
def fetch_thermal_winter():
    # platform restricted to landsat-8/9 -- Landsat 7 uses a single 'lwir' band (different
    # name AND bandpass/calibration than 8/9's lwir11), and odc.stac's band-alias resolver
    # can fail outright ("No such band/alias: lwir11") when a batch mixes it in -- same
    # fail-closed platform filter s2_util.py already uses for the optical Landsat fetch.
    it=list(CAT.search(collections=['landsat-c2-l2'],bbox=BBOX,
        datetime="2013-01-01/2026-06-01",query={'eo:cloud_cover':{'lt':60}}).items())
    it=[i for i in it if i.datetime.month in (12,1,2)
        and i.properties.get('platform') in ('landsat-8','landsat-9')]
    print(f"winter (Dec-Feb) Landsat 8/9 scenes found: {len(it)}")
    if not it: return None
    ds=odc.stac.load(it,bands=["lwir11","qa_pixel"],bbox=BBOX,resolution=30.0,
                     crs=crs,groupby="solar_day",chunks={})
    st_dn=ds["lwir11"]; qa=ds["qa_pixel"]
    st_k=st_dn.astype("float32")*ST_SCALE+ST_OFFSET
    ok=((qa&QA_EXCLUDE_MASK)==0)&(st_dn>0)
    st_k=st_k.where(ok)
    med=st_k.median(dim="time",skipna=True)
    n_obs=ok.sum(dim="time")
    out_t=np.full((rows,cols),np.nan,"float32"); out_n=np.full((rows,cols),0,"int32")
    reproject(np.asarray(med.values,"float32"),out_t,src_transform=ds.odc.transform,src_crs=crs,
              dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear)
    reproject(np.asarray(n_obs.values,"int32"),out_n,src_transform=ds.odc.transform,src_crs=crs,
              dst_transform=tr,dst_crs=crs,resampling=Resampling.nearest)
    return out_t,out_n

temp_k,n_obs=fetch_thermal_winter()
temp_c=temp_k-273.15 if temp_k is not None else None
valid_t=(temp_k is not None)&np.isfinite(temp_k)&(n_obs>=3)
print(f"thermal composite: {valid_t.sum():,} px with >=3 clear winter observations "
      f"(median count {np.nanmedian(n_obs[valid_t]) if valid_t.sum() else float('nan'):.0f})")

# ---- (4) elevation-controlled comparison: flank-zone vs rest-of-grid, per 200m band ----
rows_out=[]
if temp_k is not None:
    print("\n=== elevation-controlled thermal anomaly (flank zone vs rest, per 200m band) ===")
    bands=np.arange(2000,4700,200)
    for b0 in bands:
        b1=b0+200
        inb=valid_t&np.isfinite(elev)&(elev>=b0)&(elev<b1)
        f_px=inb&flank; r_px=inb&(~flank)
        if f_px.sum()<5 or r_px.sum()<20: continue
        t_f=np.nanmedian(temp_c[f_px]); t_r=np.nanmedian(temp_c[r_px])
        print(f"  {b0}-{b1}m: flank-zone n={f_px.sum():4d} medianT={t_f:+6.2f}C  |  rest n={r_px.sum():5d} medianT={t_r:+6.2f}C  "
              f"anomaly={t_f-t_r:+.2f}C")
        rows_out.append((b0,b1,int(f_px.sum()),float(t_f),int(r_px.sum()),float(t_r),float(t_f-t_r)))
    if rows_out:
        anoms=[r[6] for r in rows_out]
        print(f"\nmean elevation-controlled anomaly across bands: {np.mean(anoms):+.2f}C "
              f"({'warmer than surroundings -- consistent with a persistent heat source' if np.mean(anoms)>0.5 else 'no consistent warm anomaly'})")
else:
    print("no usable thermal composite -- skipping anomaly comparison")

with open(AOUT/"ash_analysis/GEOTHERMAL_VENT_check_stats.csv","w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["metric","value"])
    w.writerow(["hotspot_px",int(hot.sum())])
    w.writerow(["flank_zone_px",int(flank.sum())])
    w.writerow(["overlap_px",int(overlap_hf.sum())])
    w.writerow(["pct_hotspot_in_flank",round(pct_hotspot_in_flank,2)])
    w.writerow(["pct_flank_is_hotspot",round(pct_flank_is_hotspot,2)])
    w.writerow(["hotspot_mean_elev_m",round(float(np.nanmean(elev[hot])),1)])
    w.writerow(["flank_elev_band_m",f"{FLANK_ELEV[0]:.0f}-{FLANK_ELEV[1]:.0f}"])
    w.writerow([])
    w.writerow(["elev_band_lo","elev_band_hi","flank_n","flank_medianT_C","rest_n","rest_medianT_C","anomaly_C"])
    for r in rows_out: w.writerow(r)
print("-> GEOTHERMAL_VENT_check_stats.csv")

# ---- figure: hotspot/coldspot/flank-zone map + thermal anomaly panel ----
left,bottom,right,top=rasterio.transform.array_bounds(rows,cols,tr)
ext=[left,right,bottom,top]
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(17,9))

gm_show=np.where(gm,1,np.nan)
ax1.imshow(gm_show,cmap=ListedColormap(["#e8e4d8"]),extent=ext,zorder=1)
cat=np.full((rows,cols),np.nan)
cat[flank]=1; cat[hot]=2; cat[hot&flank]=3
im=ax1.imshow(cat,cmap=ListedColormap(["#f4a261","#2c7bb6","#a8481f"]),extent=ext,zorder=2,vmin=1,vmax=3)
shp.boundary.plot(ax=ax1,color='k',linewidth=0.6,zorder=3)
sx,sy=x0,y0
ax1.plot(sx,sy,marker="^",color="red",markersize=10,zorder=4,label="summit")
ax1.set_xlim(left,right); ax1.set_ylim(bottom,top)
ax1.legend(loc="lower left")
ax1.set_title(f"orange = Apakhonchich flank zone (documented vent elev+bearing)\n"
              f"blue = hotspot (rate<=P5) | ember = overlap ({pct_hotspot_in_flank:.0f}% of hotspot)")

if rows_out:
    b0s=[r[0]+100 for r in rows_out]; tf=[r[3] for r in rows_out]; tr_=[r[5] for r in rows_out]
    ax2.plot(tf,b0s,"o-",color="#a8481f",label="flank zone")
    ax2.plot(tr_,b0s,"o-",color="#999999",label="rest of grid")
    ax2.set_xlabel("winter (Dec-Feb) median surface temp [C]"); ax2.set_ylabel("elevation [m]")
    ax2.set_title("elevation-controlled winter thermal comparison\n(Landsat C2L2 lwir11, per 200m band)")
    ax2.legend(); ax2.grid(alpha=0.3)
else:
    ax2.text(0.5,0.5,"no thermal composite available",ha="center",va="center",transform=ax2.transAxes)

fig.tight_layout(); fig.savefig(AOUT/"ash_analysis/GEOTHERMAL_VENT_check_map.png",dpi=600,bbox_inches="tight")
print("-> GEOTHERMAL_VENT_check_map.png")
