#!/usr/bin/env python3
"""
STEP 38: Does mapped ash/debris cover show up thermally, across the whole
year -- not just the winter geothermal check (scripts 36/37)?

Motivation: this report's central mechanism is that ash/debris lowers albedo
and (below a critical thickness) speeds melt by absorbing more solar
radiation. That predicts a testable, independent thermal signature: ash-
covered ice should read WARMER than clean ice at the same elevation whenever
there's sun to absorb (spring/summer/autumn), by a mechanism completely
separate from the optical NDSI-based ash detection this report otherwise
relies on. Winter is the interesting exception: script 36 already showed a
persistent vent-driven warm anomaly there, but for ordinary debris (no
geothermal heat), thick ash's insulating behavior (the Ostrem-curve
mechanism, see the hypsometric section) could plausibly show the OPPOSITE
sign in deep winter -- an insulated surface radiating less waste heat than
bare, actively-melting ice would in summer.

Method: Landsat 8/9 Collection-2 surface temperature (lwir11), same
scale/QA handling as scripts 36/37, composited per ASTRONOMICAL SEASON
(DJF/MAM/JJA/SON, all years pooled per season for enough cloud-free
coverage) rather than just winter. Compared elevation-band-controlled
(200 m bands, same convention as script 28/36/37) between three ash/debris
classes already established in script 35: "persistent" (flagged in >=3/4
eruption masks -- mostly the low-tongue snowmelt-timing artifact, not real
tephra), "eruption-specific" (flagged in exactly 1/4 -- likely real tephra),
and "never" (clean ice, the baseline).

Outputs: THERMAL_vs_ASHDEBRIS_seasonal.png, THERMAL_vs_ASHDEBRIS_stats.csv
"""
import sys; sys.path.insert(0,"/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/code/ash_analysis"); import _style
import glob, csv, numpy as np, rasterio, rasterio.features, geopandas as gpd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from rasterio.warp import reproject, Resampling
import pystac_client, planetary_computer, odc.stac
from s2_util import QA_EXCLUDE_MASK
import warnings; warnings.filterwarnings("ignore")

AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation")
GLINV=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/code/ash_analysis/GLINV_main_massif.gpkg")
RATE_TIF=AOUT/"summer_trend_rate_glacieronly.tif"
crs="EPSG:32657"

MASKS={
 "2019 Oct":       AOUT/"ash_analysis/ASHMASK_2019_Oct_moderate.tif",
 "2020 Dec":       AOUT/"ash_analysis/ASHMASK_2020_Dec_moderate.tif",
 "2022 Nov":       AOUT/"ash_analysis/ASHMASK_2022_Nov_moderate.tif",
 "2023 PAROXYSM":  AOUT/"ash_analysis/ASHMASK_2023_PAROXYSM_moderate.tif",
}
PERSISTENT_THRESHOLD=3

BBOX=[159.889,55.472,161.359,56.522]
CAT=pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1',
                              modifier=planetary_computer.sign_inplace)
ST_SCALE,ST_OFFSET=0.00341802,149.0

SEASONS={"DJF":(12,1,2),"MAM":(3,4,5),"JJA":(6,7,8),"SON":(9,10,11)}

# ---- (0) grid, glacier mask, elevation (full-grid mosaic, debiased -- see script 36/37) ----
with rasterio.open(RATE_TIF) as s:
    tr=s.transform; rows,cols=s.shape
shp=gpd.read_file(GLINV).to_crs(crs)
gm=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
    out_shape=(rows,cols),transform=tr,fill=0,dtype="uint8").astype(bool)
with rasterio.open(AOUT/"FULL_GRID_elevation_mosaic.tif") as s:
    elev=s.read(1).astype("float32"); elev[elev==s.nodata]=np.nan

def load_on_ref(f):
    with rasterio.open(f) as s:
        a=(s.read(1).astype("float32")==1).astype("float32")
        d=np.full((rows,cols),0.0,"float32")
        reproject(a,d,src_transform=s.transform,src_crs=s.crs,dst_transform=tr,dst_crs=crs,resampling=Resampling.nearest)
    return d>0.5

ash_layers={name: load_on_ref(f) for name,f in MASKS.items()}
n_flagged=np.stack(list(ash_layers.values())).sum(axis=0)
persistent = gm & (n_flagged>=PERSISTENT_THRESHOLD)
specific   = gm & (n_flagged==1)
never      = gm & (n_flagged==0)
print(f"persistent={persistent.sum():,}  eruption-specific={specific.sum():,}  never={never.sum():,}")

# ---- (1) per-season Landsat 8/9 thermal composite (all years pooled per season) ----
def fetch_thermal_season(months):
    it=list(CAT.search(collections=['landsat-c2-l2'],bbox=BBOX,
        datetime="2013-01-01/2026-06-01",query={'eo:cloud_cover':{'lt':60}}).items())
    it=[i for i in it if i.datetime.month in months
        and i.properties.get('platform') in ('landsat-8','landsat-9')]
    print(f"  {len(it)} Landsat 8/9 scenes")
    if not it: return None,None
    excluded=set()
    for attempt in range(6):
        use=[i for i in it if i.id not in excluded]
        ds=odc.stac.load(use,bands=["lwir11","qa_pixel"],bbox=BBOX,resolution=30.0,
                         crs=crs,groupby="solar_day",chunks={})
        st_dn=ds["lwir11"]; qa=ds["qa_pixel"]
        st_k=(st_dn.astype("float32")*ST_SCALE+ST_OFFSET).where(((qa&QA_EXCLUDE_MASK)==0)&(st_dn>0))
        med=st_k.median(dim="time",skipna=True); n_obs=(~st_k.isnull()).sum(dim="time")
        try:
            med_v=np.asarray(med.values,"float32"); n_v=np.asarray(n_obs.values,"int32")
            break
        except Exception as e:
            # a scene's asset is unreadable on the provider's end (not a transient signed-URL
            # expiry -- retrying the exact same item fails identically every time). The failing
            # URL's asset filename embeds the full USGS product id (acquisition AND processing
            # date: LC08_L2SP_PPPRRR_ACQDATE_PROCDATE_02_T1), while item.id omits the processing
            # date (LC08_L2SP_PPPRRR_ACQDATE_02_T1) -- verified live against the STAC catalog.
            # item.id with its trailing "_02_T1" stripped is therefore an unambiguous prefix of
            # the failing path (unlike matching on a single date substring, which can collide
            # between one scene's acquisition date and another's processing date).
            msg=str(e)
            bad=[i for i in use if i.id[:-6] in msg]
            if not bad:
                raise
            print(f"  scene read failed, excluding and retrying: {bad[0].id}")
            excluded.update(i.id for i in bad)
            continue
    else:
        raise RuntimeError(f"too many bad scenes in {months}")
    out_t=np.full((rows,cols),np.nan,"float32"); out_n=np.full((rows,cols),0,"int32")
    reproject(med_v,out_t,src_transform=ds.odc.transform,src_crs=crs,
              dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear)
    reproject(n_v,out_n,src_transform=ds.odc.transform,src_crs=crs,
              dst_transform=tr,dst_crs=crs,resampling=Resampling.nearest)
    return out_t,out_n

bands=np.arange(1000,4600,200)
season_class_anom={}   # (season,class) -> mean elevation-controlled anomaly [C]
rows_out=[]
for sname,months in SEASONS.items():
    print(f"\n=== {sname} ===")
    temp_k,n_obs=fetch_thermal_season(months)
    if temp_k is None:
        print("  no data, skipped"); continue
    temp_c=temp_k-273.15
    valid_t=np.isfinite(temp_k)&(n_obs>=3)&np.isfinite(elev)
    print(f"  valid px (>=3 obs): {valid_t.sum():,}")
    for cname,cmask in [("persistent",persistent),("eruption_specific",specific),("never",never)]:
        anoms=[]
        for b0 in bands:
            b1=b0+200
            inb=valid_t&(elev>=b0)&(elev<b1)
            c_px=inb&cmask; r_px=inb&never&(~cmask)
            if c_px.sum()<10 or r_px.sum()<20: continue
            anoms.append(np.nanmedian(temp_c[c_px])-np.nanmedian(temp_c[r_px]))
        m=float(np.mean(anoms)) if anoms else np.nan
        season_class_anom[(sname,cname)]=m
        print(f"  {cname:18s}: elevation-controlled anomaly vs clean = {m:+.2f}C  (n_bands={len(anoms)})")
        rows_out.append((sname,cname,m,len(anoms)))

with open(AOUT/"ash_analysis/THERMAL_vs_ASHDEBRIS_stats.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["season","class","elevation_controlled_anomaly_C","n_elev_bands"])
    for r in rows_out: w.writerow(r)
print("\n-> THERMAL_vs_ASHDEBRIS_stats.csv")

# ---- figure: seasonal curve, one line per class ----
fig,ax=plt.subplots(figsize=(9,6))
order=["DJF","MAM","JJA","SON"]
colors={"persistent":"#a8481f","eruption_specific":"#2c7bb6","never":"#999999"}
for cname,col in colors.items():
    y=[season_class_anom.get((s,cname),np.nan) for s in order]
    ax.plot(order,y,"o-",color=col,label=cname.replace("_"," "),linewidth=2,markersize=8)
ax.axhline(0,color="k",lw=0.8,ls="--")
ax.set_ylabel("elevation-controlled thermal anomaly vs clean ice [C]")
ax.set_title("Ash/debris thermal signature across the year\n(Landsat 8/9 surface temperature, all years pooled per season)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(AOUT/"ash_analysis/THERMAL_vs_ASHDEBRIS_seasonal.png",dpi=300,bbox_inches="tight")
print("-> THERMAL_vs_ASHDEBRIS_seasonal.png")
