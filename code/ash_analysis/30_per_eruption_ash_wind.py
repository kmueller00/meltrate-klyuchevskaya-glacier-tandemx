#!/usr/bin/env python3
import sys; sys.path.insert(0,"/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/code/ash_analysis"); import _style
"""
STEP 30: Per-eruption (not unioned) ash-cover-by-aspect test against ERA5 wind.

Step 29 found a strong elevation-controlled azimuthal ash pattern in the
UNION of all 4 eruptions' ash masks (SW/W ~50% vs E/SE ~30%). Two
explanations compete:
  (a) WIND-DRIVEN: prevailing wind at each eruption carried tephra onto a
      specific flank -- if so, each eruption's individual ash-aspect bias
      should track ITS OWN wind direction, and should differ eruption to
      eruption if the wind differed.
  (b) SOLAR-ASPECT / SNOWMELT-TIMING confound: SW/W slopes get more
      afternoon insolation, melt out earlier, and the ash mask (NDSI-drop +
      darkening) is already known to conflate fresh tephra with plain
      snow-loss -- if so, ALL FOUR eruptions should show roughly the SAME
      SW/W bias regardless of that eruption's actual wind.

This script computes the elevation-controlled ash-by-aspect-octant
distribution SEPARATELY for each of the 4 eruption masks, and pulls ERA5
near-surface wind direction (10m and 100m, via Open-Meteo archive -- note:
pressure-level/plume-height wind is NOT available on this free endpoint,
confirmed empirically, so this is a near-surface proxy only, most relevant
for PROXIMAL ashfall onto the edifice itself) for a window after each
eruption date.
Outputs: PER_ERUPTION_ash_by_aspect.csv, ERUPTION_wind.csv,
         PER_ERUPTION_ash_aspect_rose.png
"""
import numpy as np, rasterio, rasterio.features, geopandas as gpd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from rasterio.warp import reproject, Resampling
from datetime import date, timedelta
import glob, csv, json, urllib.request, urllib.parse, warnings
warnings.filterwarnings("ignore")

AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation")
GLINV=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/code/ash_analysis/GLINV_main_massif.gpkg")
RATE_TIF=AOUT/"summer_trend_rate_glacieronly.tif"   # grid template only (script 30 doesn't
# use rate values itself) -- switched from the old two-bucket tif (65.6% coverage) to
# match the grid every other script now uses, so the elevation mosaic below lines up.
crs="EPSG:32657"; RES=30.0
LAT,LON=56.06,160.63

ERUPTIONS=[("2019_Oct",      AOUT/"ash_analysis/ASHMASK_2019_Oct_moderate.tif",      date(2019,10,25)),
           ("2020_Dec",      AOUT/"ash_analysis/ASHMASK_2020_Dec_moderate.tif",      date(2020,12,9)),
           ("2022_Nov",      AOUT/"ash_analysis/ASHMASK_2022_Nov_moderate.tif",      date(2022,11,20)),
           ("2023_PAROXYSM", AOUT/"ash_analysis/ASHMASK_2023_PAROXYSM_moderate.tif", date(2023,11,1))]

with rasterio.open(RATE_TIF) as s:
    tr=s.transform; rows,cols=s.shape

shp=gpd.read_file(GLINV).to_crs(crs)
gm=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
    out_shape=(rows,cols),transform=tr,fill=0,dtype='uint8').astype(bool)

def load_on(f,resamp=Resampling.bilinear):
    with rasterio.open(f) as s:
        a=s.read(1).astype("float32")
        if s.nodata is not None: a[a==s.nodata]=np.nan
        o=np.full((rows,cols),np.nan,"float32")
        reproject(a,o,src_transform=s.transform,src_crs=s.crs,dst_transform=tr,dst_crs=crs,resampling=resamp)
    return o
# full-grid elevation composite (98.3% coverage) instead of a single reference
# scene (63.5%) -- see script 36/37 for why this was fixed.
with rasterio.open(AOUT/"FULL_GRID_elevation_mosaic.tif") as s:
    Z=s.read(1).astype("float32"); Z[Z==s.nodata]=np.nan
gy,gx=np.gradient(Z,RES)
aspect_deg=(np.degrees(np.arctan2(gx,-gy))+360)%360
valid=gm&np.isfinite(Z)

oct_names=["N","NE","E","SE","S","SW","W","NW"]
bands=np.arange(1300,4600,200)

# ---------- per-eruption ash-by-aspect, elevation-controlled ----------
rows_out=[]
per_erup_octant={}
for name,mpath,edate in ERUPTIONS:
    if not mpath.exists():
        print(f"MISSING mask for {name}: {mpath}"); continue
    with rasterio.open(mpath) as s:
        a=s.read(1)
        o=np.zeros((rows,cols),"float32")
        reproject((a==1).astype("float32"),o,src_transform=s.transform,src_crs=s.crs,
                  dst_transform=tr,dst_crs=crs,resampling=Resampling.average)
    ash=o>0.3
    tot=np.zeros(8); wsum=np.zeros(8)
    for b0 in bands:
        b1=b0+200; sel=valid&(Z>=b0)&(Z<b1)
        n=sel.sum()
        if n<200: continue
        octs=((aspect_deg[sel]+22.5)//45 %8).astype(int)
        ash_sel=ash[sel]
        for k in range(8):
            m=octs==k
            if m.sum()>30:
                tot[k]+=100*ash_sel[m].mean()*m.sum(); wsum[k]+=m.sum()
    overall=tot/np.maximum(wsum,1)
    per_erup_octant[name]=overall
    dom_k=int(np.argmax(overall)); dom=oct_names[dom_k]
    print(f"{name} (erupted {edate}): ash-cover% by octant -> " +
          " ".join(f"{o}={v:.0f}" for o,v in zip(oct_names,overall)) +
          f"   | dominant={dom} ({overall[dom_k]:.0f}%)")
    row=dict(eruption=name, eruption_date=edate.isoformat(), dominant_octant=dom)
    for o,v in zip(oct_names,overall): row[f"ash_pct_{o}"]=round(float(v),1)
    rows_out.append(row)

with open(AOUT/"PER_ERUPTION_ash_by_aspect.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows_out[0].keys()),lineterminator="\n"); w.writeheader()
    for r in rows_out: w.writerow(r)
print("-> PER_ERUPTION_ash_by_aspect.csv")

# ---------- ERA5 near-surface wind for a window after each eruption ----------
print("\nERA5 near-surface wind (10m & 100m), 7-day window starting at eruption date:")
wind_rows=[]
def circmean_deg(degs,weights=None):
    r=np.radians(degs)
    if weights is None: weights=np.ones_like(degs)
    s=np.sum(weights*np.sin(r)); c=np.sum(weights*np.cos(r))
    return (np.degrees(np.arctan2(s,c))+360)%360

for name,mpath,edate in ERUPTIONS:
    start=edate; end=edate+timedelta(days=7)
    q=urllib.parse.urlencode(dict(latitude=LAT,longitude=LON,
        start_date=start.isoformat(),end_date=end.isoformat(),
        hourly="winddirection_10m,windspeed_10m,winddirection_100m,windspeed_100m",timezone="UTC"))
    url=f"https://archive-api.open-meteo.com/v1/archive?{q}"
    try:
        with urllib.request.urlopen(url,timeout=60) as r: met=json.load(r)
        h=met["hourly"]
        d10=np.array(h["winddirection_10m"],float); s10=np.array(h["windspeed_10m"],float)
        d100=np.array(h["winddirection_100m"],float); s100=np.array(h["windspeed_100m"],float)
        mean10=circmean_deg(d10,s10); mean100=circmean_deg(d100,s100)
        mws10=np.nanmean(s10); mws100=np.nanmean(s100)
        # direction wind BLOWS TOWARD (meteorological convention reports direction wind blows FROM)
        toward10=(mean10+180)%360; toward100=(mean100+180)%360
        def compass(deg):
            octs8=["N","NE","E","SE","S","SW","W","NW"]
            return octs8[int((deg+22.5)//45 %8)]
        print(f"  {name}: wind FROM {compass(mean10)} ({mean10:.0f}deg,10m,{mws10:.1f}m/s) / "
              f"FROM {compass(mean100)} ({mean100:.0f}deg,100m,{mws100:.1f}m/s)  "
              f"-> blows TOWARD {compass(toward10)}(10m) / {compass(toward100)}(100m)")
        wind_rows.append(dict(eruption=name,eruption_date=edate.isoformat(),
            mean_wind_from_deg_10m=round(mean10,1),mean_windspeed_10m=round(mws10,2),
            mean_wind_from_deg_100m=round(mean100,1),mean_windspeed_100m=round(mws100,2),
            blows_toward_octant_10m=compass(toward10),blows_toward_octant_100m=compass(toward100)))
    except Exception as e:
        print(f"  {name}: ERA5 fetch FAILED ({e})")

with open(AOUT/"ERUPTION_wind.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(wind_rows[0].keys()),lineterminator="\n"); w.writeheader()
    for r in wind_rows: w.writerow(r)
print("-> ERUPTION_wind.csv")

# ---------- figure: small-multiple polar bar per eruption ----------
fig,axes=plt.subplots(1,4,figsize=(18,5),subplot_kw=dict(projection='polar'))
theta=np.radians([0,45,90,135,180,225,270,315])
for ax,(name,_,edate) in zip(axes,ERUPTIONS):
    if name not in per_erup_octant: continue
    vals=per_erup_octant[name]
    ax.bar(theta,vals,width=0.7,color="#c9a227",edgecolor="#7d6416")
    ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
    ax.set_xticks(theta); ax.set_xticklabels(oct_names,fontsize=8)
    ax.set_title(f"{name}\n({edate})",fontsize=10)
    wr=[w for w in wind_rows if w["eruption"]==name]
    if wr:
        ax.set_ylabel(f"wind→{wr[0]['blows_toward_octant_100m']}",fontsize=8)
fig.suptitle("Ash-cover [%] by aspect octant, per eruption (elevation-controlled)\n"
             "vs ERA5 100m wind direction the mask is blowing toward (label under each panel)",fontsize=12)
fig.tight_layout(); fig.savefig(AOUT/"PER_ERUPTION_ash_aspect_rose.png",dpi=600,bbox_inches="tight")
print("-> PER_ERUPTION_ash_aspect_rose.png")
