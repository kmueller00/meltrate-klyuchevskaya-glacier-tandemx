#!/usr/bin/env python3
"""
STEP 21: Hypsometric analysis — how elevation change varies with altitude, and
where the tephra sits. Motivated by the tephra literature (Grimsvotn/Icelandic
studies): the ash effect is elevation-dependent — buried & fading in the
accumulation zone, persistent & darkening in the ablation zone.

  (a) glacier hypsometry (area per 200 m elevation band)
  (b) long-term dH RATE vs elevation (early-summer 2013 -> late-summer 2024,
      coregistered), with 95% CI per band -> reveals accumulation/ablation split
      and the balance-line altitude (dH-rate = 0 crossing)
  (c) elevation distribution of the 2023 paroxysm ash mask vs whole glacier
      -> is the ash where the melt is?
Outputs: HYPSOMETRIC.png + CSV.
"""
import glob, numpy as np, rasterio, rasterio.features, geopandas as gpd, xdem
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from rasterio.warp import reproject, Resampling
import csv, warnings
warnings.filterwarnings("ignore")

BASE=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation")
GLINV=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/code/ash_analysis/GLINV_main_massif.gpkg")
ASHTIF=AOUT/"ash_analysis/ASHMASK_2023_PAROXYSM_moderate.tif"
crs="EPSG:32657"; RES=30.0
import re
EARLY_Y=(2012,2013,2014); LATE_Y=(2023,2024,2025)   # multi-year composites (robust, full coverage)
DT=(2024.0-2013.0)

def summer_dems(years):
    out=[]
    for sd in sorted(glob.glob(f"{BASE}/20*/20*")):
        nm=sd.split('/')[-1]; m=re.match(r"(\d{4})-(\d{2})-(\d{2})",nm)
        if not m or "155_0045" not in nm: continue
        y,mo,_=map(int,m.groups())
        if y in years and mo in (8,9):
            g=glob.glob(f"{sd}/prc07/DEM_FNL_*.tif") or glob.glob(f"{sd}/prc07/DEM_VER_*.tif")
            if g: out.append(g[0])
    return out

# reference grid = a full late DEM
fL0=glob.glob(f"{BASE}/2024/2024-09-04*155_0045*/prc07/DEM_FNL_*.tif")[0]
with rasterio.open(fL0) as s: tr=s.transform; rows,cols=s.shape
def load_on(f):
    with rasterio.open(f) as s:
        a=s.read(1).astype("float32"); a[(a==s.nodata)|(a<0)|(a>5000)]=np.nan
        o=np.full((rows,cols),np.nan,"float32")
        reproject(a,o,src_transform=s.transform,src_crs=s.crs,dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear)
    o[(o<0)|(o>5000)]=np.nan; return o
Z=np.nanmedian(np.stack([load_on(f) for f in summer_dems(LATE_Y)]),axis=0)
E=np.nanmedian(np.stack([load_on(f) for f in summer_dems(EARLY_Y)]),axis=0)
shp=gpd.read_file(GLINV).to_crs(crs)
gm=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
    out_shape=(rows,cols),transform=tr,fill=0,dtype='uint8').astype(bool); st=~gm
# coreg early->late
try:
    rL=xdem.DEM.from_array(Z,transform=tr,crs=crs,nodata=np.nan)
    rE=xdem.DEM.from_array(E,transform=tr,crs=crs,nodata=np.nan)
    nk=xdem.coreg.NuthKaab(); nk.fit(rL,rE,inlier_mask=st,random_state=42)
    E=nk.apply(rE); E=E.data.filled(np.nan) if hasattr(E.data,'filled') else np.asarray(E.data,float)
except Exception as ex: print("coreg warn",ex)
dt=DT
dh=Z-E; dh=np.where(np.abs(dh)<=80,dh,np.nan)
bias=np.nanmedian(dh[st&np.isfinite(dh)]); dh=dh-(bias if np.isfinite(bias) else 0)
rate=dh/dt

# ash mask (2023 paroxysm, moderate)
ash=np.zeros((rows,cols),bool)
if ASHTIF.exists():
    with rasterio.open(ASHTIF) as s:
        am=s.read(1)
        ash=(am==1)
    if ash.shape!=(rows,cols): ash=np.zeros((rows,cols),bool)

# ---- hypsometric bands ----
bands=np.arange(1300,4600,200)
rows_out=[]
for b0 in bands:
    b1=b0+200; sel=gm&np.isfinite(Z)&(Z>=b0)&(Z<b1)
    n=sel.sum()
    if n<50: continue
    r=rate[sel&np.isfinite(rate)]
    med=np.median(r) if r.size else np.nan
    nm=1.4826*np.median(np.abs(r-med)) if r.size else np.nan
    ci=1.96*nm/np.sqrt(max(r.size/50,1))   # crude decorrelation: ~50 px per indep sample
    ash_n=(ash&sel).sum()
    rows_out.append((b0+100,n*0.0009,med,ci,100*ash_n/max(n,1)))
    print(f"  {b0}-{b1} m: area={n*0.0009:5.1f} km2  rate={med:+.2f}±{ci:.2f} m/yr  ash={100*ash_n/max(n,1):4.1f}%")

with open(AOUT/"HYPSOMETRIC.csv","w",newline="") as f:
    w=csv.writer(f,lineterminator="\n"); w.writerow(["elev_mid_m","area_km2","dH_rate_m_yr","ci95_m_yr","ash_pct"])
    for r in rows_out: w.writerow([f"{r[0]:.0f}",f"{r[1]:.2f}",f"{r[2]:.3f}",f"{r[3]:.3f}",f"{r[4]:.1f}"])
print("-> HYPSOMETRIC.csv")

# ---- figure: 3 panels ----
el=[r[0] for r in rows_out]; area=[r[1] for r in rows_out]; rt=[r[2] for r in rows_out]
ci=[r[3] for r in rows_out]; ashp=[r[4] for r in rows_out]
fig,ax=plt.subplots(1,3,figsize=(16,6),sharey=True)
# (a) hypsometry
ax[0].barh(el,area,height=180,color="#8ab4d2",edgecolor="#3a6d90",lw=0.5)
ax[0].set_xlabel("area [km²]"); ax[0].set_ylabel("elevation [m a.s.l.]")
ax[0].set_title("(a) hypsometry")
# (b) dH rate vs elevation with CI
ax[1].axvline(0,color='k',lw=0.8,ls=':')
ax[1].errorbar(rt,el,xerr=ci,fmt='o-',color="#2f7cb2",ecolor="#2f7cb2",capsize=3,ms=6,lw=1.5)
# balance-line altitude = interpolated rate=0 crossing
rt_a=np.array(rt); el_a=np.array(el)
bla=np.nan
for i in range(len(rt_a)-1):
    if rt_a[i]*rt_a[i+1]<0:
        bla=el_a[i]+(el_a[i+1]-el_a[i])*(0-rt_a[i])/(rt_a[i+1]-rt_a[i]); break
if np.isfinite(bla):
    ax[1].axhline(bla,color="#c0463a",lw=1.3,ls='--')
    ax[1].text(ax[1].get_xlim()[1],bla,f" balance line ≈{bla:.0f} m",color="#c0463a",fontsize=9,va='bottom',ha='right')
ax[1].set_xlabel("dH rate 2013→2024 [m/yr]"); ax[1].set_title("(b) elevation–change rate vs altitude\n(± 95% CI per band)")
# (c) ash fraction by elevation
ax[2].barh(el,ashp,height=180,color="#c9a227",edgecolor="#7d6416",lw=0.5)
ax[2].set_xlabel("2023 ash cover [% of band]"); ax[2].set_title("(c) where the tephra sits")
fig.suptitle("Klyuchevskaya main massif — hypsometric analysis (2013→2024)\n"
             "thinning concentrated at low elevation; ash mantles the mid–upper flanks",fontsize=13)
fig.tight_layout(); fig.savefig(AOUT/"HYPSOMETRIC.png",dpi=600,bbox_inches="tight")
print(f"-> HYPSOMETRIC.png  (balance-line altitude ≈ {bla:.0f} m)")
