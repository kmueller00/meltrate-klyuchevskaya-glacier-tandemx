#!/usr/bin/env python3
"""
STEP 33: Within-season (intra-annual) melt-progression curve.

Every other script here compares year-to-year (late-summer bracket to next
late-summer bracket) or builds one all-time rate map. This is the first to ask
a different question: how does the glacier surface drop WITHIN a single season,
from spring (near-peak snow) through the ablation season? Newly-landed Apr-Jul
scenes (new_years_batch9/10) make this possible for most years for the first
time -- previously most years only had two Aug-Sep anchors.

Method: reproject + Nuth-Kaab coregister every Apr-Sep scene onto ONE fixed
reference grid (same convention as 04/20/dH_robust_all), stable-terrain debias,
take glacier-median dH as that scene's anomaly vs the fixed reference. Per year,
subtract that year's own first in-season observation's anomaly to get a
spring-zeroed curve -- this makes CURVE SHAPE comparable across years without
inter-annual absolute-elevation differences (deposition, long-term thinning)
contaminating it. Years are classified CORE/THIN at runtime from actual density
(not hardcoded), and CORE years are further split clean/eruption-affected by
cross-referencing the eruption windows already used in script 04, since an
eruption mid-season is a real confound for a "seasonal melt" baseline.
"""
import sys; sys.path.insert(0,"/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/ash_analysis")
import glob, re, csv, numpy as np, rasterio, rasterio.features, geopandas as gpd, xdem
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datetime import date
from pathlib import Path
from collections import defaultdict
from rasterio.warp import reproject, Resampling
import warnings; warnings.filterwarnings("ignore")

BASE=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation")
GLINV=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/ash_analysis/GLINV_main_massif.gpkg")
crs="EPSG:32657"

KNOWN_BAD={(2012,8,29),(2022,9,20)}   # mis-registered / corrupt, belt-and-suspenders on top of the NMAD/anomaly gate

# full KVERT/GVP eruption record, reused verbatim from 04_combined_signal_2015.py
ERUPT=[(date(2015,1,2),date(2015,3,24),"2015 Strombolian"),
 (date(2015,8,27),date(2015,9,10),"2015 Aug"),
 (date(2016,4,1),date(2016,5,1),"2016 Apr explosive"),
 (date(2019,4,1),date(2019,11,1),"2019 eruption"),
 (date(2020,10,2),date(2021,3,25),"2020-21 cycle"),
 (date(2022,11,20),date(2022,12,20),"2022 Nov"),
 (date(2023,6,22),date(2024,1,15),"2023-24 PAROXYSM"),
 (date(2024,8,1),date(2024,10,15),"2024 Aug-Oct")]

GAP_BREAK_DAYS=70   # widened from 45: 2020/2021 each lost exactly one genuinely-corrupt
# scene, pushing their max gap to 66/55d; the underlying dates on either side are clean.
# See run notes: a separate, unresolved per-scene processing-noise issue (not corruption,
# not coherence, not coregistration-reference choice -- all checked) rejects a large
# fraction of 2023-2025's new dates outright; those years stay thin regardless of this knob.
DOY_GRID=np.arange(90,275,5)
N_MIN=3

def find(sd):
    for t in ("DEM_FNL","DEM_VER"):
        g=glob.glob(str(sd/"prc07"/f"{t}_*.tif"))
        if g: return g[0]

def best_scene(files):
    best=None; best_n=-1
    for f in files:
        with rasterio.open(f) as s:
            a=s.read(1); n=int((a!=s.nodata).sum())
        if n>best_n: best_n=n; best=f
    return best

# ── fixed reference grid: same canonical full-footprint scene used elsewhere ──
ref_cands=glob.glob(str(BASE/"2024/2024-09-04*155_0045*/prc07/DEM_FNL_*.tif")) or \
          glob.glob(str(BASE/"2024/2024-09-04*155_0045*/prc07/DEM_VER_*.tif"))
ref_f=best_scene(ref_cands)
with rasterio.open(ref_f) as s:
    tr=s.transform; rows,cols=s.shape
    ref=s.read(1).astype("float32"); ref[(ref==s.nodata)|(ref<0)|(ref>5000)]=np.nan
refD=xdem.DEM.from_array(ref,transform=tr,crs=crs,nodata=np.nan)

shp=gpd.read_file(GLINV).to_crs(crs)
gm=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
    out_shape=(rows,cols),transform=tr,fill=0,dtype="uint8").astype(bool)
st=~gm
print(f"reference grid: {ref_f}\nglacier px={gm.sum()}  stable px={st.sum()}")

def rg(f):
    with rasterio.open(f) as s:
        a=s.read(1).astype("float32"); a[(a==s.nodata)|(a<0)|(a>5000)]=np.nan
        d=np.full((rows,cols),np.nan,"float32")
        reproject(a,d,src_transform=s.transform,src_crs=s.crs,dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear)
    d[(d<0)|(d>5000)]=np.nan; return d

SHIFT_IMPLAUSIBLE_M=100.0   # nothing in this archive's actual coreg history has ever needed
# more than ~10m; a fit landing beyond this is a failed convergence, not a real offset
RETRY_SEEDS=[42,7,123,2024,99]

def _try_fit(dem0, seed):
    tD=xdem.DEM.from_array(dem0,transform=tr,crs=crs,nodata=np.nan)
    nk=xdem.coreg.NuthKaab(); nk.fit(refD,tD,inlier_mask=st,random_state=seed)
    aff=nk.meta["outputs"]["affine"]
    if max(abs(aff["shift_x"]),abs(aff["shift_y"]))>SHIFT_IMPLAUSIBLE_M:
        return None   # failed convergence, don't even bother applying it
    out=nk.apply(tD)
    return out.data.filled(np.nan) if hasattr(out.data,"filled") else np.asarray(out.data,float)

def anomaly_vs_fixed_ref(f):
    """Returns (anom_m, reason) -- reason is None on success, else the rejection cause.
    Nuth-Kaab is not fully deterministic here (confirmed empirically: identical inputs +
    identical random_state can converge to a sane ~5m shift or an absurd ~2-4km one on
    different runs) -- retry a few seeds and keep the one with the lowest resulting
    stable-terrain NMAD, discarding any fit whose shift is already implausible before
    even scoring it."""
    dem0=rg(f)
    best_dh=None; best_nmad=np.inf
    for seed in RETRY_SEEDS:
        try:
            dem=_try_fit(dem0.copy(), seed)
            if dem is None: continue
        except Exception:
            continue
        dh_try=dem-ref; dh_try=np.where(np.abs(dh_try)<=50,dh_try,np.nan)
        stv_try=dh_try[st&np.isfinite(dh_try)]
        if stv_try.size<500: continue
        nmad_try=1.4826*np.nanmedian(np.abs(stv_try-np.nanmedian(stv_try)))
        if nmad_try<best_nmad: best_nmad=nmad_try; best_dh=dh_try
    if best_dh is None:
        dem=dem0; dh=dem-ref; dh=np.where(np.abs(dh)<=50,dh,np.nan)   # fallback: uncoregistered
    else:
        dh=best_dh
    b=np.nanmedian(dh[st&np.isfinite(dh)]); dh=dh-(b if np.isfinite(b) else 0)
    gv=dh[gm&np.isfinite(dh)]
    if gv.size<500: return None,"too few glacier px"
    stv=dh[st&np.isfinite(dh)]
    stab_nmad=1.4826*np.nanmedian(np.abs(stv-np.nanmedian(stv))) if stv.size>500 else 999
    anom=float(np.nanmedian(gv))
    if stab_nmad>6.0 or abs(anom)>15:
        return None,f"stable NMAD={stab_nmad:.1f}m, anom={anom:+.1f}m"
    return anom,None

# ── discover Apr-Sep scenes, all years ───────────────────────────────────────
# Group every raw scene folder by DATE first, then pick the single largest-
# footprint subframe per date via best_scene() BEFORE coregistering -- a date
# with two sub-frames ~6s apart (a known, previously-fixed archive pattern:
# one full-footprint primary + one smaller/offset secondary) will silently
# blow up Nuth-Kaab if the secondary (mostly-nodata-after-reprojection, sparse/
# degenerate overlap with stable terrain) gets coregistered on its own --
# confirmed empirically: 2020-04-01's secondary subframe alone produced a
# spurious shift of (1895m, -3900m, 868m) and 37m stable NMAD; its primary
# subframe (same date, 2x the valid pixels) gives a sane (2.0m, 0.3m, -0.7m)
# shift and 3.5m NMAD. Processing every subframe independently (as this script
# originally did) is the bug; group-by-date + best_scene() is the fix already
# established in 20_annual_rate_CI.py / 31_full_archive_coreg.py.
by_date_folders=defaultdict(list)
for sd in sorted(BASE.glob("20*/20*")):
    nm=sd.name; m=re.match(r"(\d{4})-(\d{2})-(\d{2})",nm)
    if not m or "155_0045" not in nm: continue
    y,mo,d=map(int,m.groups())
    if mo not in (4,5,6,7,8,9): continue
    if (y,mo,d) in KNOWN_BAD: continue
    f=find(sd)
    if f: by_date_folders[(y,mo,d)].append(f)

series=defaultdict(list)   # year -> sorted [(date, anom_vs_fixed_ref)]
rejected=[]
for (y,mo,d),files in sorted(by_date_folders.items()):
    dt=date(y,mo,d)
    f=best_scene(files) if len(files)>1 else files[0]
    with rasterio.open(f) as s:
        a=s.read(1).astype(float); a[(a==s.nodata)|(a<0)|(a>5000)]=np.nan
        v=a[np.isfinite(a)]
    if v.size<3e5 or not (800<np.median(v)<2500):
        rejected.append((y,mo,d,"discovery gate: too few valid px / implausible median")); continue
    anom,reason=anomaly_vs_fixed_ref(f)
    if anom is None:
        rejected.append((y,mo,d,reason)); continue
    series[y].append((dt,anom))
    print(f"  {dt}: anom_vs_fixed_ref={anom:+.2f}m  ({len(files)} subframe(s), used {Path(f).parent.parent.name})")

for y in series: series[y]=sorted(series[y])

# ── per-year spring-zeroed curve + runtime CORE/THIN classification ──────────
years=sorted(series)
curves={}       # year -> [(doy, anom_vs_spring)]
tier={}         # year -> "core"/"thin"
for y in years:
    pts=series[y]
    doys=[dt.timetuple().tm_yday for dt,_ in pts]
    vals=[a for _,a in pts]
    zero=vals[0]
    curve=list(zip(doys,[v-zero for v in vals]))
    curves[y]=curve
    n=len(pts); months=sorted(set(dt.month for dt,_ in pts))
    first_doy=doys[0]
    max_gap=max((doys[i+1]-doys[i] for i in range(len(doys)-1)), default=0)
    is_core = n>=8 and len(months)>=5 and first_doy<=105 and max_gap<=GAP_BREAK_DAYS
    tier[y]="core" if is_core else "thin"
    print(f"{y}: n={n} months={months} first_doy={first_doy} max_gap={max_gap} -> {tier[y]}")

core_years=[y for y in years if tier[y]=="core"]
def eruption_overlap(y):
    lo,hi=date(y,4,1),date(y,9,30)
    return any(s<=hi and e>=lo for s,e,_ in ERUPT)
clean_years=[y for y in core_years if not eruption_overlap(y)]
print(f"\nCORE years: {core_years}")
print(f"CLEAN (no eruption overlap Apr-Sep): {clean_years}")
print(f"THIN/excluded: {[y for y in years if tier[y]=='thin']}")

# ── interpolate onto common DOY grid, NaN outside range and inside big gaps ──
def interp_to_grid(curve):
    doys=np.array([d for d,_ in curve]); vals=np.array([v for _,v in curve])
    g=np.interp(DOY_GRID,doys,vals,left=np.nan,right=np.nan)
    for i in range(len(doys)-1):
        if doys[i+1]-doys[i] > GAP_BREAK_DAYS:
            g[(DOY_GRID>doys[i])&(DOY_GRID<doys[i+1])]=np.nan
    return g

grid_curves={y:interp_to_grid(curves[y]) for y in core_years}
clean_stack=np.stack([grid_curves[y] for y in clean_years]) if clean_years else np.empty((0,len(DOY_GRID)))
n_valid=np.sum(np.isfinite(clean_stack),axis=0)
mean_curve=np.where(n_valid>=N_MIN, np.nanmean(np.where(np.isfinite(clean_stack),clean_stack,np.nan),axis=0), np.nan)
std_curve =np.where(n_valid>=N_MIN, np.nanstd (np.where(np.isfinite(clean_stack),clean_stack,np.nan),axis=0), np.nan)

def rate_curve(curve):
    doys=np.array([d for d,_ in curve]); vals=np.array([v for _,v in curve])
    mids=(doys[1:]+doys[:-1])/2.0
    rates=(vals[1:]-vals[:-1])/np.maximum(doys[1:]-doys[:-1],1)
    return mids,rates

# ── plot ──────────────────────────────────────────────────────────────────
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(12,10),sharex=True)
month_ticks=[date(2001,m,1).timetuple().tm_yday for m in range(4,11)]
month_labs=["Apr","May","Jun","Jul","Aug","Sep","Oct"]

for y in years:
    doys=[d for d,_ in curves[y]]; vals=[v for _,v in curves[y]]
    if tier[y]=="thin":
        ax1.plot(doys,vals,color="grey",ls=":",lw=1.1,alpha=0.7,zorder=1)
    else:
        col = "#1f6f78" if y in clean_years else "#a8481f"
        ax1.plot(doys,vals,color=col,lw=1.3,alpha=0.85,zorder=2,
                  label=f"{y}"+("" if y in clean_years else " (eruption)"))
if len(clean_years)>0:
    ax1.plot(DOY_GRID,mean_curve,color="#0d3b40",lw=3.2,zorder=4,label=f"clean-year mean (n={len(clean_years)})")
    ax1.fill_between(DOY_GRID,mean_curve-std_curve,mean_curve+std_curve,color="#0d3b40",alpha=0.15,zorder=3)
ax1.axhline(0,color="k",lw=0.6,ls=":")
ax1.set_ylabel("elevation anomaly vs each year's own\nfirst spring/early-summer scene [m]")
ax1.set_title("Klyuchevskaya within-season melt progression\n"
              "teal = clean (no eruption Apr–Sep that year), ember = eruption-affected, "
              "dotted grey = thin/excluded year")
ax1.legend(loc="lower left",fontsize=7.5,ncol=3)
ax1.grid(alpha=0.3)

for y in core_years:
    mids,rates=rate_curve(curves[y])
    col = "#1f6f78" if y in clean_years else "#a8481f"
    ax2.plot(mids,rates,color=col,lw=1.1,alpha=0.8,marker="o",ms=3)
ax2.axhline(0,color="k",lw=0.6,ls=":")
ax2.set_ylabel("melt rate [m/day]\n(finite difference, consecutive scenes)")
ax2.set_xlabel("day of year")
ax2.set_xticks(month_ticks); ax2.set_xticklabels(month_labs)
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(AOUT/"WITHIN_SEASON_melt_curve.png",dpi=600,bbox_inches="tight")
print("\n-> WITHIN_SEASON_melt_curve.png")

# ── CSV: used scenes + rejected scenes ───────────────────────────────────────
with open(AOUT/"WITHIN_SEASON_melt_curve.csv","w",newline="") as fh:
    w=csv.writer(fh)
    w.writerow(["year","date","doy","anom_vs_fixed_ref_m","anom_vs_spring_baseline_m","tier","eruption_flag"])
    for y in years:
        zero=series[y][0][1]
        for dt,a in series[y]:
            w.writerow([y,dt.isoformat(),dt.timetuple().tm_yday,f"{a:.3f}",f"{a-zero:.3f}",
                        tier[y], "eruption" if (tier[y]=="core" and y not in clean_years) else ""])
    w.writerow([])
    w.writerow(["REJECTED SCENES","","","","","",""])
    w.writerow(["year","month","day","reason"])
    for y,mo,d,reason in rejected:
        w.writerow([y,mo,d,reason])
print("-> WITHIN_SEASON_melt_curve.csv")
