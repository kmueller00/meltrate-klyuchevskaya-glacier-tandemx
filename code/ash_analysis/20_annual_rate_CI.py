#!/usr/bin/env python3
"""
STEP 20 (FLAGSHIP): Per-year glacier-wide elevation-change rate with a rigorous,
spatially-correlated confidence interval.

For each consecutive summer->summer pair (penetration-immune):
  * median-composite each year's Aug-Sep DEMs, Nuth-Kaab coregister B->A,
    de-bias on stable terrain -> dH; rate = dH / dt_years
  * glacier-wide mean rate = area-weighted mean of glacier dH / dt
  * UNCERTAINTY (Rolstad 2009 / Hugonnet 2022 via xdem.spatialstats):
      - sample the empirical variogram of dH on STABLE (off-glacier) terrain
      - fit a sum of 2 models (Gaussian short range + spherical long range)
      - Neff = number_effective_samples(glacier_area, variogram)  [not N_pixels!]
      - sigma_mean = sigma_stable / sqrt(Neff);  95% CI = 1.96 * sigma_mean / dt
    Falls back to a decorrelation-length Neff if the variogram fit fails.
Outputs: ANNUAL_rate_CI.png (time series, error bars, eruptions marked) + CSV.
"""
import glob, re, numpy as np, rasterio, rasterio.features, geopandas as gpd, xdem
import xdem.spatialstats as ss
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import date2num
from pathlib import Path
from datetime import date
from collections import defaultdict
from rasterio.warp import reproject, Resampling
import csv, warnings
warnings.filterwarnings("ignore")

BASE=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation")
GLINV=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/code/ash_analysis/GLINV_main_massif.gpkg")
crs="EPSG:32657"; RES=30.0
ERUPT=[(2019.8,"2019 Oct",0),(2020.94,"2020 Dec",0),(2022.9,"2022 Nov",0),
       (2023.83,"2023 paroxysm",1),(2024.6,"2024",0),(2015.05,"2015",1)]

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


# summer DEMs by year
byyear=defaultdict(list); dates_by=defaultdict(list)
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
    byyear[y].append(f); dates_by[y].append(date(y,mo,d))
years=sorted(byyear)
print("summer years:",years)

shp=gpd.read_file(GLINV).to_crs(crs)
GL_AREA=shp.geometry.area.sum()   # m2

def load_on(f,tr,rows,cols):
    with rasterio.open(f) as s:
        a=s.read(1).astype("float32"); a[(a==s.nodata)|(a<0)|(a>5000)]=np.nan
        o=np.full((rows,cols),np.nan,"float32")
        reproject(a,o,src_transform=s.transform,src_crs=s.crs,dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear)
    o[(o<0)|(o>5000)]=np.nan; return o

def ymed(y,tr,rows,cols):
    return np.nanmedian(np.stack([load_on(f,tr,rows,cols) for f in byyear[y]]),axis=0)

def stable_uncertainty(dh, st, gl_area, gsd=RES):
    """Self-contained Rolstad(2009)-style error: robust sigma on stable terrain +
    empirical variogram (manual) -> decorrelation range -> Neff for glacier area."""
    m=st&np.isfinite(dh); ys,xs=np.where(m); vals=dh[ys,xs]
    if vals.size<2000: return np.nan,np.nan,1.0,np.nan,"too few stable px"
    rng=np.random.default_rng(0)
    if xs.size>6000:
        idx=rng.choice(xs.size,6000,replace=False); xs,ys,vals=xs[idx],ys[idx],vals[idx]
    co=np.c_[xs*gsd, ys*gsd]
    npair=300000
    i=rng.integers(0,xs.size,npair); j=rng.integers(0,xs.size,npair)
    d=np.hypot(co[i,0]-co[j,0], co[i,1]-co[j,1]); sv=0.5*(vals[i]-vals[j])**2
    edges=np.linspace(0,3000,16); mids=0.5*(edges[:-1]+edges[1:]); gamma=np.full(15,np.nan)
    for b in range(15):
        s=(d>=edges[b])&(d<edges[b+1])
        if s.sum()>100: gamma[b]=np.mean(sv[s])          # mean semivariance (non-robust: captures true variance)
    sill=np.nanmedian(gamma[-4:]) if np.isfinite(gamma[-4:]).any() else np.nanmax(gamma)
    # sigma = sqrt(sill) is the correlated-error std to propagate (NOT the NMAD of the
    # over-smooth composite, which underestimates); floor at single-DEM-scale 0.3 m
    sigma=max(np.sqrt(max(sill,0)), 0.3)
    L=3000.0
    for b in range(15):
        if np.isfinite(gamma[b]) and gamma[b]>=0.95*sill: L=mids[b]; break
    L=max(L,gsd*2)
    neff=max(gl_area/(np.pi*L**2),1.0)   # Rolstad short-range approximation
    sig_mean=sigma/np.sqrt(neff)
    return sigma,L,neff,sig_mean,f"sill^0.5={sigma:.2f}m L={L:.0f}m"

rows_out=[]
for ya,yb in [(years[i],years[i+1]) for i in range(len(years)-1)]:
    fA=best_scene(byyear[yb])
    with rasterio.open(fA) as s: tr=s.transform; rows,cols=s.shape
    A=ymed(ya,tr,rows,cols); B=ymed(yb,tr,rows,cols)
    gm=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
        out_shape=(rows,cols),transform=tr,fill=0,dtype='uint8').astype(bool); st=~gm
    # coreg B->A
    try:
        rA=xdem.DEM.from_array(A,transform=tr,crs=crs,nodata=np.nan)
        rB=xdem.DEM.from_array(B,transform=tr,crs=crs,nodata=np.nan)
        nk=xdem.coreg.NuthKaab(); nk.fit(rA,rB,inlier_mask=st,random_state=42)
        B=nk.apply(rB); B=B.data.filled(np.nan) if hasattr(B.data,'filled') else np.asarray(B.data,float)
    except Exception: pass
    dh=B-A; dh=np.where(np.abs(dh)<=60,dh,np.nan)
    bias=np.nanmedian(dh[st&np.isfinite(dh)]); dh=dh-(bias if np.isfinite(bias) else 0)
    dt=(np.median([date2num(d) for d in dates_by[yb]])-np.median([date2num(d) for d in dates_by[ya]]))/365.25
    gv=dh[gm&np.isfinite(dh)]
    if gv.size<1000 or dt<0.5: continue
    mean_dh=np.mean(gv); rate=mean_dh/dt
    sig_stab,L,neff,sig_mean,method=stable_uncertainty(dh,st,GL_AREA)
    ci=1.96*sig_mean/dt
    tmid=(ya+yb)/2+0.6   # late-summer midpoint
    rows_out.append((ya,yb,tmid,rate,ci,sig_stab,neff,gv.size))
    print(f"{ya}->{yb} (dt={dt:.2f}y): rate={rate:+.2f} ± {ci:.2f} m/yr | "
          f"sig_stab={sig_stab:.2f} m, Neff={neff:.0f} ({gv.size:,} px) [{method}]")

# CSV
with open(AOUT/"ANNUAL_rate_CI.csv","w",newline="") as f:
    w=csv.writer(f,lineterminator="\n"); w.writerow(["year_a","year_b","t_mid","rate_m_yr","ci95_m_yr","sigma_stable_m","Neff","n_px"])
    for r in rows_out: w.writerow([r[0],r[1],f"{r[2]:.2f}",f"{r[3]:.3f}",f"{r[4]:.3f}",f"{r[5]:.2f}",f"{r[6]:.0f}",r[7]])
print("-> ANNUAL_rate_CI.csv")

# ---- figure ----
fig,ax=plt.subplots(figsize=(14,6.5))
for t,lab,major in ERUPT:
    ax.axvspan(t-0.06,t+0.06,color=('firebrick' if major else 'orange'),alpha=0.28 if major else 0.16,zorder=0)
    ax.text(t,0.98,lab,rotation=90,va='top',ha='right',fontsize=8,
            color=('firebrick' if major else 'darkorange'),transform=ax.get_xaxis_transform())
t=[r[2] for r in rows_out]; rate=[r[3] for r in rows_out]; ci=[r[4] for r in rows_out]
ax.axhline(0,color='k',lw=0.7,ls=':')
# CI band + points
ax.errorbar(t,rate,yerr=ci,fmt='o',ms=8,color="#2f7cb2",ecolor="#2f7cb2",elinewidth=1.8,
            capsize=4,capthick=1.6,zorder=4,mfc="#2f7cb2",mec='k',mew=0.5)
ax.plot(t,rate,'-',color="#2f7cb2",lw=1,alpha=0.5,zorder=3)
ax.fill_between(t,[r-c for r,c in zip(rate,ci)],[r+c for r,c in zip(rate,ci)],color="#2f7cb2",alpha=0.12,zorder=1)
# mean rate line
allr=np.array(rate); allw=1/np.array(ci)**2; wmean=np.sum(allr*allw)/np.sum(allw)
ax.axhline(wmean,color="#c0463a",lw=1.5,ls='--',zorder=2,label=f"inverse-variance mean {wmean:+.2f} m/yr")
ax.set_ylabel("glacier–wide elevation–change rate [m yr$^{−1}$]\n(summer→summer, main massif, 95% CI)")
ax.set_xlabel("year")
ax.set_title("Klyuchevskaya main massif — annual elevation–change rate with rigorous 95% CI\n"
             "CI from stable–terrain variogram (Rolstad 2009 / Hugonnet 2022): N_eff, not N_pixels")
ax.legend(loc='lower left',fontsize=10); ax.grid(alpha=0.25)
ax.set_xlim(min(t)-0.6,max(t)+0.6)
fig.tight_layout(); fig.savefig(AOUT/"ANNUAL_rate_CI.png",dpi=600,bbox_inches="tight")
print(f"-> ANNUAL_rate_CI.png  (weighted-mean rate {wmean:+.2f} m/yr, n={len(rows_out)} intervals)")
