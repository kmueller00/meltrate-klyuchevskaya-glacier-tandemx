#!/usr/bin/env python3
"""
STEP 15: All-ascending glacier elevation series 2012-2025 with
  (a) per-season outlier rejection (a point is dropped if it deviates from the
      other observations OF ITS OWN SEASON: |residual vs the season's rolling
      median| > 3*NMAD of the season's residuals),
  (b) TWO trend depictions: simple linear fit per season (slope +/- 95% CI) and
      a penalized smoothing spline (GCV-selected lambda; falls back to LOWESS) -
      a standard descriptive non-parametric smoother,
  (c) main-massif glacier mask only (southern tongue-only glaciers excluded).
Also writes the per-date glacier-median series to CSV for downstream use
(climate comparison in step 16).
"""
import glob, re, numpy as np, rasterio, rasterio.features, geopandas as gpd, xdem
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import date2num
from pathlib import Path
from datetime import date
from collections import defaultdict
from rasterio.warp import reproject, Resampling
import csv

BASE=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation")
GLINV=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/ash_analysis/GLINV_main_massif.gpkg")
crs="EPSG:32657"

ERUPT=[(date(2013,1,25),date(2013,11,30),"2013 MAJOR",1),
 (date(2015,1,2),date(2015,3,24),"2015",0),(date(2015,8,27),date(2015,9,10),"2015 Aug",0),
 (date(2016,4,1),date(2016,5,1),"2016",0),(date(2019,4,1),date(2019,11,1),"2019",1),
 (date(2020,10,2),date(2021,3,25),"2020-21",1),(date(2022,11,20),date(2022,12,20),"2022",0),
 (date(2023,6,22),date(2024,1,15),"2023-24 PAROXYSM",1),(date(2024,8,1),date(2024,10,15),"2024",0)]

def find(sd):
    for t in ("DEM_FNL","DEM_VER"):
        g=glob.glob(str(sd/"prc07"/f"{t}_*.tif"))
        if g: return g[0]

dems=[]
for sd in [Path(p) for p in glob.glob(str(BASE/"20*/20*"))]:
    nm=sd.name; m=re.match(r"(\d{4})-(\d{2})-(\d{2})",nm)
    if not m or "155_0045" not in nm: continue
    d=date(*map(int,m.groups()))
    if d.month==5 or d.month==10: continue          # shoulder months: ambiguous season
    f=find(sd)
    if not f: continue
    with rasterio.open(f) as s:
        a=s.read(1).astype(float); a[(a==s.nodata)|(a<0)|(a>5000)]=np.nan; v=a[np.isfinite(a)]
    if v.size<3e5 or not (800<np.median(v)<2500): continue
    dems.append((d,f))
dems=sorted(set(dems))
print(f"usable ascending DEMs 2012-2025: {len(dems)}")

ref_f=[f for d,f in dems if d.year==2025 and d.month in (1,2,3)][0]
with rasterio.open(ref_f) as s: tr=s.transform; rows,cols=s.shape
def rg(f):
    with rasterio.open(f) as s:
        a=s.read(1).astype("float32"); a[(a==s.nodata)|(a<0)|(a>5000)]=np.nan
        d=np.full((rows,cols),np.nan,"float32")
        reproject(a,d,src_transform=s.transform,src_crs=s.crs,dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear)
    d[(d<0)|(d>5000)]=np.nan; return d
shp=gpd.read_file(GLINV).to_crs(crs)
gm=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
    out_shape=(rows,cols),transform=tr,fill=0,dtype="uint8").astype(bool)
st=~gm
ref=rg(ref_f); refD=xdem.DEM.from_array(ref,transform=tr,crs=crs,nodata=np.nan)

series=[]
for d,f in dems:
    try:
        dem=rg(f)
    except Exception as e:
        print(f"  SKIP {d} (read error: {e})"); continue
    try:
        tD=xdem.DEM.from_array(dem,transform=tr,crs=crs,nodata=np.nan)
        nk=xdem.coreg.NuthKaab(); nk.fit(refD,tD,inlier_mask=st,random_state=42)
        dem=nk.apply(tD); dem=dem.data.filled(np.nan) if hasattr(dem.data,'filled') else np.asarray(dem.data,float)
    except Exception: pass
    dh=dem-ref; dh=np.where(np.abs(dh)<=50,dh,np.nan)
    b=np.nanmedian(dh[st&np.isfinite(dh)]); dh=dh-(b if np.isfinite(b) else 0)
    gv=dh[gm&np.isfinite(dh)]
    if gv.size<500: continue
    stv=dh[st&np.isfinite(dh)]
    nmad=1.4826*np.nanmedian(np.abs(stv-np.nanmedian(stv))) if stv.size>500 else 999
    anom=np.nanmedian(gv)
    if nmad>8 or abs(anom)>15: continue            # corrupt-DEM gate (8 m: 2nd-stage outlier rejection below handles residuals)
    seas="W" if (d.month>=11 or d.month<=4) else "S"
    series.append((d,anom,seas))

# merge sub-scenes per date
bd=defaultdict(list)
for d,a,se in series: bd[d].append((a,se))
series=sorted([(d,float(np.median([a for a,_ in v])),v[0][1]) for d,v in bd.items()])
print(f"date-points after merge: {len(series)}")

# ---- PER-SEASON OUTLIER REJECTION: deviate from own-season rolling median ----
def season_outliers(pts, w=7, k=3.0):
    """pts=[(datenum,anom)]; returns keep-mask via residuals from rolling median."""
    if len(pts)<5: return [True]*len(pts)
    x=np.array([p[0] for p in pts]); y=np.array([p[1] for p in pts])
    rm=np.array([np.median(y[max(0,i-w//2):min(len(y),i+w//2+1)]) for i in range(len(y))])
    res=y-rm
    nm=1.4826*np.median(np.abs(res-np.median(res)))
    return list(np.abs(res-np.median(res))<=k*max(nm,0.3))

W=[(date2num(d),a,d) for d,a,se in series if se=="W"]
S=[(date2num(d),a,d) for d,a,se in series if se=="S"]
kW=season_outliers([(x,a) for x,a,_ in W]); kS=season_outliers([(x,a) for x,a,_ in S])
W_in=[p for p,k in zip(W,kW) if k]; W_out=[p for p,k in zip(W,kW) if not k]
S_in=[p for p,k in zip(S,kS) if k]; S_out=[p for p,k in zip(S,kS) if not k]
print(f"winter: {len(W_in)} kept, {len(W_out)} outliers removed")
print(f"summer: {len(S_in)} kept, {len(S_out)} outliers removed")
for x,a,d in W_out+S_out: print(f"  OUTLIER {d}: {a:+.2f} m")

# CSV for downstream climate step
with open(AOUT/"glacier_median_series.csv","w",newline="") as fo:
    wcsv=csv.writer(fo,lineterminator="\n"); wcsv.writerow(["date","season","anom_m","kept"])
    for x,a,d in W_in: wcsv.writerow([d.isoformat(),"W",f"{a:.3f}",1])
    for x,a,d in W_out: wcsv.writerow([d.isoformat(),"W",f"{a:.3f}",0])
    for x,a,d in S_in: wcsv.writerow([d.isoformat(),"S",f"{a:.3f}",1])
    for x,a,d in S_out: wcsv.writerow([d.isoformat(),"S",f"{a:.3f}",0])
print("-> glacier_median_series.csv")

# ---- fits ----
def linfit(pts):
    x=np.array([p[0] for p in pts])/365.25; y=np.array([p[1] for p in pts])
    n=len(x); A=np.vstack([x,np.ones(n)]).T
    coef,res,_,_=np.linalg.lstsq(A,y,rcond=None)
    yhat=A@coef; s2=np.sum((y-yhat)**2)/(n-2)
    sx=np.sqrt(s2/np.sum((x-x.mean())**2))
    from scipy import stats
    ci=stats.t.ppf(0.975,n-2)*sx
    return coef[0],ci,coef  # slope m/yr, 95% CI, (slope,intercept in yr-units)

def spline_fit(pts):
    x=np.array([p[0] for p in pts]); y=np.array([p[1] for p in pts])
    o=np.argsort(x); x,y=x[o],y[o]
    xu,idx=np.unique(x,return_index=True); yu=y[idx]
    try:
        from scipy.interpolate import make_smoothing_spline
        sp=make_smoothing_spline(xu,yu)                 # GCV-selected lambda
        xs=np.linspace(xu[0],xu[-1],400); return xs,sp(xs),"penalized smoothing spline (GCV)"
    except Exception:
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            sm=lowess(yu,xu,frac=0.35,return_sorted=True)
            return sm[:,0],sm[:,1],"LOWESS (frac=0.35)"
        except Exception:
            from scipy.interpolate import UnivariateSpline
            sp=UnivariateSpline(xu,yu,s=len(xu)*np.var(yu)*0.5)
            xs=np.linspace(xu[0],xu[-1],400); return xs,sp(xs),"smoothing spline"

COL={"S":"#e0662a","W":"#1f77b4"}
def base_ax(ax):
    for s,e,lab,major in ERUPT:
        ax.axvspan(date2num(s),date2num(e),color=('firebrick' if major else 'orange'),
                   alpha=0.22 if major else 0.11,zorder=0)
    ax.axhline(0,color='k',lw=0.5,ls=':')
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(date2num(date(2012,1,1)),date2num(date(2026,3,1)))
    ax.set_ylabel("glacier–median elevation anomaly [m]\n(vs 2025 ref; main massif only)")

for MODE in ("linear","spline"):
    fig,ax=plt.subplots(figsize=(15,6)); base_ax(ax)
    for pts_in,pts_out,se,lab in [(S_in,S_out,"S","summer (surface)"),(W_in,W_out,"W","winter (penetration-biased)")]:
        if pts_in:
            ax.scatter([p[0] for p in pts_in],[p[1] for p in pts_in],s=55,c=COL[se],
                       marker='^' if se=="S" else 'o',edgecolor='k',lw=0.4,zorder=3,label=lab)
        if pts_out:
            ax.scatter([p[0] for p in pts_out],[p[1] for p in pts_out],s=55,facecolor='none',
                       marker='^' if se=="S" else 'o',edgecolor=COL[se],lw=1.2,zorder=3,
                       label=f"removed outlier ({'summer' if se=='S' else 'winter'})")
        if len(pts_in)>=5:
            if MODE=="linear":
                sl,ci,coef=linfit(pts_in)
                xs=np.array([min(p[0] for p in pts_in),max(p[0] for p in pts_in)])
                ax.plot(xs,coef[0]*xs/365.25+coef[1],'-',color=COL[se],lw=2.2,zorder=2)
                ax.text(0.99,0.97 if se=="S" else 0.90,
                        f"{'summer' if se=='S' else 'winter'} trend: {sl:+.2f} ± {ci:.2f} m/yr",
                        transform=ax.transAxes,ha='right',va='top',fontsize=10,color=COL[se],fontweight='bold')
            else:
                xs,ys,meth=spline_fit(pts_in)
                ax.plot(xs,ys,'-',color=COL[se],lw=2.2,zorder=2)
    if MODE=="linear":
        ax.set_title("Klyuchevskaya main massif — glacier elevation 2012–2025, LINEAR trend per season\n"
                     "outliers (open symbols) rejected within–season (3×NMAD vs rolling median); ± = 95% CI")
        out=AOUT/"TREND_linear_2012-2025.png"
    else:
        ax.set_title(f"Klyuchevskaya main massif — glacier elevation 2012–2025, SMOOTHING SPLINE per season\n"
                     f"descriptive non–parametric smoother ({spline_fit(S_in)[2] if len(S_in)>4 else ''}); same outlier removal")
        out=AOUT/"TREND_spline_2012-2025.png"
    ax.legend(loc='lower left',fontsize=9)
    fig.tight_layout(); fig.savefig(out,dpi=600,bbox_inches="tight")
    print(f"-> {out.name}")