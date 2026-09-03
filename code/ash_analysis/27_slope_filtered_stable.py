#!/usr/bin/env python3
import sys; sys.path.insert(0,"/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/code/ash_analysis"); import _style
"""
STEP 27: Slope-filtered stable terrain for co-registration.

Literature basis (see project memory for full citations):
  - STEEP exclusion: >40 deg is the standard cutoff (Hugonnet et al. 2022 /
    xdem convention; Berthier et al. 2016 used 45 deg). Elevation-difference
    dispersion grows strongly with slope (vertical error ~ tan(slope) x
    horizontal error), so steep pixels inflate/bias both the coregistration
    fit and the stable-terrain sigma used for our CIs.
  - FLAT exclusion: Nuth & Kaab (2011) solves dh/tan(slope) vs aspect; this
    is UNDEFINED as slope->0 (no aspect on flat ground), so near-flat pixels
    carry no horizontal-shift information and destabilise the fit. Standard
    implementations (demcompare, xdem) filter slopes below a few degrees.
    We use <2 deg as the flat cutoff (conservative low end of the ~2-5 deg
    range used in practice).

This script rebuilds the ANNUAL_rate_CI series (step 20) with a
slope-filtered stable-terrain mask (2 deg <= slope <= 40 deg) instead of the
"everything off-glacier" mask used previously, and reports how much the
per-year rate and its CI change -- i.e. whether slope-selection materially
affects the trend/CI, or just tightens the stable-terrain sigma.
"""
import glob, re, numpy as np, rasterio, rasterio.features, geopandas as gpd, xdem
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from rasterio.warp import reproject, Resampling
import csv, warnings
warnings.filterwarnings("ignore")

BASE=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation")
GLINV=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/code/ash_analysis/GLINV_main_massif.gpkg")
crs="EPSG:32657"; RES=30.0
SLOPE_LO,SLOPE_HI=2.0,40.0   # literature-backed stable-terrain slope band [deg]

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
        reproject(a,o,src_transform=s.transform,src_crs=s.crs,dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear)
    o[(o<0)|(o>5000)]=np.nan; return o

def ymed(y,tr,rows,cols):
    return np.nanmedian(np.stack([load_on(f,tr,rows,cols) for f in byyear[y]]),axis=0)

def stable_uncertainty(dh, st, gl_area, gsd=RES):
    m=st&np.isfinite(dh); ys,xs=np.where(m); vals=dh[ys,xs]
    if vals.size<2000: return np.nan,np.nan,1.0,np.nan
    rng=np.random.default_rng(0)
    if xs.size>6000:
        idx=rng.choice(xs.size,6000,replace=False); xs,ys,vals=xs[idx],ys[idx],vals[idx]
    co=np.c_[xs*gsd, ys*gsd]; npair=300000
    i=rng.integers(0,xs.size,npair); j=rng.integers(0,xs.size,npair)
    d=np.hypot(co[i,0]-co[j,0], co[i,1]-co[j,1]); sv=0.5*(vals[i]-vals[j])**2
    edges=np.linspace(0,3000,16); mids=0.5*(edges[:-1]+edges[1:]); gamma=np.full(15,np.nan)
    for b in range(15):
        s=(d>=edges[b])&(d<edges[b+1])
        if s.sum()>100: gamma[b]=np.mean(sv[s])
    sill=np.nanmedian(gamma[-4:]) if np.isfinite(gamma[-4:]).any() else np.nanmax(gamma)
    sigma=max(np.sqrt(max(sill,0)), 0.3)
    L=3000.0
    for b in range(15):
        if np.isfinite(gamma[b]) and gamma[b]>=0.95*sill: L=mids[b]; break
    L=max(L,gsd*2)
    neff=max(gl_area/(np.pi*L**2),1.0)
    return sigma,L,neff,sigma/np.sqrt(neff)

shp=gpd.read_file(GLINV).to_crs(crs)
GL_AREA=shp.geometry.area.sum()
fA0=best_scene(byyear[years[len(years)//2]])
with rasterio.open(fA0) as s: tr=s.transform; rows,cols=s.shape
gm=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
    out_shape=(rows,cols),transform=tr,fill=0,dtype='uint8').astype(bool)
st_all=~gm
Zref=load_on(fA0,tr,rows,cols)
gy,gx=np.gradient(Zref,RES)
slope_deg=np.degrees(np.arctan(np.hypot(gx,gy)))
st_slope=st_all & (slope_deg>=SLOPE_LO) & (slope_deg<=SLOPE_HI)
print(f"stable terrain: {st_all.sum():,} px unfiltered -> {st_slope.sum():,} px slope-filtered "
      f"({100*st_slope.sum()/max(st_all.sum(),1):.1f}% retained, band {SLOPE_LO}-{SLOPE_HI} deg)")

rows_out=[]
for ya,yb in [(years[i],years[i+1]) for i in range(len(years)-1)]:
    fA=best_scene(byyear[ya])
    with rasterio.open(fA) as s: tr2=s.transform; rows2,cols2=s.shape
    A=ymed(ya,tr2,rows2,cols2); B=ymed(yb,tr2,rows2,cols2)
    gm2=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
        out_shape=(rows2,cols2),transform=tr2,fill=0,dtype='uint8').astype(bool)
    Zref2=A.copy(); gy2,gx2=np.gradient(Zref2,RES); slope2=np.degrees(np.arctan(np.hypot(gx2,gy2)))
    st2_all=~gm2; st2_slope=st2_all&(slope2>=SLOPE_LO)&(slope2<=SLOPE_HI)
    dt=(yb-ya)
    results={}
    for tag,stmask in [("unfiltered",st2_all),("slope_filtered",st2_slope)]:
        try:
            rA=xdem.DEM.from_array(A,transform=tr2,crs=crs,nodata=np.nan)
            rB=xdem.DEM.from_array(B,transform=tr2,crs=crs,nodata=np.nan)
            nk=xdem.coreg.NuthKaab(); nk.fit(rA,rB,inlier_mask=stmask,random_state=42)
            Bc=nk.apply(rB); Bc=Bc.data.filled(np.nan) if hasattr(Bc.data,'filled') else np.asarray(Bc.data,float)
        except Exception:
            Bc=B.copy()
        dh=Bc-A; dh=np.where(np.abs(dh)<=60,dh,np.nan)
        bias=np.nanmedian(dh[stmask&np.isfinite(dh)]); dh=dh-(bias if np.isfinite(bias) else 0)
        gv=dh[gm2&np.isfinite(dh)]
        rate=np.mean(gv)/dt if gv.size>1000 else np.nan
        sigma,L,neff,sig_mean=stable_uncertainty(dh,stmask,GL_AREA)
        ci=1.96*sig_mean/dt if np.isfinite(sig_mean) else np.nan
        results[tag]=(rate,ci,sigma)
    print(f"{ya}->{yb}: unfiltered rate={results['unfiltered'][0]:+.2f}±{results['unfiltered'][1]:.2f} "
          f"(sigma_stable={results['unfiltered'][2]:.2f}m) | "
          f"slope-filtered rate={results['slope_filtered'][0]:+.2f}±{results['slope_filtered'][1]:.2f} "
          f"(sigma_stable={results['slope_filtered'][2]:.2f}m)")
    rows_out.append(dict(year_a=ya,year_b=yb,
        rate_unfilt=results['unfiltered'][0],ci_unfilt=results['unfiltered'][1],sigma_unfilt=results['unfiltered'][2],
        rate_filt=results['slope_filtered'][0],ci_filt=results['slope_filtered'][1],sigma_filt=results['slope_filtered'][2]))

with open(AOUT/"SLOPE_FILTER_comparison.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows_out[0].keys()),lineterminator="\n")
    w.writeheader()
    for r in rows_out: w.writerow(r)
print("-> SLOPE_FILTER_comparison.csv")

# ---- figure: slope map + rate comparison ----
fig,axes=plt.subplots(1,2,figsize=(14,6))
sl_show=np.where(gm,np.nan,slope_deg)
im=axes[0].imshow(sl_show,cmap='viridis',vmin=0,vmax=60)
axes[0].contour((slope_deg>=SLOPE_LO)&(slope_deg<=SLOPE_HI),levels=[.5],colors='red',linewidths=1)
axes[0].set_title(f"Off-glacier slope [deg]\nred contour = stable-terrain band used ({SLOPE_LO}-{SLOPE_HI} deg)")
plt.colorbar(im,ax=axes[0],shrink=0.7,label="slope [deg]")
mid=[(r['year_a']+r['year_b'])/2 for r in rows_out]
axes[1].errorbar(mid,[r['rate_unfilt'] for r in rows_out],yerr=[r['ci_unfilt'] for r in rows_out],
                  fmt='o-',color="#2f7cb2",capsize=3,label="unfiltered stable terrain")
axes[1].errorbar([m+0.1 for m in mid],[r['rate_filt'] for r in rows_out],yerr=[r['ci_filt'] for r in rows_out],
                  fmt='s-',color="#c0463a",capsize=3,label=f"slope-filtered ({SLOPE_LO}-{SLOPE_HI} deg)")
axes[1].axhline(0,color='k',lw=0.7,ls=':')
axes[1].set_xlabel("year"); axes[1].set_ylabel("rate [m/yr]")
axes[1].set_title("Annual rate: unfiltered vs slope-filtered stable terrain")
axes[1].legend(fontsize=9)
fig.suptitle("Effect of slope-filtering the stable-terrain mask on co-registration & rate CIs",fontsize=13)
fig.tight_layout(); fig.savefig(AOUT/"SLOPE_FILTER_comparison.png",dpi=600,bbox_inches="tight")
print("-> SLOPE_FILTER_comparison.png")
