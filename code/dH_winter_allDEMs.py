#!/usr/bin/env python3
"""
Glacier elevation change from ALL consistent winter 155_0045 DEMs.
Long-baseline 2-epoch approach (robust): pool every winter 155_0045 DEM into
EARLY vs LATE mean epochs, xdem Nuth&Kaab coregister late->early on stable
terrain, dH = late-early, rate = dH / (mean_late_date - mean_early_date).
Glacier-masked, outlier-clipped. Uses FNL (fallback VER).
"""
import numpy as np, rasterio, rasterio.features, geopandas as gpd
import xdem
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date
import glob, re, csv
from rasterio.warp import reproject, Resampling

BASE=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
OUT =Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out")
GLINV=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b/"
           "2025/2025-01-14_065954_TDT_SM_A_97516_155_0045+1-2/prc05_refdem/"
           "GLINV_rgi06_ttl_utm57_N_b_UTM.shp")
RES=30.0; ND=-9999.0; CORR_M=500.0; crs="EPSG:32657"
SPLIT_YEAR=2020   # early < 2020 <= late

def find_dem(sd):
    for tag in ("DEM_FNL","DEM_VER"):
        g=glob.glob(str(sd/"prc07"/f"{tag}_*.tif"))
        if g: return g[0]
    return None

# ── collect winter 155_0045 DEMs ─────────────────────────────────────────────
scenes=[]
for yd in sorted(BASE.glob("20*")):
    for sd in sorted(yd.glob("20*")):
        nm=sd.name; m=re.match(r"(\d{4})-(\d{2})-(\d{2})",nm)
        if not m: continue
        y,mo,da=map(int,m.groups())
        if not (mo>=11 or mo<=4): continue          # winter
        if "155_0045" not in nm: continue           # same frame only
        f=find_dem(sd)
        if f:
            with rasterio.open(f) as _s:
                _a=_s.read(1); _v=np.sum((_a!=_s.nodata)&np.isfinite(_a)&(_a>-1000)&(_a<9000))
                _h=(_s.bounds.top-_s.bounds.bottom)/1000
            if _v>2.3e6 and _h>100:        # FULL ~115km scenes only (consistent extent)
                scenes.append((date(y,mo,da),f))
early=[(d,f) for d,f in scenes if d.year< SPLIT_YEAR]
late =[(d,f) for d,f in scenes if d.year>=SPLIT_YEAR]
print(f"winter 155_0045 DEMs: {len(scenes)} total")
print(f"  EARLY (<{SPLIT_YEAR}): {len(early)}  {[d.isoformat() for d,_ in early]}")
print(f"  LATE  (>={SPLIT_YEAR}): {len(late)}  {[d.isoformat() for d,_ in late]}")

# ── common grid (intersection of all) ────────────────────────────────────────
bnds=[]
for _,f in scenes:
    with rasterio.open(f) as s: bnds.append(s.bounds)
xmin=max(b.left for b in bnds); xmax=min(b.right for b in bnds)
ymin=max(b.bottom for b in bnds); ymax=min(b.top for b in bnds)
xmin=np.ceil(xmin/RES)*RES; xmax=np.floor(xmax/RES)*RES
ymin=np.ceil(ymin/RES)*RES; ymax=np.floor(ymax/RES)*RES
cols=int((xmax-xmin)/RES); rows=int((ymax-ymin)/RES)
transform=rasterio.transform.from_origin(xmin,ymax,RES,RES)
print(f"common grid {rows}x{cols} ({(xmax-xmin)/1e3:.0f}x{(ymax-ymin)/1e3:.0f} km)")

def read_grid(f):
    with rasterio.open(f) as s:
        src=s.read(1).astype("float32")
        src[(src==s.nodata)|(src<-1000)|(src>9000)]=np.nan
        dst=np.full((rows,cols),np.nan,"float32")
        reproject(src,dst,src_transform=s.transform,src_crs=s.crs,
                  dst_transform=transform,dst_crs=crs,
                  resampling=Resampling.bilinear,src_nodata=np.nan,dst_nodata=np.nan)
    dst[(dst<-1000)|(dst>9000)]=np.nan
    return dst

def epoch_mean(lst):
    return np.nanmedian(np.stack([read_grid(f) for _,f in lst]),axis=0)

early_dem=epoch_mean(early); late_dem=epoch_mean(late)

# ── glacier mask ─────────────────────────────────────────────────────────────
shp=gpd.read_file(GLINV).to_crs(crs)
gmask=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
        out_shape=(rows,cols),transform=transform,fill=0,dtype="uint8").astype(bool)
stable=~gmask

# ── coregister late -> early (Nuth&Kaab on stable terrain) ───────────────────
refD=xdem.DEM.from_array(early_dem,transform=transform,crs=crs,nodata=np.nan)
tbaD=xdem.DEM.from_array(late_dem, transform=transform,crs=crs,nodata=np.nan)
try:
    nk=xdem.coreg.NuthKaab(); nk.fit(refD,tbaD,inlier_mask=stable,random_state=42)
    late_al=nk.apply(tbaD)
    late_dem=late_al.data.filled(np.nan) if hasattr(late_al.data,"filled") else np.asarray(late_al.data,float)
    print("  coregistered late->early (Nuth&Kaab)")
except Exception as e:
    print(f"  coreg failed ({e}); median-shift fallback")
dh=late_dem-early_dem
b=stable&np.isfinite(dh)
dh=dh-np.nanmedian(dh[b])           # zero stable terrain

# ── time baseline ────────────────────────────────────────────────────────────
def mean_ord(lst): return np.mean([d.toordinal() for d,_ in lst])
dt=(mean_ord(late)-mean_ord(early))/365.25
print(f"  baseline dt = {dt:.2f} yr")

# ── rate + outlier clip (3*NMAD on glacier) ──────────────────────────────────
rate=dh/dt
g=rate[gmask&np.isfinite(rate)]
med=np.median(g); nmad=1.4826*np.median(np.abs(g-med))
lo,hi=med-3*nmad,med+3*nmad
rate_cl=np.where(np.isfinite(rate)&(rate>=lo)&(rate<=hi),rate,np.nan)

# ── save ─────────────────────────────────────────────────────────────────────
prof=dict(driver="GTiff",height=rows,width=cols,count=1,dtype="float32",
          crs=crs,transform=transform,nodata=ND,compress="lzw")
def save(a,n):
    with rasterio.open(OUT/n,"w",**prof) as d: d.write(np.where(np.isfinite(a),a,ND).astype("float32"),1)
save(rate_cl,"winter_allDEMs_rate_m_per_yr.tif")
save(np.where(gmask,rate_cl,np.nan),"winter_allDEMs_rate_glacieronly.tif")

gg=rate_cl[gmask&np.isfinite(rate_cl)]
n_eff=max(1.0,gg.size*RES*RES/(np.pi*CORR_M**2))
err=(1.4826*np.median(np.abs(gg-np.median(gg))))/np.sqrt(n_eff)
print(f"\n=== WINTER GLACIER RATE (all {len(scenes)} winter 155_0045 DEMs) ===")
print(f"  early {early[0][0].year}-{early[-1][0].year} ({len(early)}) vs late {late[0][0].year}-{late[-1][0].year} ({len(late)}), dt={dt:.1f}yr")
print(f"  median = {np.median(gg):+.3f} +/- {err:.3f} m/yr   mean = {np.mean(gg):+.3f}   px={gg.size:,}")

# preview map + hist
vlim=np.nanpercentile(np.abs(gg),98)
fig,(ax,axh)=plt.subplots(1,2,figsize=(15,8),gridspec_kw={"width_ratios":[1.4,1]})
im=ax.imshow(np.where(gmask,rate_cl,np.nan),cmap="RdBu",vmin=-vlim,vmax=vlim,extent=[xmin,xmax,ymin,ymax])
plt.colorbar(im,ax=ax,shrink=0.8,label="Elevation-change rate [m yr$^{-1}$]")
ax.set_title(f"Klyuchevskaya winter glaciers {early[0][0].year}-{late[-1][0].year}\n"
             f"all {len(scenes)} winter 155_0045 DEMs, median {np.median(gg):+.2f} m/yr (blue=gain)")
ax.set_xlabel("UTM57N Easting [m]"); ax.set_ylabel("Northing [m]")
axh.hist(np.clip(gg,-vlim,vlim),bins=70,color="0.5")
axh.axvline(np.median(gg),color="r",ls="--",label=f"median {np.median(gg):+.2f}")
axh.set_xlabel("rate [m/yr]"); axh.set_ylabel("glacier px"); axh.legend()
axh.set_title("distribution")
fig.savefig(OUT/"PREVIEW_winter_allDEMs_rate.png",dpi=130,bbox_inches="tight")
print("\n-> winter_allDEMs_rate_m_per_yr.tif, _glacieronly.tif, PREVIEW_winter_allDEMs_rate.png")
