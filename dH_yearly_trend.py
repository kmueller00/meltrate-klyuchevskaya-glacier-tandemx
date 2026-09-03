#!/usr/bin/env python3
"""
Per-year mean DEM -> linear elevation-change TREND (m/yr) over glacier area.
Uses ALL available DEMs (DEM_FNL, fallback DEM_VER), winter scenes only
(Nov-Apr) to avoid seasonal snow bias. Coregistration-corrected per the
existing Nuth&Kaab stable-terrain approach is NOT re-done here; instead we
median-difference each yearly mean to a common reference grid and fit a
robust per-pixel linear trend over the yearly means.

Outputs (in prc07_overview_out):
  yearly_trend_m_per_yr.tif             full grid
  yearly_trend_m_per_yr_glacieronly.tif glacier pixels only
  yearly_trend_preview.png
  yearly_trend_stats.csv                per-year n + glacier-mean elevation
"""
import numpy as np, rasterio, rasterio.warp, rasterio.features, geopandas as gpd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date
import glob, csv, re, os

BASE = Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
OUT  = Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out")
GLINV= Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b/"
            "2025/2025-01-14_065954_TDT_SM_A_97516_155_0045+1-2/prc05_refdem/"
            "GLINV_rgi06_ttl_utm57_N_b_UTM.shp")
RES=30.0; ND=-9999.0; CORR_M=500.0
WINTER_ONLY=True   # Nov-Apr; avoids summer melt/penetration mixing

# ── discover all DEMs: prefer FNL, fallback VER ──────────────────────────────
def find_dem(scene_dir):
    for tag in ("DEM_FNL","DEM_VER"):
        g=glob.glob(str(scene_dir/"prc07"/f"{tag}_*.tif"))
        if g: return g[0],tag
    return None,None

scenes=[]
for yr_dir in sorted(BASE.glob("20*")):
    for sd in sorted(yr_dir.glob("20*")):
        nm=sd.name; m=re.match(r"(\d{4})-(\d{2})-(\d{2})",nm)
        if not m: continue
        y,mo,da=map(int,m.groups())
        if WINTER_ONLY and not (mo>=11 or mo<=4): continue
        if "155_0045" not in nm: continue          # the target relorbit only
        f,tag=find_dem(sd)
        if f: scenes.append((date(y,mo,da),f,tag,nm))
print(f"found {len(scenes)} winter DEMs (155_0045) with FNL/VER")

# ── keep only WELL-SAMPLED years (>= MIN_SCENES_PER_YEAR scenes) ─────────────
# sparse single-scene years give noisy per-year means; restrict to robust years.
MIN_SCENES_PER_YEAR=4
from collections import Counter
yc_count=Counter(dt.year for dt,_,_,_ in scenes)
keep_years={y for y,c in yc_count.items() if c>=MIN_SCENES_PER_YEAR}
scenes=[s for s in scenes if s[0].year in keep_years]
print(f"  well-sampled years (>={MIN_SCENES_PER_YEAR} scenes): {sorted(keep_years)}  -> {len(scenes)} DEMs")
assert len(keep_years)>=2, "need >=2 well-sampled years to compute a rate"

# ── common grid = union->intersection of all, at 30 m, EPSG:32657 ────────────
import rasterio.windows
crs="EPSG:32657"
bnds=[]
for _,f,_,_ in scenes:
    with rasterio.open(f) as s: bnds.append(s.bounds)
xmin=max(b.left for b in bnds); xmax=min(b.right for b in bnds)
ymin=max(b.bottom for b in bnds); ymax=min(b.top for b in bnds)
xmin=np.ceil(xmin/RES)*RES; xmax=np.floor(xmax/RES)*RES
ymin=np.ceil(ymin/RES)*RES; ymax=np.floor(ymax/RES)*RES
cols=int((xmax-xmin)/RES); rows=int((ymax-ymin)/RES)
transform=rasterio.transform.from_origin(xmin,ymax,RES,RES)
print(f"common grid: {rows}x{cols}  ({(xmax-xmin)/1000:.0f}x{(ymax-ymin)/1000:.0f} km)")

def read_grid(f):
    with rasterio.open(f) as s:
        from rasterio.warp import reproject, Resampling
        # pre-mask nodata to NaN in source BEFORE resampling (avoids bilinear
        # mixing valid elevations with the huge -3.4e38 nodata -> garbage)
        src=s.read(1).astype("float32")
        src[(src==s.nodata)|(src<-1000)|(src>9000)]=np.nan
        dst=np.full((rows,cols),np.nan,dtype="float32")
        reproject(src,dst,
                  src_transform=s.transform,src_crs=s.crs,
                  dst_transform=transform,dst_crs=crs,
                  resampling=Resampling.bilinear,
                  src_nodata=np.nan,dst_nodata=np.nan)
    # final guard: any out-of-range -> NaN
    dst[(dst<-1000)|(dst>9000)]=np.nan
    return dst

# ── per-year mean DEM ────────────────────────────────────────────────────────
from collections import defaultdict
byyear=defaultdict(list)
for dt,f,tag,nm in scenes: byyear[dt.year].append(f)
years=sorted(byyear)
year_means={}
for y in years:
    stack=np.stack([read_grid(f) for f in byyear[y]])
    year_means[y]=np.nanmean(stack,axis=0)
    print(f"  {y}: {len(byyear[y])} scene(s)")

# ── glacier mask ─────────────────────────────────────────────────────────────
shp=gpd.read_file(GLINV).to_crs(crs)
gmask=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
        out_shape=(rows,cols),transform=transform,fill=0,dtype="uint8").astype(bool)

# ── COREGISTER each yearly mean to a reference (xdem Nuth & Kääb on stable terrain)
# Proper 3-D alignment (x/y/z) — removes the inter-DEM offsets that otherwise
# swamp the real glacier signal. Reference = year with most scenes.
import xdem, geoutils as gu
ref_year=max(years, key=lambda y: len(byyear[y]))
stable=~gmask
ref_dem=xdem.DEM.from_array(year_means[ref_year], transform=transform, crs=crs, nodata=np.nan)
for y in years:
    if y==ref_year: continue
    try:
        tba=xdem.DEM.from_array(year_means[y], transform=transform, crs=crs, nodata=np.nan)
        nk=xdem.coreg.NuthKaab()
        nk.fit(ref_dem, tba, inlier_mask=stable, random_state=42)
        aligned=nk.apply(tba)
        year_means[y]=aligned.data.filled(np.nan) if hasattr(aligned.data,'filled') else np.asarray(aligned.data,dtype=float)
        # residual vertical bias on stable terrain -> 0
        dh=year_means[y]-year_means[ref_year]; m=stable&np.isfinite(dh)
        if m.sum()>1000: year_means[y]=year_means[y]-np.nanmedian(dh[m])
    except Exception as e:
        print(f"    {y}: coreg failed ({e}); falling back to median shift")
        dh=year_means[y]-year_means[ref_year]; m=stable&np.isfinite(dh)
        if m.sum()>1000: year_means[y]=year_means[y]-np.nanmedian(dh[m])
print(f"  coregistered all years to {ref_year} (Nuth&Kaab on stable terrain)")

# ── per-pixel rate (m/yr) over the well-sampled yearly means ─────────────────
# With only well-sampled years (here 2024,2025) require EVERY kept year present
# per pixel, then rate = linear slope (2 yrs -> simple difference / dt).
MIN_YEARS=len(years)   # require all kept (well-sampled) years at each pixel
RATE_CLIP=20.0   # clip +/-20 m/yr: keeps real volcanic/dynamic change, cuts noise
yr_arr=np.array(years,dtype=float)
Y=np.stack([year_means[y] for y in years])          # (nyear, rows, cols)
valid_count=np.sum(np.isfinite(Y),axis=0)
trend=np.full((rows,cols),np.nan,dtype="float32")
yc=yr_arr-yr_arr.mean()
for i in range(rows):
    col=Y[:,i,:]                                    # (nyear, cols)
    m=np.isfinite(col)
    cnt=m.sum(axis=0)
    ok=cnt>=MIN_YEARS                               # require all kept years
    if not ok.any(): continue
    cc=np.where(m,col,0.0); ww=m.astype(float)
    sw=ww.sum(0); swx=(ww*yc[:,None]).sum(0); swy=(ww*cc).sum(0)
    swxx=(ww*yc[:,None]**2).sum(0); swxy=(ww*yc[:,None]*cc).sum(0)
    denom=sw*swxx-swx**2
    sl=np.where((denom!=0)&ok,(sw*swxy-swx*swy)/np.where(denom==0,1,denom),np.nan)
    trend[i,:]=sl

print(f"  pixels with >={MIN_YEARS} yrs: {np.sum((valid_count>=MIN_YEARS)&gmask):,} glacier")

# ── hard clip to +/-RATE_CLIP m/yr (physical glacier range) ──────────────────
lo,hi=-RATE_CLIP,RATE_CLIP
trend_cl=np.where(np.isfinite(trend)&(trend>=lo)&(trend<=hi),trend,np.nan)

# ── save ─────────────────────────────────────────────────────────────────────
prof=dict(driver="GTiff",height=rows,width=cols,count=1,dtype="float32",
          crs=crs,transform=transform,nodata=ND,compress="lzw")
def save(arr,name):
    with rasterio.open(OUT/name,"w",**prof) as d:
        d.write(np.where(np.isfinite(arr),arr,ND).astype("float32"),1)
save(trend_cl,"yearly_trend_m_per_yr.tif")
save(np.where(gmask,trend_cl,np.nan),"yearly_trend_m_per_yr_glacieronly.tif")

gg=trend_cl[gmask&np.isfinite(trend_cl)]
n_eff=max(1.0,gg.size*RES*RES/(np.pi*CORR_M**2))
err=(1.4826*np.median(np.abs(gg-np.median(gg))))/np.sqrt(n_eff)
print(f"\n=== GLACIER TREND ({years[0]}-{years[-1]}, {len(scenes)} DEMs, {len(years)} yrs) ===")
print(f"  median rate = {np.median(gg):+.3f} +/- {err:.3f} m/yr")
print(f"  mean   rate = {np.mean(gg):+.3f} m/yr   glacier px={gg.size:,}")

# stats csv
with open(OUT/"yearly_trend_stats.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["year","n_scenes","glacier_mean_elev_m"])
    for y in years:
        gm=year_means[y][gmask]; gm=gm[np.isfinite(gm)]
        w.writerow([y,len(byyear[y]),round(float(np.mean(gm)),2) if gm.size else ""])

# preview: map + histogram side by side so the distribution is visible
vlim=min(RATE_CLIP, np.nanpercentile(np.abs(gg),98))
fig,(ax,axh)=plt.subplots(1,2,figsize=(15,8),gridspec_kw={"width_ratios":[1.4,1]})
im=ax.imshow(np.where(gmask,trend_cl,np.nan),cmap="RdBu",vmin=-vlim,vmax=vlim,
             extent=[xmin,xmax,ymin,ymax])
plt.colorbar(im,ax=ax,shrink=0.8,label="Elevation-change rate [m yr$^{-1}$]")
ax.set_title(f"Klyuchevskaya glaciers — yearly trend {years[0]}-{years[-1]}\n"
             f"{len(scenes)} winter DEMs (>={MIN_YEARS} yr/pixel), median {np.median(gg):+.2f} m/yr")
ax.set_xlabel("UTM57N Easting [m]"); ax.set_ylabel("Northing [m]")
# histogram
axh.hist(np.clip(gg,-vlim,vlim),bins=80,color="0.5")
axh.axvline(np.median(gg),color="r",ls="--",label=f"median {np.median(gg):+.2f}")
axh.axvline(np.percentile(gg,5),color="b",ls=":",label=f"5th {np.percentile(gg,5):+.1f}")
axh.axvline(np.percentile(gg,95),color="b",ls=":",label=f"95th {np.percentile(gg,95):+.1f}")
axh.set_xlabel("rate [m/yr]"); axh.set_ylabel("glacier pixels"); axh.legend()
axh.set_title(f"distribution (clip +/-{RATE_CLIP:.0f} m/yr)\n"
              f"|rate|>2: {100*np.mean(np.abs(gg)>2):.0f}%   |rate|>10: {100*np.mean(np.abs(gg)>10):.0f}%")
fig.savefig(OUT/"yearly_trend_preview.png",dpi=130,bbox_inches="tight")
print(f"  distribution: 5%={np.percentile(gg,5):+.1f} 95%={np.percentile(gg,95):+.1f} m/yr; "
      f"|rate|>10 m/yr at {100*np.mean(np.abs(gg)>10):.0f}% of pixels")
print("\n-> yearly_trend_m_per_yr[_glacieronly].tif, yearly_trend_preview.png, yearly_trend_stats.csv")
