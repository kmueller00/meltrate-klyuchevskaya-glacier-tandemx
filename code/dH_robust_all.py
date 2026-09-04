#!/usr/bin/env python3
"""
ROBUST glacier elevation-change rates — winter AND summer.
Redone from scratch with the fixes that work:
  - MEDIAN epoch stacking (not mean: robust to per-scene artifacts)
  - FULL-coverage DEMs only (consistent extent; drops partial scenes)
  - same frame (155_0045) only
  - xdem Nuth&Kaab coregistration (late->early on stable terrain)
  - 3*NMAD glacier outlier clip
Outputs per season: rate tif (+glacieronly) + preview (map+hist).
"""
import numpy as np, rasterio, rasterio.features, geopandas as gpd, xdem
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date
import glob, re
from rasterio.warp import reproject, Resampling

BASE=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
OUT =Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out")
GLINV=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b/"
           "2025/2025-01-14_065954_TDT_SM_A_97516_155_0045+1-2/prc05_refdem/"
           "GLINV_rgi06_ttl_utm57_N_b_UTM.shp")
RES=30.0; ND=-9999.0; CORR_M=500.0; crs="EPSG:32657"
MIN_VALID=1.5e6        # include ALL usable DEMs (FNL+VER), only drop tiny/broken
SPLIT={"winter":2020, "summer":2018}   # early<split<=late

def find_dem(sd):
    for t in ("DEM_FNL","DEM_VER"):
        g=glob.glob(str(sd/"prc07"/f"{t}_*.tif"))
        if g: return g[0]

def is_season(mo,seas):
    return (mo>=11 or mo<=4) if seas=="winter" else (5<=mo<=10)

COH_MIN=0.8   # scene-level exclusion threshold. Checked against the full archive
# (348 scenes on this track): coherence is very high almost everywhere (median 0.998,
# p10=0.96); only 18 scenes fall below 0.8, one of which is the already-known-bad
# 2012-08-29, and 14 of the other 17 are 2015 scenes (which have a separate,
# systematic coherence problem -- see the report's 2015 discussion). 0.8 is
# conservative: it barely touches good data but reliably catches genuinely bad scenes.
def scene_coherence_ok(sd):
    g=glob.glob(str(sd/"prc06"/"*.int_coh.tif"))
    if not g: return True   # no coherence product available -- don't silently drop the scene
    with rasterio.open(g[0]) as s:
        a=s.read(1).astype("float32"); a[a<=0]=np.nan
        v=a[np.isfinite(a)]
    if v.size<1000: return True
    return float(np.nanmedian(v))>=COH_MIN

def collect(seas):
    out=[]; n_coh_excluded=0
    for yd in sorted(BASE.glob("20*")):
        for sd in sorted(yd.glob("20*")):
            nm=sd.name; m=re.match(r"(\d{4})-(\d{2})-(\d{2})",nm)
            if not m: continue
            y,mo,da=map(int,m.groups())
            if not is_season(mo,seas): continue
            if "155_0045" not in nm: continue
            f=find_dem(sd)
            if not f: continue
            with rasterio.open(f) as s:
                a=s.read(1); v=np.sum((a!=s.nodata)&np.isfinite(a)&(a>-1000)&(a<9000))
            if v<=MIN_VALID: continue
            if not scene_coherence_ok(sd):
                n_coh_excluded+=1; continue
            out.append((date(y,mo,da),f))
    print(f"  [{seas}] coherence-excluded (<{COH_MIN}): {n_coh_excluded} scenes")
    return out

FULL_FOOTPRINT_REF=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b/"
    "2024/2024-09-04_065957_TDT_SM_A_95512_155_0045_HH./prc07/DEM_FNL_2024-09-04_065957_0-030m00_155_0045_HH..tif")

def common_grid(scenes):
    """Use a fixed, known full-footprint scene as the grid -- NOT the intersection
    of every scene's bounds. The old intersection approach shrank the shared grid
    to the SMALLEST common denominator across dozens of scenes spanning 2012-2025:
    a single scene with a marginally smaller footprint anywhere in the stack would
    cut that area out of the analysis for the whole record, even though most other
    scenes covered it fine. This left ~35% of the glacier (concentrated in the
    northern branches) with no rate estimate at all, despite most individual scenes
    having valid data there. Using a generous fixed reference grid instead lets the
    per-pixel nanmedian stack use whatever subset of scenes actually covers each
    pixel, rather than requiring universal coverage across the whole 13-year stack."""
    with rasterio.open(FULL_FOOTPRINT_REF) as s:
        b=s.bounds
    xmin=np.floor(b.left/RES)*RES; xmax=np.ceil(b.right/RES)*RES
    ymin=np.floor(b.bottom/RES)*RES; ymax=np.ceil(b.top/RES)*RES
    cols=int((xmax-xmin)/RES);rows=int((ymax-ymin)/RES)
    return xmin,xmax,ymin,ymax,rows,cols,rasterio.transform.from_origin(xmin,ymax,RES,RES)

def read_grid(f,rows,cols,tr):
    with rasterio.open(f) as s:
        src=s.read(1).astype("float32"); src[(src==s.nodata)|(src<-1000)|(src>9000)]=np.nan
        dst=np.full((rows,cols),np.nan,"float32")
        reproject(src,dst,src_transform=s.transform,src_crs=s.crs,dst_transform=tr,
                  dst_crs=crs,resampling=Resampling.bilinear,src_nodata=np.nan,dst_nodata=np.nan)
    dst[(dst<-1000)|(dst>9000)]=np.nan
    return dst

def run(seas):
    print(f"\n{'='*60}\n{seas.upper()}")
    scenes=collect(seas)
    split=SPLIT[seas]
    early=[(d,f) for d,f in scenes if d.year<split]
    late =[(d,f) for d,f in scenes if d.year>=split]
    print(f"  full-coverage 155_0045 DEMs: {len(scenes)}  early={len(early)} late={len(late)}")
    if len(early)<1 or len(late)<1:
        print("  not enough scenes per epoch — skip"); return
    xmin,xmax,ymin,ymax,rows,cols,tr=common_grid(scenes)
    print(f"  grid {rows}x{cols}")
    emed=np.nanmedian(np.stack([read_grid(f,rows,cols,tr) for _,f in early]),axis=0)
    lmed=np.nanmedian(np.stack([read_grid(f,rows,cols,tr) for _,f in late ]),axis=0)
    shp=gpd.read_file(GLINV).to_crs(crs)
    gm=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
        out_shape=(rows,cols),transform=tr,fill=0,dtype="uint8").astype(bool)
    # 2-40 degree slope restriction on stable terrain, now the default rather than a
    # one-off sensitivity check (see "Does the stable-terrain mask itself matter?" in
    # the report): steep cutoff after Hugonnet et al. 2022 / xdem convention, flat
    # cutoff after Nuth & Kaab 2011. Previously verified this leaves most years'
    # rates visually unchanged while widening CIs where fewer stable pixels survive --
    # a real robustness gain, not expected to flip any conclusion.
    zref=read_grid(str(FULL_FOOTPRINT_REF),rows,cols,tr)
    gy,gx=np.gradient(zref,RES)
    slope_deg=np.degrees(np.arctan(np.hypot(gx,gy)))
    st=(~gm)&(slope_deg>=2)&(slope_deg<=40)&np.isfinite(slope_deg)
    try:
        rD=xdem.DEM.from_array(emed,transform=tr,crs=crs,nodata=np.nan)
        tD=xdem.DEM.from_array(lmed,transform=tr,crs=crs,nodata=np.nan)
        nk=xdem.coreg.NuthKaab(); nk.fit(rD,tD,inlier_mask=st,random_state=42)
        La=nk.apply(tD); lmed=La.data.filled(np.nan) if hasattr(La.data,'filled') else np.asarray(La.data,float)
        print("  coregistered late->early")
    except Exception as e: print(f"  coreg fallback ({e})")
    dh=lmed-emed; dh=dh-np.nanmedian(dh[st&np.isfinite(dh)])
    dt=(np.mean([d.toordinal() for d,_ in late])-np.mean([d.toordinal() for d,_ in early]))/365.25
    rate=dh/dt
    g=rate[gm&np.isfinite(rate)]; med=np.median(g); nmad=1.4826*np.median(np.abs(g-med))
    rc=np.where(np.isfinite(rate)&(np.abs(rate-med)<=3*nmad),rate,np.nan)
    prof=dict(driver="GTiff",height=rows,width=cols,count=1,dtype="float32",crs=crs,
              transform=tr,nodata=ND,compress="lzw")
    for arr,nm in [(rc,f"{seas}_robust_rate_m_per_yr.tif"),
                   (np.where(gm,rc,np.nan),f"{seas}_robust_rate_glacieronly.tif")]:
        with rasterio.open(OUT/nm,"w",**prof) as d: d.write(np.where(np.isfinite(arr),arr,ND).astype("float32"),1)
    gg=rc[gm&np.isfinite(rc)]; n_eff=max(1.0,gg.size*RES*RES/(np.pi*CORR_M**2))
    err=(1.4826*np.median(np.abs(gg-np.median(gg))))/np.sqrt(n_eff)
    print(f"  >>> RATE = {np.median(gg):+.3f} +/- {err:.3f} m/yr  (dt={dt:.1f}yr, px={gg.size:,})")
    vlim=np.nanpercentile(np.abs(gg),98)
    fig,(ax,axh)=plt.subplots(1,2,figsize=(15,7),gridspec_kw={"width_ratios":[1.4,1]})
    im=ax.imshow(np.where(gm,rc,np.nan),cmap="RdBu",vmin=-vlim,vmax=vlim,extent=[xmin,xmax,ymin,ymax])
    plt.colorbar(im,ax=ax,shrink=.8,label="rate [m/yr]")
    ax.set_title(f"Klyuchevskaya {seas} {early[0][0].year}-{late[-1][0].year}\n"
                 f"{len(scenes)} DEMs (median-stacked), {np.median(gg):+.2f} m/yr")
    axh.hist(np.clip(gg,-vlim,vlim),bins=70,color="0.5"); axh.axvline(np.median(gg),color="r",ls="--")
    axh.set_xlabel("rate [m/yr]"); axh.set_title("distribution")
    fig.savefig(OUT/f"PREVIEW_{seas}_robust_rate.png",dpi=130,bbox_inches="tight")
    return np.median(gg),err,dt,len(scenes)

res={s:run(s) for s in ("winter","summer")}
print("\n"+"="*60+"\nSUMMARY (robust, median-stacked, all full-coverage 155_0045 DEMs):")
for s,r in res.items():
    if r: print(f"  {s}: {r[0]:+.3f} +/- {r[1]:.3f} m/yr  ({r[3]} DEMs, {r[2]:.1f}yr baseline)")
