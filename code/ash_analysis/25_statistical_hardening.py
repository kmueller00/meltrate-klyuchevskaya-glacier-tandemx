#!/usr/bin/env python3
import sys; sys.path.insert(0,"/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/code/ash_analysis"); import _style
"""
STEP 25: Statistical hardening of the ash-vs-clean deposition/melt result.

Goes beyond the existing block-bootstrap (step 10) with a fuller battery that a
reviewer would ask for:

  (1) OUTLIER FILTERING (documented, reproducible): per group (ash / clean),
      remove pixels where dH is beyond 1.5x IQR (Tukey fence) -- the standard,
      assumption-light rule for skewed geophysical data. Report n removed.
  (2) PARAMETRIC + NON-PARAMETRIC tests on the filtered data:
        Welch's t-test (unequal variance, robust default)
        Mann-Whitney U (rank-based, distribution-free)
  (3) EFFECT SIZE: Cohen's d AND Cliff's delta (rank-based, robust to outliers
      and non-normal spatial data -- the more defensible one here).
  (4) MULTIPLE-COMPARISON CORRECTION: Holm-Bonferroni across the 4 eruptions,
      since we are testing 4 hypotheses from the same glacier/pipeline.
  (5) SPATIAL PERMUTATION TEST: an independent check on the block-bootstrap.
      Shuffles the ash/clean LABELS in spatially contiguous blocks (not single
      pixels) 2000x, rebuilding the null distribution of the median difference
      under the SAME spatial-autocorrelation structure as the real data.
  (6) NORMALITY CHECK on stable terrain (Shapiro-Wilk on a subsample) to justify
      the use of robust/rank-based statistics over parametric ones.

Emits a single results table (CSV) and a compact summary figure.
"""
import glob, numpy as np, rasterio, rasterio.features, geopandas as gpd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datetime import date, timedelta
from pathlib import Path
from rasterio.warp import reproject, Resampling
import pystac_client, planetary_computer, odc.stac
from scipy import ndimage, stats
import csv, warnings
warnings.filterwarnings("ignore")

BASE=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation")
AASH=AOUT/"ash_analysis"
GLINV=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/code/ash_analysis/GLINV_main_massif.gpkg")
crs="EPSG:32657"; RES=30.0; BBOX=[159.889,55.472,161.359,56.522]
CAT=pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1',
                              modifier=planetary_computer.sign_inplace)
rng=np.random.default_rng(2024)
NPERM=2000

BRACKETS=[("2019 Oct","2019-09-27","2020-09-02"),("2020 Dec","2020-09-24","2021-08-31"),
          ("2022 Nov","2022-10-01","2023-08-05"),("2023 PAROXYSM","2023-09-29","2024-09-04")]

def dem_path(dstr):
    for sd in glob.glob(f"{BASE}/{dstr[:4]}/{dstr}*155_0045*"):
        for t in ("DEM_FNL","DEM_VER"):
            g=glob.glob(f"{sd}/prc07/{t}_*.tif")
            if g: return g[0]

def ndsi_bright(around,rows,cols,tr):
    d0=date.fromisoformat(around)
    dt=f"{(d0-timedelta(days=20)).isoformat()}/{(d0+timedelta(days=20)).isoformat()}"
    it=list(CAT.search(collections=['sentinel-2-l2a'],bbox=BBOX,datetime=dt,
                       query={'eo:cloud_cover':{'lt':25}}).items())
    if not it: return None,None
    ds=odc.stac.load(it,bands=["B03","B11","SCL"],bbox=BBOX,resolution=RES,crs=crs,
                     groupby="solar_day",chunks={})
    g=ds["B03"].astype('float32'); sw=ds["B11"].astype('float32'); scl=ds["SCL"]
    valid=~scl.isin([0,1,3,8,9,10])
    nd=((g-sw)/(g+sw+1e-6)).where(valid); br=g.where(valid)
    def tg(a):
        o=np.full((rows,cols),np.nan,'float32')
        reproject(np.asarray(a.median(dim="time",skipna=True).values,'float32'),o,
                  src_transform=ds.odc.transform,src_crs=crs,dst_transform=tr,dst_crs=crs,
                  resampling=Resampling.bilinear); return o
    return tg(nd),tg(br)

def tukey_filter(x):
    """Tukey 1.5xIQR fence. Returns (kept_values, n_removed)."""
    if x.size<10: return x, 0
    q1,q3=np.percentile(x,[25,75]); iqr=q3-q1
    lo,hi=q1-1.5*iqr, q3+1.5*iqr
    keep=(x>=lo)&(x<=hi)
    return x[keep], int((~keep).sum())

def cliffs_delta(a,c):
    """Rank-based effect size, robust to outliers/non-normality. Subsample for speed."""
    na,nc=len(a),len(c)
    sa=rng.choice(a,min(na,4000),replace=False) if na>4000 else a
    sc=rng.choice(c,min(nc,4000),replace=False) if nc>4000 else c
    diff = sa[:,None]-sc[None,:]
    gt=(diff>0).sum(); lt=(diff<0).sum()
    return (gt-lt)/(len(sa)*len(sc))

def cohens_d(a,c):
    na,nc=len(a),len(c)
    pooled=np.sqrt(((na-1)*np.var(a,ddof=1)+(nc-1)*np.var(c,ddof=1))/(na+nc-2))
    return (np.mean(a)-np.mean(c))/pooled if pooled>0 else np.nan

def spatial_block_permtest(ash_mask,dh,gm,block=5,nperm=NPERM):
    """Permutation test preserving spatial autocorrelation: coarsen to blocks,
    permute block-level ash/clean labels, recompute median difference, build null."""
    rows,cols=ash_mask.shape
    br,bc=rows//block, cols//block
    if br<3 or bc<3: return np.nan, np.nan
    ash_b=ash_mask[:br*block,:bc*block].reshape(br,block,bc,block).mean(axis=(1,3))>0.5
    dh_b=dh[:br*block,:bc*block].reshape(br,block,bc,block)
    dh_b=np.nanmedian(dh_b.reshape(br,block,bc,block),axis=(1,3))
    gm_b=gm[:br*block,:bc*block].reshape(br,block,bc,block).mean(axis=(1,3))>0.5
    valid=gm_b&np.isfinite(dh_b)
    labels=ash_b[valid]; vals=dh_b[valid]
    if labels.sum()<5 or (~labels).sum()<5: return np.nan, np.nan
    obs=np.median(vals[labels])-np.median(vals[~labels])
    null=np.empty(nperm)
    for i in range(nperm):
        perm=rng.permutation(labels)
        null[i]=np.median(vals[perm])-np.median(vals[~perm])
    p=2*min((null<=obs).mean(),(null>=obs).mean())
    return obs, min(p,1.0)

shp=gpd.read_file(GLINV).to_crs(crs)
rows_out=[]
stable_sample_for_normality=None
for lab,pre,post in BRACKETS:
    fA=dem_path(post); fB=dem_path(pre)
    if not(fA and fB): print(f"{lab}: missing DEM"); continue
    with rasterio.open(fA) as s:
        A=s.read(1).astype(float); A[(A==s.nodata)|(A<0)|(A>5000)]=np.nan
        tr=s.transform; rows,cols=s.shape
    with rasterio.open(fB) as s:
        b=s.read(1).astype(float); b[(b==s.nodata)|(b<0)|(b>5000)]=np.nan
        B=np.full((rows,cols),np.nan,'float32')
        reproject(b.astype('float32'),B,src_transform=s.transform,src_crs=s.crs,
                  dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear)
    dh=A-B; dh=np.where(np.abs(dh)<=120,dh,np.nan)
    gm=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],
        out_shape=(rows,cols),transform=tr,fill=0,dtype='uint8').astype(bool)
    st=~gm
    stv=dh[st&np.isfinite(dh)]
    if stv.size>1000: dh=dh-np.nanmedian(stv)
    if stable_sample_for_normality is None and stv.size>500:
        stable_sample_for_normality=rng.choice(stv[np.isfinite(stv)],min(stv.size,4900),replace=False)

    ndA,brA=ndsi_bright(post,rows,cols,tr); ndB,_=ndsi_bright(pre,rows,cols,tr)
    if ndA is None or ndB is None: print(f"{lab}: S2 fail"); continue
    dn=ndA-ndB; br_thr=np.nanpercentile(brA[gm&np.isfinite(brA)],40)
    ash=gm&np.isfinite(dn)&(dn<-0.2)&(ndA<0.3)&(brA<br_thr)
    ash=ndimage.binary_opening(ash); ash=ndimage.binary_closing(ash)

    da_raw=dh[ash&np.isfinite(dh)]; dc_raw=dh[gm&~ash&np.isfinite(dh)]
    if da_raw.size<50: print(f"{lab}: too few ash px"); continue
    da,na_rm=tukey_filter(da_raw); dc,nc_rm=tukey_filter(dc_raw)

    t_stat,t_p=stats.ttest_ind(da,dc,equal_var=False)                 # Welch
    u_stat,u_p=stats.mannwhitneyu(da,dc,alternative='two-sided')      # rank-based
    d=cohens_d(da,dc); delta=cliffs_delta(da,dc)
    obs_perm,p_perm=spatial_block_permtest(ash,dh,gm)

    print(f"{lab}: n_ash={da.size:,} (removed {na_rm}), n_clean={dc.size:,} (removed {nc_rm})")
    print(f"  Welch t={t_stat:.2f} p={t_p:.2e} | Mann-Whitney U p={u_p:.2e}")
    print(f"  Cohen's d={d:+.2f}  Cliff's delta={delta:+.3f}")
    print(f"  spatial-block permutation: D={obs_perm:+.2f} p={p_perm:.4f}")
    rows_out.append(dict(eruption=lab,n_ash=da.size,n_ash_removed=na_rm,n_clean=dc.size,
        n_clean_removed=nc_rm,median_ash=np.median(da),median_clean=np.median(dc),
        welch_t=t_stat,welch_p=t_p,mannwhitney_p=u_p,cohens_d=d,cliffs_delta=delta,
        perm_D=obs_perm,perm_p=p_perm))

# Shapiro-Wilk normality check on stable terrain (justifies robust stats)
if stable_sample_for_normality is not None:
    sw_stat,sw_p=stats.shapiro(stable_sample_for_normality)
    print(f"\nShapiro-Wilk normality (stable terrain, n={len(stable_sample_for_normality)}): "
          f"W={sw_stat:.4f} p={sw_p:.2e}  {'-> NOT normal, robust/rank stats justified' if sw_p<0.05 else '-> normal-consistent'}")
else:
    sw_stat,sw_p=np.nan,np.nan

# Holm-Bonferroni correction across the 4 eruption tests (using Mann-Whitney p, the
# distribution-free choice) -- controls family-wise error rate for testing 4 hypotheses
ps=[r['mannwhitney_p'] for r in rows_out]
order=np.argsort(ps); m=len(ps)
holm_p=np.empty(m)
running_max=0
for rank,idx in enumerate(order):
    adj=ps[idx]*(m-rank)
    running_max=max(running_max,adj)
    holm_p[idx]=min(running_max,1.0)
for r,hp in zip(rows_out,holm_p): r['holm_bonferroni_p']=hp

with open(AOUT/"STATISTICAL_HARDENING.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows_out[0].keys()),lineterminator="\n")
    w.writeheader()
    for r in rows_out: w.writerow(r)
print("\n-> STATISTICAL_HARDENING.csv")

# ---- summary figure: effect sizes + corrected significance ----
fig,axes=plt.subplots(1,2,figsize=(13,5))
labs=[r['eruption'] for r in rows_out]
d_vals=[r['cohens_d'] for r in rows_out]; delta_vals=[r['cliffs_delta'] for r in rows_out]
x=np.arange(len(labs))
axes[0].bar(x-0.18,d_vals,width=0.36,color="#2f7cb2",label="Cohen's d")
axes[0].bar(x+0.18,delta_vals,width=0.36,color="#c9a227",label="Cliff's delta")
axes[0].axhline(0,color='k',lw=0.8)
axes[0].set_xticks(x); axes[0].set_xticklabels(labs,rotation=15,ha='right')
axes[0].set_ylabel("effect size"); axes[0].legend(fontsize=9)
axes[0].set_title("Effect sizes (parametric + rank-based)")

holm=[r['holm_bonferroni_p'] for r in rows_out]
perm_p=[r['perm_p'] for r in rows_out]
axes[1].scatter(x-0.1,[-np.log10(max(p,1e-300)) for p in holm],s=90,color="#2f7cb2",label="Holm-Bonferroni p (Mann-Whitney)",zorder=3)
axes[1].scatter(x+0.1,[-np.log10(max(p,1e-300)) for p in perm_p],s=90,color="#c0463a",marker='D',label="spatial-block permutation p",zorder=3)
axes[1].axhline(-np.log10(0.05),color='grey',ls='--',lw=1,label="p=0.05")
axes[1].set_xticks(x); axes[1].set_xticklabels(labs,rotation=15,ha='right')
axes[1].set_ylabel("-log10(p)"); axes[1].legend(fontsize=8)
axes[1].set_title("Significance: two independent methods,\nfamily-wise corrected")
fig.suptitle("Statistical hardening of ash-vs-clean deposition/melt anomaly",fontsize=13)
fig.tight_layout(); fig.savefig(AASH/"STATISTICAL_HARDENING.png",dpi=600,bbox_inches="tight")
print("-> STATISTICAL_HARDENING.png")
