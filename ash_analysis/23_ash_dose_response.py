#!/usr/bin/env python3
"""
STEP 23: Ash 'dose-response' — does the NEXT-season melt scale with HOW DARK the
ash made the surface? Pixel-level test of the albedo mechanism for the 2023
paroxysm.

  dose     = NDSI drop across the eruption  (summer2023 - summer2024); larger = darker
  response = year-2 surface change          (dH summer2024 -> summer2025); - = melt
Because melt also increases downglacier, we CONTROL FOR ELEVATION: the relationship
is shown (i) raw and (ii) within a mid-elevation band (2000-2900 m) where most ash
sits, and we report partial Spearman (dose vs response | elevation).
Outputs: ASH_DOSE_RESPONSE.png + stats.
"""
import glob, numpy as np, rasterio, rasterio.features, geopandas as gpd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datetime import date, timedelta
from pathlib import Path
from rasterio.warp import reproject, Resampling
import pystac_client, planetary_computer, odc.stac
from scipy import stats
import warnings; warnings.filterwarnings("ignore")

BASE=Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation/ash_analysis")
GLINV=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/ash_analysis/GLINV_main_massif.gpkg")
crs="EPSG:32657"; RES=30.0; BBOX=[159.889,55.472,161.359,56.522]
S23="2023-09-29"; S24="2024-09-04"; S25="2025-09-13"
CAT=pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1',
                              modifier=planetary_computer.sign_inplace)

def dem(dstr):
    for sd in glob.glob(f"{BASE}/{dstr[:4]}/{dstr}*155_0045*"):
        for t in ("DEM_FNL","DEM_VER"):
            g=glob.glob(f"{sd}/prc07/{t}_*.tif")
            if g: return g[0]

def load_on(f,tr,rows,cols):
    with rasterio.open(f) as s:
        a=s.read(1).astype("float32"); a[(a==s.nodata)|(a<0)|(a>5000)]=np.nan
        o=np.full((rows,cols),np.nan,"float32")
        reproject(a,o,src_transform=s.transform,src_crs=s.crs,dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear)
    o[(o<0)|(o>5000)]=np.nan; return o

def ndsi(around,rows,cols,tr):
    d0=date.fromisoformat(around)
    dt=f"{(d0-timedelta(days=20)).isoformat()}/{(d0+timedelta(days=20)).isoformat()}"
    it=list(CAT.search(collections=['sentinel-2-l2a'],bbox=BBOX,datetime=dt,query={'eo:cloud_cover':{'lt':25}}).items())
    ds=odc.stac.load(it,bands=["B03","B11","SCL"],bbox=BBOX,resolution=RES,crs=crs,groupby="solar_day",chunks={})
    g=ds["B03"].astype('float32'); sw=ds["B11"].astype('float32'); scl=ds["SCL"]
    nd=((g-sw)/(g+sw+1e-6)).where(~scl.isin([0,1,3,8,9,10]))
    o=np.full((rows,cols),np.nan,'float32')
    reproject(np.asarray(nd.median(dim="time",skipna=True).values,'float32'),o,
              src_transform=ds.odc.transform,src_crs=crs,dst_transform=tr,dst_crs=crs,resampling=Resampling.bilinear)
    return o

f24=dem(S24)
with rasterio.open(f24) as s: tr=s.transform; rows,cols=s.shape; Z=s.read(1).astype(float); Z[(Z==s.nodata)|(Z<0)|(Z>5000)]=np.nan
shp=gpd.read_file(GLINV).to_crs(crs)
gm=rasterio.features.rasterize([(g,1) for g in shp.geometry if g is not None],out_shape=(rows,cols),transform=tr,fill=0,dtype='uint8').astype(bool)
st=~gm
D24=load_on(dem(S24),tr,rows,cols); D25=load_on(dem(S25),tr,rows,cols); D23=load_on(dem(S23),tr,rows,cols)
# year-2 response dH (2024->2025), de-biased
dh2=D25-D24; dh2=np.where(np.abs(dh2)<=60,dh2,np.nan); dh2=dh2-np.nanmedian(dh2[st&np.isfinite(dh2)])
nd23=ndsi(S23,rows,cols,tr); nd24=ndsi(S24,rows,cols,tr)
dose=nd23-nd24    # positive = darkened (NDSI dropped)

sel=gm&np.isfinite(dose)&np.isfinite(dh2)&np.isfinite(Z)
dz=dose[sel]; rp=dh2[sel]; ele=Z[sel]
print(f"pixels: {sel.sum():,}")
# raw + partial correlation
rho,p=stats.spearmanr(dz,rp)
# partial: residualize both on elevation (rank), then correlate
def rankresid(a,b):
    ra=stats.rankdata(a); rb=stats.rankdata(b); c=np.polyfit(rb,ra,1); return ra-np.polyval(c,rb)
pr,pp=stats.spearmanr(rankresid(dz,ele),rankresid(rp,ele))
print(f"dose vs year2-dH: Spearman rho={rho:+.3f} p={p:.1e}")
print(f"  partial (| elevation): rho={pr:+.3f} p={pp:.1e}")

# binned dose-response, overall + mid-band
band=(ele>=2000)&(ele<2900)
def binned(mask):
    d=dz[mask]; r=rp[mask]
    qs=np.quantile(d,np.linspace(0,1,9)); mids=[];meds=[];los=[];his=[]
    for i in range(8):
        s=(d>=qs[i])&(d<=qs[i+1])
        if s.sum()<30: continue
        mids.append(np.median(d[s])); meds.append(np.median(r[s]))
        los.append(np.percentile(r[s],25)); his.append(np.percentile(r[s],75))
    return map(np.array,(mids,meds,los,his))

fig,ax=plt.subplots(1,2,figsize=(14,6))
for a,(mask,ttl,rr,ppv) in zip(ax,[(np.ones_like(dz,bool),"all glacier pixels",rho,p),
                                    (band,"mid-elevation band 2000–2900 m",pr,pp)]):
    mids,meds,los,his=binned(mask)
    a.fill_between(mids,los,his,color="#c9a227",alpha=0.25)
    a.plot(mids,meds,'o-',color="#9a4a1e",lw=2,ms=7)
    a.axhline(0,color='k',lw=0.7,ls=':')
    a.set_xlabel("ash dose  =  NDSI drop 2023→2024  (→ darker)")
    a.set_ylabel("year–2 surface change 2024→2025 [m]  (− = melt)")
    a.set_title(f"{ttl}\nSpearman ρ={rr:+.2f} (p={ppv:.0e})")
    a.grid(alpha=0.25)
fig.suptitle("Ash dose–response: darker ash → more year–2 melt (2023 paroxysm)\n"
             "left: raw; right: elevation–controlled (partial correlation)",fontsize=13)
fig.tight_layout(); fig.savefig(AOUT/"ASH_DOSE_RESPONSE.png",dpi=600,bbox_inches="tight")
print("-> ASH_DOSE_RESPONSE.png")
