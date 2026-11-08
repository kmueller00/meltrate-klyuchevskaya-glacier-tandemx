#!/usr/bin/env python3
"""Post-eruption ash masks (PROTOCOL_v2.md). Writes only to this folder."""
import numpy as np, rasterio, pystac_client, planetary_computer, odc.stac
from datetime import date
from rasterio.warp import reproject, Resampling, transform_bounds
from scipy import ndimage
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT="/home/saturn/gwgi/gwgifu0h"; RES=f"{ROOT}/meltrate-klyuchevskaya-glacier-tandemx/prc07_overview_out/RESULTS_presentation"; OUT=f"{ROOT}/00_libs/paper_checks/decisive/v2"
W={"2019_Oct":(date(2019,9,25),date(2019,10,24),date(2019,10,28),date(2019,11,30)),"2020_Dec":(date(2020,11,9),date(2020,12,8),date(2020,12,10),date(2021,1,15)),
   "2022_Nov":(date(2022,10,16),date(2022,11,15),date(2022,11,27),date(2022,12,31)),"2023_PAROXYSM":(date(2023,10,1),date(2023,10,30),date(2023,11,3),date(2023,12,5))}
with rasterio.open(f"{RES}/FULL_GRID_elevation_mosaic.tif") as s:
    Z=s.read(1).astype("float32"); TR=s.transform; CRS=s.crs; SH=s.shape; BB=s.bounds
Z[(Z<=0)|(Z>6000)]=np.nan
gy,gx=np.gradient(Z,30.0); SL=np.arctan(np.hypot(gx,gy)); AS=np.arctan2(-gx,gy)
LL=transform_bounds(CRS,"EPSG:4326",*BB)
CAT=pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",modifier=planetary_computer.sign_inplace)
def old(n):
    with rasterio.open(f"{RES}/ash_analysis/ASHMASK_{n}_strict.tif") as s:
        o=np.full(SH,255,"uint8"); reproject(s.read(1),o,src_transform=s.transform,src_crs=s.crs,dst_transform=TR,dst_crs=CRS,resampling=Resampling.nearest,src_nodata=None,dst_nodata=255)
    return o
def days(d0,d1):
    it=list(CAT.search(collections=["sentinel-2-l2a"],bbox=LL,datetime=f"{d0}/{d1}",query={"eo:cloud_cover":{"lt":60}}).items())
    sun={}
    for i in it: sun[i.datetime.date()]=(i.properties.get("s2:mean_solar_zenith"),i.properties.get("s2:mean_solar_azimuth"))
    ds=odc.stac.load(it,bands=["B02","B03","B04","B11","SCL"],bbox=LL,resolution=30.0,crs=CRS,groupby="solar_day",chunks={})
    return ds,sun
def on(a,ds,rs):
    o=np.full(SH,np.nan,"float32"); reproject(np.asarray(a,"float32"),o,src_transform=ds.odc.transform,src_crs=CRS,dst_transform=TR,dst_crs=CRS,resampling=rs,src_nodata=np.nan,dst_nodata=np.nan); return o
def composite(d0,d1,first,gm):
    ds,sun=days(d0,d1); T=list(ds.time.values); T=T if first else T[::-1]
    N=np.full(SH,np.nan,"float32"); G=N.copy(); R=N.copy(); B=N.copy(); DT=np.zeros(SH,"int32"); log=[]
    for t in T:
        x=ds.sel(time=t).compute(); dn=np.datetime64(t,"D").astype(object)
        g=on(x.B03.astype("float32").values,ds,Resampling.bilinear); sw=on(x.B11.astype("float32").values,ds,Resampling.bilinear)
        r=on(x.B04.astype("float32").values,ds,Resampling.bilinear); b=on(x.B02.astype("float32").values,ds,Resampling.bilinear)
        sc=np.full(SH,0,"uint8"); reproject(x.SCL.values.astype("uint8"),sc,src_transform=ds.odc.transform,src_crs=CRS,dst_transform=TR,dst_crs=CRS,resampling=Resampling.nearest,src_nodata=0,dst_nodata=0)
        z,az=sun.get(dn,(None,None))
        if z is None: log.append(f"{dn}: no sun angles"); continue
        zr,azr=np.radians(z),np.radians(az); cosi=np.cos(zr)*np.cos(SL)+np.sin(zr)*np.sin(SL)*np.cos(azr-AS)
        ok=~np.isin(sc,[0,1,3,8,9,10])&(cosi>=0.15)&np.isfinite(g)&np.isfinite(sw)&((g+sw)>0)
        new=ok&~np.isfinite(N); nd=(g-sw)/(g+sw+1e-6)
        N[new]=nd[new]; G[new]=g[new]; R[new]=r[new]; B[new]=b[new]; DT[new]=dn.toordinal()
        log.append(f"{dn}: sun elev {90-z:.1f}, new glacier px {int((new&gm).sum())}, filled {100*np.isfinite(N[gm]).mean():.1f}%")
        if np.isfinite(N[gm]).mean()>0.995: break
    return N,G,R,B,DT,log
fig,axs=plt.subplots(2,4,figsize=(19,10),constrained_layout=True)
print("mask, ash km2, clean km2, excluded km2, ash median elev, ash% <2000/2000-3000/>3000, overlap with old strict ash %")
for j,(n,(a,b,c,d)) in enumerate(W.items()):
    o=old(n); gm=o!=255
    pre,_,_,_,dpre,l1=composite(a,b,False,gm); post,g,r,bl,dpost,l2=composite(c,d,True,gm)
    print(f"\n== {n}\n PRE "+" | ".join(l1)+"\n POST "+" | ".join(l2),flush=True)
    snow=gm&(pre>=0.4); valid=snow&np.isfinite(post)
    ash=ndimage.binary_opening(valid&(post<0.4)); clean=valid&(post>=0.4)&~ash
    drop=pre-post; lo=ndimage.binary_opening(valid&((post<0.6)|(drop>=0.2))); lc=valid&~lo
    for tag,A,C in (("strict",ash,clean),("loose",lo,lc)):
        m=np.full(SH,255,"uint8"); m[C]=0; m[A]=1
        with rasterio.open(f"{OUT}/ASHV2_{n}_{tag}.tif","w",driver="GTiff",height=SH[0],width=SH[1],count=1,dtype="uint8",crs=CRS,transform=TR,nodata=255,compress="lzw") as t: t.write(m,1)
        zz=Z[A&np.isfinite(Z)]
        print(f"{n} {tag}: ash {A.sum()*900/1e6:.1f} clean {C.sum()*900/1e6:.1f} excl {(gm&~A&~C).sum()*900/1e6:.1f} km2; med elev {np.median(zz) if zz.size else float('nan'):.0f}; <2000 {100*(zz<2000).mean() if zz.size else 0:.0f}% 2000-3000 {100*((zz>=2000)&(zz<3000)).mean() if zz.size else 0:.0f}% >3000 {100*(zz>=3000).mean() if zz.size else 0:.0f}%; overlap with old ash {100*(o[A]==1).mean() if A.sum() else 0:.0f}%",flush=True)
    rgb=np.dstack([r,g,bl]); v=np.nanpercentile(rgb[gm],99.5) if np.isfinite(rgb[gm]).any() else 1
    ax=axs[0,j]; ax.imshow(np.clip(np.nan_to_num(rgb/v),0,1)**0.6); ax.contour(gm.astype(float),[.5],colors="cyan",linewidths=.5); ax.contour(ash.astype(float),[.5],colors="red",linewidths=.6); ax.set_title(f"{n}: first clear POST, red = strict ash",fontsize=10); ax.axis("off")
    ax=axs[1,j]; im=ax.imshow(np.where(gm,post,np.nan),cmap="RdBu",vmin=-.5,vmax=1); ax.contour(gm.astype(float),[.5],colors="k",linewidths=.4); ax.set_title("NDSI POST on glacier",fontsize=10); ax.axis("off")
fig.savefig(f"{OUT}/masks_qc.png",dpi=95); print("done")
