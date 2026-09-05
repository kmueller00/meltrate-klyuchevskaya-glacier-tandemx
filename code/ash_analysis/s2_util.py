"""Shared Sentinel-2 NDSI/brightness fetch with near+wide gap-filling.

Replaces the old ndsi_bright() duplicated across steps 08/10/11/17/18, which
used a single rigid +/-20 day window and NO fallback -- a cloud gap or swath
edge in that window silently shrank the ash mask (NaN pixels are excluded by
`np.isfinite(dn)`, no warning). This version composites a NEAR window first
(same +/-20 d as before) and fills any remaining gaps from a WIDER window,
same two-layer strategy as step 19's S2 export, and reports the achieved
glacier-area coverage so incomplete coverage is visible, not silently assumed.
Windows are clamped so they never cross the eruption date (pre stays strictly
before, post stays strictly after), same safeguard as step 19.
"""
import numpy as np
from datetime import date, timedelta
from rasterio.warp import reproject, Resampling
import pystac_client, planetary_computer, odc.stac

BBOX=[159.889,55.472,161.359,56.522]
CAT=pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1',
                              modifier=planetary_computer.sign_inplace)

L8_SCALE, L8_OFFSET = 2.75e-5, -0.2          # landsat-c2-l2 raster:bands, verified live 2026-08-25
QA_EXCLUDE_MASK = 0b0011111                   # bits 0-4: fill|dilated_cloud|cirrus|cloud|cloud_shadow (bit5 snow kept)

def _fetch_window(d0,d1,rows,cols,tr,crs,cloud_lt=40):
    if d0>=d1: return None,None
    it=list(CAT.search(collections=['sentinel-2-l2a'],bbox=BBOX,
        datetime=f"{d0.isoformat()}/{d1.isoformat()}",
        query={'eo:cloud_cover':{'lt':cloud_lt}}).items())
    if not it: return None,None
    ds=odc.stac.load(it,bands=["B03","B11","SCL"],bbox=BBOX,resolution=30.0,
                     crs=crs,groupby="solar_day",chunks={})
    g=ds["B03"].astype('float32'); sw=ds["B11"].astype('float32'); scl=ds["SCL"]
    valid=~scl.isin([0,1,3,8,9,10])
    nd=((g-sw)/(g+sw+1e-6)).where(valid); br=g.where(valid)
    def tg(a):
        o=np.full((rows,cols),np.nan,'float32')
        reproject(np.asarray(a.median(dim="time",skipna=True).values,'float32'),o,
                  src_transform=ds.odc.transform,src_crs=crs,dst_transform=tr,dst_crs=crs,
                  resampling=Resampling.bilinear); return o
    return tg(nd),tg(br)

def _fetch_window_landsat(d0,d1,rows,cols,tr,crs,cloud_lt=40):
    """Landsat 8 OLI equivalent of _fetch_window, for pre-2016 dates where
    Sentinel-2 doesn't usefully cover this site. green/swir16 need the
    Collection-2 scale+offset applied manually (odc.stac doesn't auto-apply
    it) -- unlike S2 L2A's raw-DN ratio shortcut, Landsat's additive offset
    does NOT cancel out of the NDSI ratio, so skipping this would bias it."""
    if d0>=d1: return None,None
    it=list(CAT.search(collections=['landsat-c2-l2'],bbox=BBOX,
        datetime=f"{d0.isoformat()}/{d1.isoformat()}",
        query={'eo:cloud_cover':{'lt':cloud_lt},'platform':{'eq':'landsat-8'}}).items())
    it=[i for i in it if i.properties.get('platform')=='landsat-8']  # fail-closed re-filter
    if not it: return None,None
    ds=odc.stac.load(it,bands=["green","swir16","qa_pixel"],bbox=BBOX,resolution=30.0,
                     crs=crs,groupby="solar_day",chunks={})
    g_dn=ds["green"]; sw_dn=ds["swir16"]; qa=ds["qa_pixel"]
    g=g_dn.astype('float32')*L8_SCALE+L8_OFFSET; sw=sw_dn.astype('float32')*L8_SCALE+L8_OFFSET
    # unlike S2's raw positive-DN ratio, Landsat SR reflectance can be slightly
    # negative after atmospheric correction (dark/noisy pixels). NDSI is only
    # mathematically bounded to [-1,1] when both bands are >=0 (then |g-sw|<=g+sw
    # always); a sum-only denominator guard isn't enough -- a strongly negative
    # band paired with a moderate positive one can keep the sum just above a small
    # floor while the difference stays large, still blowing the ratio up. Require
    # both bands individually non-negative, plus a sum floor for the near-zero/
    # near-zero case, rather than relying on the +1e-6 epsilon alone.
    valid=((qa&QA_EXCLUDE_MASK)==0)&(g_dn>0)&(sw_dn>0)&(g>=0)&(sw>=0)&((g+sw)>0.02)
    nd=((g-sw)/(g+sw+1e-6)).where(valid); br=g.where(valid)
    def tg(a):
        o=np.full((rows,cols),np.nan,'float32')
        reproject(np.asarray(a.median(dim="time",skipna=True).values,'float32'),o,
                  src_transform=ds.odc.transform,src_crs=crs,dst_transform=tr,dst_crs=crs,
                  resampling=Resampling.bilinear); return o
    return tg(nd),tg(br)

_FETCHERS={"s2":_fetch_window,"landsat":_fetch_window_landsat}
_SRC_TAG={"s2":"S2","landsat":"Landsat8"}

def ndsi_bright(around, rows, cols, tr, crs, erupt_date=None, side=None,
                 near_days=20, wide_days=60, gm=None, label="", source="s2"):
    """side='pre' clamps the window to end before erupt_date; side='post'
    clamps it to start after erupt_date. gm (glacier boolean mask) is used
    only to report coverage; pass None to skip the print. source='s2'
    (default) or 'landsat' (Landsat 8 OLI, for pre-2016 dates)."""
    fetch=_FETCHERS.get(source)
    if fetch is None: raise ValueError(f"ndsi_bright: unknown source {source!r}")
    d0=date.fromisoformat(around) if isinstance(around,str) else around
    near=(d0-timedelta(days=near_days), d0+timedelta(days=near_days))
    wide=(d0-timedelta(days=wide_days), d0+timedelta(days=wide_days))
    if erupt_date is not None and side=="pre":
        cap=erupt_date-timedelta(days=1)
        near=(near[0],min(near[1],cap)); wide=(wide[0],min(wide[1],cap))
    elif erupt_date is not None and side=="post":
        cap=erupt_date+timedelta(days=1)
        near=(max(near[0],cap),near[1]); wide=(max(wide[0],cap),wide[1])
    nd_n,br_n=fetch(*near,rows,cols,tr,crs)
    nd_w,br_w=fetch(*wide,rows,cols,tr,crs)
    def combine(n,w):
        if n is None: return w
        if w is None: return n
        return np.where(np.isfinite(n),n,w)
    nd=combine(nd_n,nd_w); br=combine(br_n,br_w)
    if nd is None: return None,None
    if gm is not None:
        vf_near=np.isfinite(nd_n)[gm].mean()*100 if nd_n is not None else 0.0
        vf=np.isfinite(nd)[gm].mean()*100
        print(f"  {label} {_SRC_TAG[source]} [{near[0]}..{near[1]}]: glacier valid {vf_near:.1f}% (near) -> {vf:.1f}% (near+wide)")
    return nd,br

def fetch_tci(around, rows, cols, tr, crs, erupt_date=None, side=None,
               near_days=20, wide_days=60, label=""):
    """Sentinel-2 true-color (B04/B03/B02) composite for map backgrounds --
    same near+wide gap-filling and eruption-date clamping as ndsi_bright,
    contrast-stretched to the 2nd-98th percentile. Returns an (rows,cols,3)
    float32 array in [0,1] (gaps filled white, not black), or None if no
    scenes were found in either window."""
    d0=date.fromisoformat(around) if isinstance(around,str) else around
    near=(d0-timedelta(days=near_days), d0+timedelta(days=near_days))
    wide=(d0-timedelta(days=wide_days), d0+timedelta(days=wide_days))
    if erupt_date is not None and side=="pre":
        cap=erupt_date-timedelta(days=1)
        near=(near[0],min(near[1],cap)); wide=(wide[0],min(wide[1],cap))
    elif erupt_date is not None and side=="post":
        cap=erupt_date+timedelta(days=1)
        near=(max(near[0],cap),near[1]); wide=(max(wide[0],cap),wide[1])
    def fetch(d0,d1,cloud_lt=40):
        if d0>=d1: return None
        it=list(CAT.search(collections=['sentinel-2-l2a'],bbox=BBOX,
            datetime=f"{d0.isoformat()}/{d1.isoformat()}",
            query={'eo:cloud_cover':{'lt':cloud_lt}}).items())
        if not it: return None
        ds=odc.stac.load(it,bands=["B04","B03","B02","SCL"],bbox=BBOX,resolution=30.0,
                         crs=crs,groupby="solar_day",chunks={})
        scl=ds["SCL"]; valid=~scl.isin([0,1,3,8,9,10])
        out=np.full((rows,cols,3),np.nan,'float32')
        for i,b in enumerate(["B04","B03","B02"]):
            a=ds[b].astype('float32').where(valid)
            med=np.asarray(a.median(dim="time",skipna=True).values,'float32')
            o=np.full((rows,cols),np.nan,'float32')
            reproject(med,o,src_transform=ds.odc.transform,src_crs=crs,dst_transform=tr,dst_crs=crs,
                      resampling=Resampling.bilinear)
            out[:,:,i]=o
        return out
    rgb=fetch(*near); rgb_w=fetch(*wide)
    if rgb is None: rgb=rgb_w
    elif rgb_w is not None:
        for i in range(3):
            rgb[:,:,i]=np.where(np.isfinite(rgb[:,:,i]),rgb[:,:,i],rgb_w[:,:,i])
    if rgb is None:
        print(f"  {label} TCI: no scenes found"); return None
    lo=np.nanpercentile(rgb,2); hi=np.nanpercentile(rgb,98)
    rgb=np.clip((rgb-lo)/(hi-lo+1e-6),0,1)
    rgb=np.where(np.isfinite(rgb),rgb,1.0).astype('float32')
    print(f"  {label} TCI [{near[0]}..{near[1]}]")
    return rgb
