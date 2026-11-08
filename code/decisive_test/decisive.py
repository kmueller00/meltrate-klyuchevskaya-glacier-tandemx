#!/usr/bin/env python3
"""Decisive test of eruption signals; see PROTOCOL.md (read-only on the data, writes only to this folder)."""
import csv, glob, os, sys, time, warnings
import numpy as np, rasterio, rasterio.features, geopandas as gpd, xdem
from rasterio.warp import reproject, Resampling
warnings.filterwarnings("ignore")
ROOT = "/home/saturn/gwgi/gwgifu0h"
BASE = f"{ROOT}/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b"
RES = f"{ROOT}/meltrate-klyuchevskaya-glacier-tandemx/prc07_overview_out/RESULTS_presentation"
GLACIERS = f"{ROOT}/02_ref_dem/CP30_construction/10NAS/utm57_N_b/GLINV_rgi06_ttl_utm57_N_b_UTM.shp"
OUT = f"{ROOT}/00_libs/paper_checks/decisive"
SMOKE = os.environ.get("SMOKE") == "1"
YEARS = list(range(2023, 2026)) if SMOKE else list(range(2012, 2026))
NBOOT = 50 if SMOKE else 1000
SEEDS = (42, 1, 7, 13, 99)
TARGETS = {"2019_Oct": [(2019, "T")], "2020_Dec": [(2020, "T")], "2022_Nov": [(2022, "T")], "2023_PAROXYSM": [(2023, "Y1"), (2024, "Y2")]}
if SMOKE:
    TARGETS = {"2023_PAROXYSM": [(2023, "Y1"), (2024, "Y2")]}
kept = {r["date"]: r["kept"] for r in csv.DictReader(open(f"{ROOT}/00_libs/paper_checks/glacier_median_series.csv"))}
rng = np.random.default_rng(11)
log = open(f"{OUT}/{'smoke' if SMOKE else 'verdicts'}.txt", "w")
def P(*a):
    print(*a, flush=True); print(*a, file=log, flush=True)

with rasterio.open(f"{RES}/FULL_GRID_elevation_mosaic.tif") as s:
    Z = s.read(1).astype("float32"); TR = s.transform; CRS = s.crs; SH = s.shape
    if s.nodata is not None: Z[Z == s.nodata] = np.nan
Z[(Z <= 0) | (Z > 6000)] = np.nan
gy, gx = np.gradient(np.where(np.isfinite(Z), Z, np.nan), 30.0)
SLOPE = np.degrees(np.arctan(np.hypot(gx, gy))); ASPECT = (np.degrees(np.arctan2(-gx, gy)) + 360) % 360
gl = gpd.read_file(GLACIERS).to_crs(CRS).buffer(300)
ICE = rasterio.features.rasterize([(g, 1) for g in gl if g is not None and not g.is_empty], out_shape=SH, transform=TR, fill=0, dtype="uint8").astype(bool)
ST = ~ICE & np.isfinite(Z) & (SLOPE >= 2) & (SLOPE <= 40)
P(f"grid {SH}, stable pixels {ST.sum():,}")
refD = xdem.DEM.from_array(Z, transform=TR, crs=CRS, nodata=np.nan)

def read(f):
    with rasterio.open(f) as s:
        a = s.read(1).astype("float32"); a[(a == s.nodata) | (a < 0) | (a > 5000)] = np.nan
        o = np.full(SH, np.nan, "float32")
        reproject(a, o, src_transform=s.transform, src_crs=s.crs, dst_transform=TR, dst_crs=CRS, resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
    o[(o < 0) | (o > 5000)] = np.nan
    return o, int(np.isfinite(a).sum()), float(np.nanmedian(a))

def nmad_stable(d):
    v = (d - Z)[ST]; v = v[np.isfinite(v)]
    return 999.0 if v.size < 50000 else 1.4826 * float(np.median(np.abs(v - np.median(v))))

def coreg(d0):
    best = None
    for seed in SEEDS:
        try:
            t = xdem.DEM.from_array(d0.copy(), transform=TR, crs=CRS, nodata=np.nan)
            nk = xdem.coreg.NuthKaab(); nk.fit(refD, t, inlier_mask=ST, random_state=seed)
            aff = nk.meta["outputs"]["affine"]; dx, dy = float(aff["shift_x"]), float(aff["shift_y"])
            if max(abs(dx), abs(dy)) > 60:
                continue
            o = nk.apply(t); o = o.data.filled(np.nan) if hasattr(o.data, "filled") else np.asarray(o.data, float)
            o = o.astype("float32"); n = nmad_stable(o)
            if best is None or n < best[0]:
                best = (n, o, dx, dy, seed)
        except Exception:
            continue
    return best

yy, xx = np.mgrid[0:SH[0], 0:SH[1]]; yy = yy.astype("float32") / 1000; xx = xx.astype("float32") / 1000
def deplane(d):
    r = d - Z; ok = ST & np.isfinite(r); idx = np.flatnonzero(ok)[::20]
    A = np.c_[np.ones(idx.size), xx.ravel()[idx], yy.ravel()[idx]]; v = r.ravel()[idx]
    for _ in range(3):
        c = np.linalg.lstsq(A, v, rcond=None)[0]; e = v - A @ c; m = np.abs(e) <= 3 * 1.4826 * np.median(np.abs(e)); A, v = A[m], v[m]
    return d - (c[0] + c[1] * xx + c[2] * yy).astype("float32")

A, B, EPOCH = {}, {}, {}; qc = []
for y in YEARS:
    dates = sorted({p.split("/")[-1][:10] for p in glob.glob(f"{BASE}/{y}/{y}-0[89]-*155_0045*")})
    for d in dates:
        fa, fb = [], []
        for sd in sorted(glob.glob(f"{BASE}/{y}/{d}*155_0045*")):
            g = None
            for t in ("DEM_FNL", "DEM_VER"):
                gg = glob.glob(f"{sd}/prc07/{t}_*.tif")
                if gg: g = gg[0]; break
            if not g: continue
            t0 = time.time(); o, nv, med = read(g); rec = dict(date=d, scene=os.path.basename(sd)[:17], valid=nv, med=round(med, 1), kept=kept.get(d, "NA"))
            if nv < 3e5 or not (800 <= med <= 2500):
                qc.append({**rec, "status": "rejected: size/median"}); continue
            b = coreg(o)
            if b is None:
                qc.append({**rec, "status": "rejected: coregistration"}); continue
            n, o2, dx, dy, seed = b
            r = (o2 - Z)[ST]; o2 = o2 - float(np.nanmedian(r)); n = nmad_stable(o2)
            if n > 8:
                qc.append({**rec, "status": "rejected: NMAD", "nmad": round(n, 2)}); continue
            o2[np.abs(o2 - Z) > 120] = np.nan
            fa.append(o2); fb.append(deplane(o2))
            qc.append({**rec, "status": "ok", "nmad": round(n, 2), "dx": round(dx, 2), "dy": round(dy, 2), "seed": seed})
            print(d, os.path.basename(sd)[:17], f"NMAD {n:.2f} dx {dx:+.1f} dy {dy:+.1f} {time.time()-t0:.0f}s", flush=True)
        if fa:
            A[d] = np.nanmedian(np.stack(fa), axis=0).astype("float32") if len(fa) > 1 else fa[0]
            B[d] = np.nanmedian(np.stack(fb), axis=0).astype("float32") if len(fb) > 1 else fb[0]
            EPOCH.setdefault(y, []).append(d)
with open(f"{OUT}/scenes.csv", "w", newline="") as f:
    keys = ["date", "scene", "valid", "med", "kept", "status", "nmad", "dx", "dy", "seed"]
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(qc)
P("dates per epoch:", {y: len(v) for y, v in EPOCH.items()})

def composite(store, dates):
    S = np.stack([store[d] for d in dates]); n = np.isfinite(S).sum(0); m = np.nanmedian(S, axis=0).astype("float32")
    m[n < min(2, len(dates))] = np.nan
    return m
CACHE = {}
def comp(var, y, gated=False, drop=None):
    ds = [d for d in EPOCH.get(y, []) if (not gated or kept.get(d) == "1") and d != drop]
    if not ds: return None
    key = (var, y, tuple(ds))
    if key not in CACHE:
        CACHE[key] = composite(A if var == "A" else B, ds)
    return CACHE[key]
def dh_of(E1, E2):
    if E1 is None or E2 is None: return None
    d = E2 - E1; d[np.abs(d) > 120] = np.nan
    v = d[ST]; v = v[np.isfinite(v)]
    return d - float(np.median(v)) if v.size > 1000 else d

def load_mask(name, kind):
    with rasterio.open(f"{RES}/ash_analysis/ASHMASK_{name}_{kind}.tif") as s:
        o = np.full(SH, 255, "uint8")
        reproject(s.read(1), o, src_transform=s.transform, src_crs=s.crs, dst_transform=TR, dst_crs=CRS, resampling=Resampling.nearest, src_nodata=None, dst_nodata=255)
    return o == 1, o == 0

def band_matched(da, za, dc, zc, step=200, nmin=30, aa=None, ac=None):
    ed = np.arange(1000, 5000 + step, step); ib, jb = np.digitize(za, ed), np.digitize(zc, ed)
    if aa is not None:
        ib = ib * 10 + (aa // 45).astype(int); jb = jb * 10 + (ac // 45).astype(int)
    order = np.argsort(jb); js = jb[order]; dcs = dc[order]; tot = w = 0.0
    for b in np.unique(ib):
        a = da[ib == b]; lo, hi = np.searchsorted(js, b, "left"), np.searchsorted(js, b, "right")
        if a.size >= nmin and hi - lo >= nmin:
            tot += a.size * (np.median(a) - np.median(dcs[lo:hi])); w += a.size
    return (tot / w if w else np.nan), w / max(da.size, 1)

def compute(dh, ash, clean, boot=0):
    ok = np.isfinite(dh) & np.isfinite(Z); a, c = ok & ash, ok & clean
    if a.sum() < 200 or c.sum() < 200: return None
    r = dict(n_ash=int(a.sum()), n_clean=int(c.sum()), ash=float(np.median(dh[a])), clean=float(np.median(dh[c])))
    r["D_raw"] = r["ash"] - r["clean"]
    r["D"], r["matched_frac"] = band_matched(dh[a], Z[a], dh[c], Z[c])
    r["D100"], _ = band_matched(dh[a], Z[a], dh[c], Z[c], step=100)
    r["D_aspect"], r["aspect_frac"] = band_matched(dh[a], Z[a], dh[c], Z[c], aa=ASPECT[a], ac=ASPECT[c])
    if boot:
        m = ash | clean; idx = np.flatnonzero((ok & m).ravel()); rr, cc = np.unravel_index(idx, SH)
        t = (rr // 50) * 100000 + cc // 50; o = np.argsort(t); idx, t = idx[o], t[o]
        v, z, isa = dh.ravel()[idx], Z.ravel()[idx], ash.ravel()[idx]
        u, st = np.unique(t, return_index=True); en = np.r_[st[1:], t.size]; rw, mt = [], []
        for _ in range(boot):
            pick = rng.integers(0, u.size, u.size); ii = np.concatenate([np.arange(st[p], en[p]) for p in pick])
            vv, zz, aa = v[ii], z[ii], isa[ii]
            if aa.sum() < 50 or (~aa).sum() < 50: continue
            rw.append(np.median(vv[aa]) - np.median(vv[~aa])); mt.append(band_matched(vv[aa], zz[aa], vv[~aa], zz[~aa])[0])
        r["D_lo"], r["D_hi"] = float(np.nanpercentile(mt, 2.5)), float(np.nanpercentile(mt, 97.5))
        r["raw_lo"], r["raw_hi"] = float(np.nanpercentile(rw, 2.5)), float(np.nanpercentile(rw, 97.5)); r["tiles"] = int(u.size)
    return r

rows = []; lodo = []
PAIRS = [(y, y + 1) for y in YEARS[:-1]]
for name, tg in TARGETS.items():
    ash, clean = load_mask(name, "strict"); ashm, cleanm = load_mask(name, "moderate")
    P(f"\n===== {name}: ash {ash.sum():,} clean {clean.sum():,}")
    tmap = {y: lab for y, lab in tg}
    results = {}
    for y1, y2 in PAIRS:
        d = dh_of(comp("A", y1), comp("A", y2))
        s = compute(d, ash, clean, boot=NBOOT) if d is not None else None
        if s is None: continue
        s.update(mask=name, pair=f"{y1}-{y2}", y1=y1, role=tmap.get(y1, "other"), dates1=len(EPOCH.get(y1, [])), dates2=len(EPOCH.get(y2, [])), variant="primary")
        rows.append(s); results[y1] = s
        P(f"  {y1}->{y2} {tmap.get(y1,''):>3} D {s['D']:+.2f} [{s['D_lo']:+.2f},{s['D_hi']:+.2f}]  D_raw {s['D_raw']:+.2f}  aspect {s['D_aspect']:+.2f}  ash {s['ash']:+.2f} clean {s['clean']:+.2f}  n_ash {s['n_ash']:,} matched {s['matched_frac']:.2f}")
    first_tg = min(tmap)
    pla = [results[y]["D"] for y in sorted(results) if y + 1 <= first_tg]   # pairs ending no later than the first target epoch
    P(f"  placebo pairs before eruption: n={len(pla)}, range {min(pla):+.2f}..{max(pla):+.2f}" if pla else "  no placebo")
    for y1, lab in tg:
        if y1 not in results: P(f"  {lab}: no result"); continue
        s = results[y1]; D = s["D"]
        vs = {"primary": D}
        d = dh_of(comp("B", y1), comp("B", y1 + 1)); x = compute(d, ash, clean); vs["plane"] = x["D"] if x else np.nan
        if x: rows.append({**x, "mask": name, "pair": f"{y1}-{y1+1}", "y1": y1, "role": lab, "variant": "plane"})
        d = dh_of(comp("A", y1, gated=True), comp("A", y1 + 1, gated=True)); x = compute(d, ash, clean) if d is not None else None; vs["trend_gated"] = x["D"] if x else np.nan
        if x: rows.append({**x, "mask": name, "pair": f"{y1}-{y1+1}", "y1": y1, "role": lab, "variant": "trend_gated"})
        d = dh_of(comp("A", y1), comp("A", y1 + 1)); x = compute(d, ashm, cleanm); vs["moderate"] = x["D"] if x else np.nan
        if x: rows.append({**x, "mask": name, "pair": f"{y1}-{y1+1}", "y1": y1, "role": lab, "variant": "moderate"})
        vs["bands100"] = s["D100"]
        lo = []
        for yy_, dates in ((y1, EPOCH[y1]), (y1 + 1, EPOCH[y1 + 1])):
            if len(dates) < 2: continue
            for dr in dates:
                E1 = comp("A", y1, drop=dr if yy_ == y1 else None); E2 = comp("A", y1 + 1, drop=dr if yy_ == y1 + 1 else None)
                x = compute(dh_of(E1, E2), ash, clean)
                if x: lo.append(x["D"]); lodo.append(dict(mask=name, role=lab, dropped=dr, D=x["D"], D_raw=x["D_raw"]))
        c1 = s["D_lo"] > 0 or s["D_hi"] < 0
        sign = np.sign(D); c2 = all(np.isfinite(v) and np.sign(v) == sign for v in vs.values())
        c3 = bool(lo) and all(np.sign(v) == sign for v in lo)
        c4 = bool(pla) and (D > max(pla) or D < min(pla))
        verdict = "CONFIRMED" if c1 and c2 and c3 and c4 else ("SUGGESTIVE" if c1 and sum((c2, c3, c4)) >= 2 else "NOT DETECTED")
        P(f"  --> {name} {lab}: D {D:+.2f} [{s['D_lo']:+.2f},{s['D_hi']:+.2f}]  variants " + " ".join(f"{k} {v:+.2f}" for k, v in vs.items()))
        P(f"      LODO {min(lo) if lo else float('nan'):+.2f}..{max(lo) if lo else float('nan'):+.2f} (n={len(lo)}); placebo {min(pla) if pla else float('nan'):+.2f}..{max(pla) if pla else float('nan'):+.2f}")
        P(f"      C1 CI excludes 0: {c1} | C2 sign stable: {c2} | C3 leave-one-out: {c3} | C4 outside placebo: {c4}  ==> {verdict}")
        if not SMOKE:
            with rasterio.open(f"{OUT}/dH_{name}_{lab}.tif", "w", driver="GTiff", height=SH[0], width=SH[1], count=1, dtype="float32", crs=CRS, transform=TR, nodata=-9999, compress="lzw") as t:
                x = dh_of(comp("A", y1), comp("A", y1 + 1)); t.write(np.where(np.isfinite(x), x, -9999).astype("float32"), 1)
keys = sorted({k for r in rows for k in r})
with open(f"{OUT}/{'smoke_' if SMOKE else ''}results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
with open(f"{OUT}/{'smoke_' if SMOKE else ''}lodo.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["mask", "role", "dropped", "D", "D_raw"]); w.writeheader(); w.writerows(lodo)
P("done")
