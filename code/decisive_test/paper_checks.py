#!/usr/bin/env python3
"""Robustness checks for the paper (read-only; writes only to this folder).
A. Epoch swap: repeat each eruption's ash-minus-clean anomaly with every late-summer DEM date.
B. Elevation matching: the same anomaly computed within 200 m bands, weighted by ash area.
C. Tile bootstrap (1.5 km tiles) of the raw and matched anomaly for the original brackets."""
import csv, glob, itertools
from datetime import date
import numpy as np, rasterio
from rasterio.warp import reproject, Resampling

ROOT = "/home/saturn/gwgi/gwgifu0h"
BASE = f"{ROOT}/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b"
MASKS = f"{ROOT}/meltrate-klyuchevskaya-glacier-tandemx/prc07_overview_out/RESULTS_presentation/ash_analysis"
OUT = f"{ROOT}/00_libs/paper_checks"
ERUPTIONS = [("2019_Oct", date(2019, 10, 25), "2019-09-27", "2020-09-02"),
             ("2020_Dec", date(2020, 12, 9), "2020-09-24", "2021-08-31"),
             ("2022_Nov", date(2022, 11, 20), "2022-10-01", "2023-08-05"),
             ("2023_PAROXYSM", date(2023, 11, 1), "2023-09-29", "2024-09-04")]
kept = {r["date"]: r["kept"] for r in csv.DictReader(open(f"{OUT}/glacier_median_series.csv"))}
rng = np.random.default_rng(7)


def late_summer(year, before=None):
    ds = sorted({p.split("/")[-1][:10] for p in glob.glob(f"{BASE}/{year}/{year}-*155_0045*")})
    ds = [d for d in ds if (8, 1) <= (int(d[5:7]), int(d[8:10])) <= (10, 15)]
    return [d for d in ds if before is None or date.fromisoformat(d) < before]


class Grid:
    def __init__(self, mask_file):
        with rasterio.open(mask_file) as s:
            self.m = s.read(1); self.tr = s.transform; self.crs = s.crs; self.shape = s.shape
        self.ash = self.m == 1; self.clean = self.m == 0; self.gm = self.ash | self.clean; self.cache = {}

    def files(self, d):
        out = []
        for sd in sorted(glob.glob(f"{BASE}/{d[:4]}/{d}*155_0045*")):
            for t in ("DEM_FNL", "DEM_VER"):
                g = glob.glob(f"{sd}/prc07/{t}_*.tif")
                if g:
                    out.append(g[0]); break
        return out

    def read(self, f):
        if f not in self.cache:
            with rasterio.open(f) as s:
                a = s.read(1).astype("float32"); a[(a == s.nodata) | (a < 0) | (a > 5000)] = np.nan
                o = np.full(self.shape, np.nan, "float32")
                reproject(a, o, src_transform=s.transform, src_crs=s.crs, dst_transform=self.tr, dst_crs=self.crs,
                          resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
            o[(o < 0) | (o > 5000)] = np.nan; self.cache[f] = o
        return self.cache[f]

    def best(self, d):
        fs = self.files(d)
        return max(fs, key=lambda f: np.isfinite(self.read(f)[self.gm]).sum()) if fs else None


def dh_field(G, f_pre, f_post):
    dh = G.read(f_post) - G.read(f_pre)
    dh = np.where(np.abs(dh) <= 120, dh, np.nan)
    stv = dh[~G.gm & np.isfinite(dh)]
    sh = float(np.nanmedian(stv)) if stv.size > 1000 else 0.0
    nmad = 1.4826 * float(np.nanmedian(np.abs(stv - sh))) if stv.size > 1000 else np.nan
    return dh - sh, nmad


def matched(ash_dh, ash_z, cl_dh, cl_z, step=200, nmin=30):
    edges = np.arange(1000, 5000 + step, step); tot = w = 0.0; used = 0
    ib = np.digitize(ash_z, edges); jb = np.digitize(cl_z, edges)
    for b in np.unique(ib):
        a = ash_dh[ib == b]; c = cl_dh[jb == b]
        if a.size >= nmin and c.size >= nmin:
            tot += a.size * (np.median(a) - np.median(c)); w += a.size; used += a.size
    return (tot / w if w else np.nan), used / max(ash_dh.size, 1)


def stats(G, dh, zref):
    ok = np.isfinite(dh) & np.isfinite(zref)
    a = ok & G.ash; c = ok & G.clean
    if a.sum() < 200 or c.sum() < 200:
        return None
    ma, mc = np.median(dh[a]), np.median(dh[c])
    dm, frac = matched(dh[a], zref[a], dh[c], zref[c])
    return dict(n_ash=int(a.sum()), n_clean=int(c.sum()), ash=float(ma), clean=float(mc), D_raw=float(ma - mc), D_elev=float(dm),
                matched_frac=float(frac), med_z_ash=float(np.median(zref[a])), med_z_clean=float(np.median(zref[c])))


def tile_boot(G, dh, zref, n=1000, size=50):
    ok = np.isfinite(dh) & np.isfinite(zref) & G.gm
    r, c = np.where(ok); tid = (r // size) * 100000 + c // size
    v, z, isash = dh[ok], zref[ok], G.ash[ok]
    order = np.argsort(tid); tid, v, z, isash = tid[order], v[order], z[order], isash[order]
    uid, start = np.unique(tid, return_index=True); stop = np.r_[start[1:], tid.size]
    raw, mat = [], []
    for _ in range(n):
        pick = rng.integers(0, uid.size, uid.size)
        idx = np.concatenate([np.arange(start[p], stop[p]) for p in pick])
        vv, zz, aa = v[idx], z[idx], isash[idx]
        if aa.sum() < 50 or (~aa).sum() < 50:
            continue
        raw.append(np.median(vv[aa]) - np.median(vv[~aa])); mat.append(matched(vv[aa], zz[aa], vv[~aa], zz[~aa])[0])
    q = lambda x: (float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5)))
    return q(raw), q(mat), uid.size


rows = []; log = open(f"{OUT}/summary.txt", "w")
def P(*a):
    print(*a, flush=True); print(*a, file=log, flush=True)


for name, ed, pre0, post0 in ERUPTIONS:
    G = Grid(f"{MASKS}/ASHMASK_{name}_strict.tif")
    P(f"\n=== {name}: strict mask {G.ash.sum()} ash px, {G.clean.sum()} clean px")
    y = ed.year
    pre = late_summer(y, before=ed); post = late_summer(y + 1)
    phases = [("Y1", pre, post)]
    if name == "2023_PAROXYSM":
        phases.append(("Y2", post, late_summer(y + 2)))
    for ph, A, B in phases:
        for a, b in itertools.product(A, B):
            fa, fb = G.best(a), G.best(b)
            if not fa or not fb:
                continue
            dh, nm = dh_field(G, fa, fb); s = stats(G, dh, G.read(fb))
            if s:
                rows.append(dict(eruption=name, phase=ph, pre=a, post=b, pre_kept=kept.get(a, "NA"), post_kept=kept.get(b, "NA"), stable_nmad=round(nm, 2), **s))
    # original bracket: same scene pick as the project scripts (first match), plus tile bootstrap
    fa, fb = G.files(pre0)[0], G.files(post0)[0]
    dh, nm = dh_field(G, fa, fb); s = stats(G, dh, G.read(fb))
    ci_raw, ci_mat, nt = tile_boot(G, dh, G.read(fb))
    P(f"original {pre0}->{post0}: ash {s['ash']:+.2f} clean {s['clean']:+.2f} D_raw {s['D_raw']:+.2f} [{ci_raw[0]:+.2f},{ci_raw[1]:+.2f}] "
      f"D_elev {s['D_elev']:+.2f} [{ci_mat[0]:+.2f},{ci_mat[1]:+.2f}] n_ash {s['n_ash']} tiles {nt} matched {s['matched_frac']:.2f} "
      f"median elev ash {s['med_z_ash']:.0f} clean {s['med_z_clean']:.0f} m, stable NMAD {nm:.2f}")
    G.cache.clear()

with open(f"{OUT}/pairs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for name, ph in sorted({(r["eruption"], r["phase"]) for r in rows}):
    R = [r for r in rows if r["eruption"] == name and r["phase"] == ph]
    K = [r for r in R if r["pre_kept"] == "1" and r["post_kept"] == "1"]
    for lab, S in (("all pairs", R), ("both DEMs pass trend gate", K)):
        if not S:
            P(f"{name} {ph} [{lab}]: none"); continue
        d = np.array([r["D_raw"] for r in S]); e = np.array([r["D_elev"] for r in S])
        P(f"{name} {ph} [{lab}] n={len(S)}: D_raw median {np.median(d):+.2f} range {d.min():+.2f}..{d.max():+.2f} pos {100*(d>0).mean():.0f}% | "
          f"D_elev median {np.median(e):+.2f} range {e.min():+.2f}..{e.max():+.2f} pos {100*(e>0).mean():.0f}%")
Y1 = [r for r in rows if r["eruption"] == "2023_PAROXYSM" and r["phase"] == "Y1"]; Y2 = [r for r in rows if r["eruption"] == "2023_PAROXYSM" and r["phase"] == "Y2"]
P("\n2023 reversal by middle epoch (mean over the other epoch):")
for mid in sorted({r["post"] for r in Y1}):
    a = [r for r in Y1 if r["post"] == mid]; b = [r for r in Y2 if r["pre"] == mid]
    if a and b:
        P(f"  {mid} (kept={kept.get(mid,'NA')}): Y1 raw {np.mean([r['D_raw'] for r in a]):+.2f} elev {np.mean([r['D_elev'] for r in a]):+.2f} | "
          f"Y2 raw {np.mean([r['D_raw'] for r in b]):+.2f} elev {np.mean([r['D_elev'] for r in b]):+.2f} | clean dH Y1 {np.mean([r['clean'] for r in a]):+.2f} Y2 {np.mean([r['clean'] for r in b]):+.2f} | ash dH Y1 {np.mean([r['ash'] for r in a]):+.2f} Y2 {np.mean([r['ash'] for r in b]):+.2f}")
trip = [(x, y) for x in Y1 for y in Y2 if x["post"] == y["pre"]]
rev = np.mean([x["D_raw"] > 0 and y["D_raw"] < 0 for x, y in trip]) * 100; rev2 = np.mean([x["D_elev"] > 0 and y["D_elev"] < 0 for x, y in trip]) * 100
P(f"\ntriples: {len(trip)}; reversal (Y1>0 and Y2<0): raw {rev:.0f}% elev-matched {rev2:.0f}%")
