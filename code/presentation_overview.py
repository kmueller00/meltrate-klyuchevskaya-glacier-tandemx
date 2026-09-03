#!/usr/bin/env python3
"""
Presentation overview figures for coworker:
  Fig A: Data inventory + processing status (what was processed, what failed & why)
  Fig B: Per-scene quality boxplots/distributions (NMAD, outlier fractions)
  Fig C: Coregistration shifts overview
Reads prc07_overview.csv (the per-scene stats).
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import csv
from pathlib import Path
from datetime import date
OUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out")

# ── read per-scene stats ──────────────────────────────────────────────────────
rows=[]
with open(OUT/"prc07_overview.csv") as f:
    for r in csv.DictReader(f):
        rows.append(r)
def fnum(r,k):
    try: return float(r[k])
    except: return np.nan
# parse
data=[]
for r in rows:
    nm=r['acquisition']
    track='ASC (155)' if '155_0045' in nm else 'DESC (011)'
    try: y=int(nm[:4])
    except: continue
    data.append(dict(date=nm[:10],year=y,track=track,
        nmad=fnum(r,'dh_nmad'), outl=fnum(r,'outlier_frac_%'),
        dx=fnum(r,'dx_m'),dy=fnum(r,'dy_m'),dz=fnum(r,'dz_m'),
        status=r.get('status','')))

# ===== FIG A: data inventory timeline (processed vs failed, by track) =========
fig,ax=plt.subplots(figsize=(15,5))
# known 2026 GAMMA-failed dates + the empty-SLC one
FAILED_2026=['2025-12-21','2026-01-01','2026-01-12','2026-01-23','2026-02-03']
GEO_FAIL=['2016-02-04','2016-02-15']  # geocode step
yr_track={'ASC (155)':1,'DESC (011)':0}
for d in data:
    try: dt=date.fromisoformat(d['date'])
    except: continue
    import matplotlib.dates as md
    x=md.date2num(dt); yy=yr_track[d['track']]
    bad = d['outl']>5 if np.isfinite(d['outl']) else False
    col = 'firebrick' if bad else ('#1f77b4' if 'ASC' in d['track'] else '#d62728')
    ax.scatter(x,yy,s=70,c=col,edgecolor='k',lw=0.4,zorder=3,
               marker=('x' if bad else 'o'))
# mark the failed 2026 region
import matplotlib.dates as md
ax.axvspan(md.date2num(date(2025,12,1)),md.date2num(date(2026,3,1)),color='red',alpha=0.12)
ax.text(md.date2num(date(2026,1,1)),1.35,'2026 dates:\nGAMMA SLC_cat\nfailure (unsolved)',
        ha='center',fontsize=8,color='darkred')
ax.set_yticks([0,1]); ax.set_yticklabels(['DESCENDING\n(011)','ASCENDING\n(155)'])
ax.set_ylim(-0.6,1.7)
ax.xaxis.set_major_locator(md.YearLocator()); ax.xaxis.set_major_formatter(md.DateFormatter("%Y"))
ax.set_xlim(md.date2num(date(2011,6,1)),md.date2num(date(2026,6,1)))
ax.set_title("TanDEM-X DEM inventory by track  (blue/red=OK scene, ✗=high-outlier/excluded)")
ax.grid(axis='x',alpha=0.3)
fig.tight_layout(); fig.savefig(OUT/"PRES_A_inventory.png",dpi=140,bbox_inches="tight")
print("-> PRES_A_inventory.png")

# ===== FIG B: quality boxplots (NMAD & outlier% by track) =====================
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,6))
for ax,key,lab,ylim in [(a1,'nmad','DEM vs COP30 NMAD [m]',(0,30)),
                        (a2,'outl','outlier fraction [%]',(0,15))]:
    groups,labels=[],[]
    for tr in ['ASC (155)','DESC (011)']:
        vals=[d[key] for d in data if d['track']==tr and np.isfinite(d[key]) and d[key]<1000]
        if vals: groups.append(vals); labels.append(f"{tr}\n(n={len(vals)})")
    bp=ax.boxplot(groups,labels=labels,patch_artist=True,showfliers=True)
    for p,c in zip(bp['boxes'],['#1f77b4','#d62728']): p.set_facecolor(c); p.set_alpha(0.5)
    ax.set_ylabel(lab); ax.set_ylim(*ylim); ax.grid(axis='y',alpha=0.3)
    ax.set_title(lab.split('[')[0])
fig.suptitle("Per-scene DEM quality (Δ to Copernicus 30 m reference)",fontsize=12)
fig.tight_layout(); fig.savefig(OUT/"PRES_B_quality_boxplots.png",dpi=140,bbox_inches="tight")
print("-> PRES_B_quality_boxplots.png")

# ===== FIG C: coregistration shifts =========================================
fig,ax=plt.subplots(figsize=(11,6))
for tr,col in [('ASC (155)','#1f77b4'),('DESC (011)','#d62728')]:
    dxs=[d['dx'] for d in data if d['track']==tr and np.isfinite(d['dx'])]
    dys=[d['dy'] for d in data if d['track']==tr and np.isfinite(d['dy'])]
    ax.scatter(dxs,dys,c=col,alpha=0.6,s=50,edgecolor='k',lw=0.3,label=f"{tr} (n={len(dxs)})")
ax.axhline(0,color='k',lw=0.5,ls=':'); ax.axvline(0,color='k',lw=0.5,ls=':')
ax.set_xlabel("dx shift [m]"); ax.set_ylabel("dy shift [m]")
ax.set_title("Co-registration shifts (Nuth & Kääb) per scene"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT/"PRES_C_coreg_shifts.png",dpi=140,bbox_inches="tight")
print("-> PRES_C_coreg_shifts.png")

# ===== summary stats table (printed + csv) ===================================
print("\n=== SUMMARY STATS ===")
with open(OUT/"PRES_summary_stats.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["track","n_scenes","median_NMAD_m","median_outlier_pct","high_outlier_scenes"])
    for tr in ['ASC (155)','DESC (011)']:
        ns=[d for d in data if d['track']==tr]
        nm=[d['nmad'] for d in ns if np.isfinite(d['nmad']) and d['nmad']<1000]
        ot=[d['outl'] for d in ns if np.isfinite(d['outl'])]
        hi=sum(1 for d in ns if np.isfinite(d['outl']) and d['outl']>5)
        print(f"  {tr}: n={len(ns)}  medNMAD={np.median(nm):.1f}m  medOutlier={np.median(ot):.1f}%  high-outlier={hi}")
        w.writerow([tr,len(ns),round(np.median(nm),1),round(np.median(ot),2),hi])
print("-> PRES_summary_stats.csv")
