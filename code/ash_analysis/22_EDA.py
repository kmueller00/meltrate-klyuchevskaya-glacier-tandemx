#!/usr/bin/env python3
"""
STEP 22: Exploratory data analysis of the glacier-median elevation series.
Reads glacier_median_series.csv (main massif, per-date anomalies vs 2025 ref).
  (a) data inventory: observations per year x season
  (b) distribution of anomalies, summer vs winter (the penetration offset visible)
  (c) OUTLIER-THRESHOLD SENSITIVITY: linear trend slope as the rejection cutoff
      varies 2.0 - 3.5 x NMAD -> shows the trend is insensitive to the (2.5) choice
  (d) within-summer surface-lowering rate per year (the melt signal)
Outputs: EDA_overview.png + a short text report.
"""
import csv, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datetime import date
from collections import defaultdict
from scipy import stats

AOUT="/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation"
rows=[]
with open(f"{AOUT}/glacier_median_series.csv") as f:
    for r in csv.DictReader(f):
        rows.append((date.fromisoformat(r["date"]),r["season"],float(r["anom_m"]),r["kept"]=="1"))
S=[(d,a) for d,se,a,k in rows if se=="S"]        # ALL summer (incl. flagged) for EDA
W=[(d,a) for d,se,a,k in rows if se=="W"]
print(f"{len(rows)} observations: {len(S)} summer, {len(W)} winter")

# ---- (c) outlier-threshold sensitivity on the LINEAR trend ----
def robust_slope(pts,kthr):
    x=np.array([d.year+d.timetuple().tm_yday/365.25 for d,_ in pts]); y=np.array([a for _,a in pts])
    for _ in range(3):
        if len(x)<6: break
        A=np.vstack([x,np.ones(len(x))]).T; c,_,_,_=np.linalg.lstsq(A,y,rcond=None)
        r=y-A@c; nm=1.4826*np.median(np.abs(r-np.median(r)))
        keep=np.abs(r-np.median(r))<=kthr*max(nm,0.25)
        if keep.all(): break
        x,y=x[keep],y[keep]
    A=np.vstack([x,np.ones(len(x))]).T; c,_,_,_=np.linalg.lstsq(A,y,rcond=None)
    s2=np.sum((y-A@c)**2)/(len(x)-2); se=np.sqrt(s2/np.sum((x-x.mean())**2))
    return c[0],1.96*se,len(x)
ks=np.arange(2.0,3.6,0.25)
sens={se:[robust_slope(P,k) for k in ks] for se,P in [("S",S),("W",W)]}
print("\n=== outlier-threshold sensitivity (linear slope, m/yr) ===")
for se in ("S","W"):
    print(f"  {se}: "+"  ".join(f"{k:.2f}σ:{v[0]:+.2f}±{v[1]:.2f}(n{v[2]})" for k,v in zip(ks,sens[se])))

# ---- (d) within-summer lowering per year ----
melt={}
for y in range(2012,2026):
    p=[(d,a) for d,a in S if d.year==y and 6<=d.month<=9]
    if len(p)<3: continue
    dd=np.array([(d-date(y,1,1)).days for d,_ in p]); yy=np.array([a for _,a in p])
    if dd.max()-dd.min()<30: continue
    melt[y]=-np.polyfit(dd,yy,1)[0]*(dd.max()-dd.min())

# ---- figure ----
fig=plt.figure(figsize=(15,10)); gs=fig.add_gridspec(2,2,hspace=0.28,wspace=0.22)
# (a) inventory
axa=fig.add_subplot(gs[0,0])
yrs=list(range(2012,2026))
cs=defaultdict(int); cw=defaultdict(int)
for d,a in S: cs[d.year]+=1
for d,a in W: cw[d.year]+=1
axa.bar(yrs,[cs[y] for y in yrs],label="summer",color="#e0662a")
axa.bar(yrs,[cw[y] for y in yrs],bottom=[cs[y] for y in yrs],label="winter",color="#2f7cb2")
axa.set_title("(a) observations per year"); axa.set_ylabel("# DEM dates"); axa.legend(fontsize=9); axa.grid(alpha=0.2,axis='y')
# (b) distribution
axb=fig.add_subplot(gs[0,1])
axb.hist([a for _,a in S],bins=np.arange(-2,11,0.6),alpha=0.6,color="#e0662a",label=f"summer (n={len(S)})")
axb.hist([a for _,a in W],bins=np.arange(-2,11,0.6),alpha=0.6,color="#2f7cb2",label=f"winter (n={len(W)})")
axb.axvline(np.median([a for _,a in S]),color="#e0662a",lw=2,ls='--')
axb.axvline(np.median([a for _,a in W]),color="#2f7cb2",lw=2,ls='--')
axb.set_title("(b) anomaly distribution\n(winter shifted low = X–band penetration)")
axb.set_xlabel("elevation anomaly [m] vs 2025 ref"); axb.legend(fontsize=9); axb.grid(alpha=0.2)
# (c) sensitivity
axc=fig.add_subplot(gs[1,0])
for se,col,mk in [("S","#e0662a","^"),("W","#2f7cb2","o")]:
    sl=[v[0] for v in sens[se]]; er=[v[1] for v in sens[se]]
    axc.errorbar(ks,sl,yerr=er,fmt=mk+'-',color=col,capsize=3,label=f"{'summer' if se=='S' else 'winter'}")
axc.axvline(2.5,color='k',lw=1,ls=':'); axc.text(2.5,axc.get_ylim()[0]," used",fontsize=8,rotation=90,va='bottom')
axc.set_title("(c) trend slope vs outlier–rejection threshold\n→ slope is insensitive to the cutoff choice")
axc.set_xlabel("rejection threshold [× NMAD]"); axc.set_ylabel("linear slope [m/yr]")
axc.legend(fontsize=9); axc.grid(alpha=0.2)
# (d) melt per year
axd=fig.add_subplot(gs[1,1])
my=sorted(melt); axd.bar(my,[melt[y] for y in my],color="#9a4a1e")
axd.set_title("(d) within–summer surface lowering per year"); axd.set_ylabel("lowering over obs window [m]")
axd.set_xlabel("year"); axd.grid(alpha=0.2,axis='y')
fig.suptitle("Exploratory data analysis — Klyuchevskaya glacier–median elevation series (main massif)",fontsize=14)
fig.savefig(f"{AOUT}/EDA_overview.png",dpi=600,bbox_inches="tight")
print("-> EDA_overview.png")
