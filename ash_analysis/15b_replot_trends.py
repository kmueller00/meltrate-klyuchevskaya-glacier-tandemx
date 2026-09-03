#!/usr/bin/env python3
"""Replot trend figures from glacier_median_series.csv (no recompute).
Spline is fit through YEARLY SEASON-MEDIANS (interannual granularity) to avoid
the GCV overfit that occurs on clustered within-season dates."""
import csv, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import date2num
from datetime import date
from pathlib import Path
from collections import defaultdict
from scipy import stats

AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation")
ERUPT=[(date(2013,1,25),date(2013,11,30),1),(date(2015,1,2),date(2015,3,24),0),
 (date(2015,8,27),date(2015,9,10),0),(date(2016,4,1),date(2016,5,1),0),
 (date(2019,4,1),date(2019,11,1),1),(date(2020,10,2),date(2021,3,25),1),
 (date(2022,11,20),date(2022,12,20),0),(date(2023,6,22),date(2024,1,15),1),
 (date(2024,8,1),date(2024,10,15),0)]
S_in=[];S_out=[];W_in=[];W_out=[]
with open(AOUT/"glacier_median_series.csv") as f:
    for r in csv.DictReader(f):
        d=date.fromisoformat(r["date"]); a=float(r["anom_m"]); k=r["kept"]=="1"
        t=(date2num(d),a,d)
        if r["season"]=="S": (S_in if k else S_out).append(t)
        else: (W_in if k else W_out).append(t)

def linreject(pin,pout,k=2.5,iters=2):
    """2nd-stage outlier pass: iteratively reject points >k*NMAD from the season's
    own LINEAR fit (robust-regression style). Rejected points join the outlier set."""
    for _ in range(iters):
        if len(pin)<6: break
        x=np.array([p[0] for p in pin]); y=np.array([p[1] for p in pin])
        A=np.vstack([x,np.ones(len(x))]).T
        c,_,_,_=np.linalg.lstsq(A,y,rcond=None)
        r=y-A@c; nm=1.4826*np.median(np.abs(r-np.median(r)))
        keep=np.abs(r-np.median(r))<=k*max(nm,0.25)
        new=[p for p,kk in zip(pin,keep) if not kk]
        if not new: break
        for p in new: print(f"  LINFIT-OUTLIER {p[2]}: {p[1]:+.2f} m")
        pout+=new; pin=[p for p,kk in zip(pin,keep) if kk]
    return pin,pout

def linfit(pts):
    x=np.array([p[0] for p in pts])/365.25; y=np.array([p[1] for p in pts]); n=len(x)
    A=np.vstack([x,np.ones(n)]).T
    coef,_,_,_=np.linalg.lstsq(A,y,rcond=None)
    s2=np.sum((y-A@coef)**2)/(n-2); sx=np.sqrt(s2/np.sum((x-x.mean())**2))
    return coef[0],stats.t.ppf(0.975,n-2)*sx,coef

def yearly_medians(pts):
    by=defaultdict(list)
    for x,a,d in pts: by[d.year].append((x,a))
    out=[(float(np.median([x for x,_ in v])),float(np.median([a for _,a in v]))) for y,v in sorted(by.items())]
    return out

def spline_fit(pts):
    ym=yearly_medians(pts)
    x=np.array([p[0] for p in ym]); y=np.array([p[1] for p in ym])
    from scipy.interpolate import make_smoothing_spline
    sp=make_smoothing_spline(x,y)
    xs=np.linspace(x[0],x[-1],300); return xs,sp(xs),ym

print("summer 2nd-stage:"); S_in,S_out=linreject(S_in,S_out)
print("winter 2nd-stage:"); W_in,W_out=linreject(W_in,W_out)
print(f"final kept: {len(S_in)} summer, {len(W_in)} winter")

COL={"S":"#e0662a","W":"#1f77b4"}
def base_ax(ax):
    for s,e,major in ERUPT:
        ax.axvspan(date2num(s),date2num(e),color=('firebrick' if major else 'orange'),
                   alpha=0.22 if major else 0.11,zorder=0)
    ax.axhline(0,color='k',lw=0.5,ls=':'); ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(date2num(date(2012,1,1)),date2num(date(2026,3,1)))
    ax.set_ylabel("glacier–median elevation anomaly [m]\n(vs 2025 ref; main massif only)")

for MODE in ("linear","spline"):
    fig,ax=plt.subplots(figsize=(15,6)); base_ax(ax)
    for pin,pout,se,lab in [(S_in,S_out,"S","summer (surface)"),(W_in,W_out,"W","winter (penetration-biased)")]:
        ax.scatter([p[0] for p in pin],[p[1] for p in pin],s=55,c=COL[se],
                   marker='^' if se=="S" else 'o',edgecolor='k',lw=0.4,zorder=3,label=lab)
        if pout:
            ax.scatter([p[0] for p in pout],[p[1] for p in pout],s=55,facecolor='none',
                       marker='^' if se=="S" else 'o',edgecolor=COL[se],lw=1.2,zorder=3,
                       label=f"removed outlier ({'summer' if se=='S' else 'winter'})")
        if len(pin)>=5:
            if MODE=="linear":
                sl,ci,coef=linfit(pin)
                xs=np.array([min(p[0] for p in pin),max(p[0] for p in pin)])
                ax.plot(xs,coef[0]*xs/365.25+coef[1],'-',color=COL[se],lw=2.2,zorder=2)
                ax.text(0.99,0.97 if se=="S" else 0.90,
                        f"{'summer' if se=='S' else 'winter'} trend: {sl:+.2f} ± {ci:.2f} m/yr",
                        transform=ax.transAxes,ha='right',va='top',fontsize=10,color=COL[se],fontweight='bold')
            else:
                xs,ys,ym=spline_fit(pin)
                ax.plot(xs,ys,'-',color=COL[se],lw=2.2,zorder=2)
                ax.scatter([p[0] for p in ym],[p[1] for p in ym],s=110,facecolor='none',
                           edgecolor=COL[se],lw=1.6,zorder=2,marker='s')
    if MODE=="linear":
        ax.set_title("Klyuchevskaya main massif — glacier elevation 2012–2025, LINEAR trend per season\n"
                     "outliers (open symbols) rejected within–season (3×NMAD vs rolling median); ± = 95% CI")
        out=AOUT/"TREND_linear_2012-2025.png"
    else:
        ax.set_title("Klyuchevskaya main massif — glacier elevation 2012–2025, SMOOTHING SPLINE per season\n"
                     "GCV–penalized spline through YEARLY season–medians (squares) — interannual, descriptive")
        out=AOUT/"TREND_spline_2012-2025.png"
    ax.legend(loc='lower left',fontsize=9)
    fig.tight_layout(); fig.savefig(out,dpi=600,bbox_inches="tight")
    print(f"-> {out.name}")
