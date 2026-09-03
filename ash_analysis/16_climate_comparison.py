#!/usr/bin/env python3
"""
STEP 16: Compare DEM-observed seasonal signals with climate reanalysis.

From the glacier-median series (step 15 CSV, main massif, outlier-cleaned):
  * WITHIN-SUMMER surface lowering per year  = linear fit over that year's
    Jun-Sep points (needs >=3 dates spanning >=40 d)  -> "summer melt" [m]
    (elevation lowering; includes snow ablation + compaction, not pure ice melt)
  * APPARENT WINTER ACCUMULATION before that summer = median winter anomaly
    (Nov Y-1 .. Apr Y) minus the LAST late-summer anomaly of year Y-1.
    CAVEAT: winter DEMs are penetration-biased low -> this UNDERESTIMATES
    accumulation by roughly the penetration depth (~0.4 m median here).

Climate: ERA5 reanalysis (via the Open-Meteo archive API), daily 2 m mean
temperature downscaled to 1250 m (approx. glacier median elevation) and daily
snowfall, 2012-2025 at 56.06N 160.63E:
  * PDD  = positive-degree-day sum Jun 1 - Sep 30 [degC*d]
  * SNOW = snowfall sum Oct(Y-1) - Apr(Y) [cm]
Outputs: MELT_vs_PDD scatter (r, p), ACCUM_vs_MELT per-year panel, annual CSV.
"""
import csv, json, urllib.request, urllib.parse
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datetime import date
from pathlib import Path
from scipy import stats

AOUT=Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation")
SER=AOUT/"glacier_median_series.csv"
LAT,LON,ELEV=56.06,160.63,1250

# ---- read series ----
S=[]; W=[]
with open(SER) as f:
    for r in csv.DictReader(f):
        if r["kept"]!="1": continue
        d=date.fromisoformat(r["date"]); a=float(r["anom_m"])
        (S if r["season"]=="S" else W).append((d,a))
S.sort(); W.sort()
print(f"series: {len(S)} summer, {len(W)} winter points")

# ---- within-summer lowering per year (span recorded for PDD matching) ----
melt={}; span={}
for y in range(2012,2026):
    pts=[(d,a) for d,a in S if d.year==y and 6<=d.month<=9]
    if len(pts)<3: continue
    days=np.array([(d-date(y,1,1)).days for d,_ in pts],float)
    if days.max()-days.min()<30: continue
    ys=np.array([a for _,a in pts])
    sl=np.polyfit(days,ys,1)[0]                     # m/day
    melt[y]=-sl*(days.max()-days.min())             # + = lowering over observed span
    span[y]=(min(d for d,_ in pts),max(d for d,_ in pts))
    print(f"  {y}: summer lowering {melt[y]:+.2f} m over {days.max()-days.min():.0f} d ({len(pts)} pts)")

# ---- apparent winter accumulation before each summer ----
accum={}
for y in sorted(melt):
    wpts=[a for d,a in W if (d.year==y and d.month<=4) or (d.year==y-1 and d.month>=11)]
    spre=[a for d,a in S if d.year==y-1 and d.month>=8]
    if wpts and spre:
        accum[y]=float(np.median(wpts)-spre[-1])
        print(f"  {y}: apparent winter accumulation {accum[y]:+.2f} m (penetration-biased low)")

# ---- ERA5 via Open-Meteo archive (multivariate) ----
DAILY="temperature_2m_mean,snowfall_sum,rain_sum,precipitation_sum,shortwave_radiation_sum,wind_speed_10m_max"
q=urllib.parse.urlencode(dict(latitude=LAT,longitude=LON,elevation=ELEV,
    start_date="2012-01-01",end_date="2025-12-31",daily=DAILY,timezone="UTC"))
url=f"https://archive-api.open-meteo.com/v1/archive?{q}"
with urllib.request.urlopen(url,timeout=180) as r: met=json.load(r)
dts=[date.fromisoformat(x) for x in met["daily"]["time"]]
def arr(k,fill=np.nan):
    return np.array([fill if v is None else v for v in met["daily"][k]],float)
T=arr("temperature_2m_mean"); SN=arr("snowfall_sum",0); RAIN=arr("rain_sum",0)
PRC=arr("precipitation_sum",0); SW=arr("shortwave_radiation_sum",0); WND=arr("wind_speed_10m_max")
print(f"ERA5 daily records: {len(dts)} (elev {met.get('elevation','?')} m)")
pdd={}; snow={}; pdd_span={}
mvar={}   # melt-season (span-matched) drivers per year
avar={}   # accumulation-season (Oct-Apr) drivers per year
for y in range(2012,2026):
    m=np.array([(d.year==y and 6<=d.month<=9) for d in dts])
    pdd[y]=float(np.nansum(np.clip(T[m],0,None)))
    m2=np.array([((d.year==y-1 and d.month>=10) or (d.year==y and d.month<=4)) for d in dts])
    snow[y]=float(np.nansum(SN[m2]))
    avar[y]=dict(snow_cm=snow[y], precip_mm=float(np.nansum(PRC[m2])),
                 t_mean=float(np.nanmean(T[m2])), wind_max=float(np.nanmean(WND[m2])))
    if y in span:   # drivers over EXACTLY the DEM observation window (apples-to-apples)
        s0,s1=span[y]
        ms=np.array([(s0<=d<=s1) for d in dts])
        pdd_span[y]=float(np.nansum(np.clip(T[ms],0,None)))
        ndays=max(1,int(ms.sum()))
        mvar[y]=dict(pdd=pdd_span[y]/ndays, sw=float(np.nanmean(SW[ms])),
                     rain=float(np.nansum(RAIN[ms]))/ndays, wind=float(np.nanmean(WND[ms])))
        # per-day normalisation -> comparable across different window lengths

# ---- annual table ----
years=sorted(melt)
with open(AOUT/"CLIMATE_annual_table.csv","w",newline="") as f:
    wcs=csv.writer(f); wcs.writerow(["year","summer_lowering_m","apparent_accum_m","PDD_degCd","snowfall_OctApr_cm"])
    for y in years:
        wcs.writerow([y,f"{melt[y]:.2f}",f"{accum.get(y,float('nan')):.2f}",f"{pdd[y]:.0f}",f"{snow[y]:.0f}"])
print("-> CLIMATE_annual_table.csv")

# ---- figure 1: melt vs SPAN-MATCHED PDD scatter ----
years_s=[y for y in years if y in pdd_span]
xs=np.array([pdd_span[y] for y in years_s]); ys=np.array([melt[y] for y in years_s])
r,p=stats.pearsonr(xs,ys)
fig,ax=plt.subplots(figsize=(7.5,6))
ax.scatter(xs,ys,s=70,c="#e0662a",edgecolor='k',lw=0.5,zorder=3)
for y,xx,yy in zip(years_s,xs,ys): ax.annotate(str(y),(xx,yy),textcoords="offset points",xytext=(6,4),fontsize=8.5)
if len(xs)>2:
    b=np.polyfit(xs,ys,1); xx=np.linspace(xs.min(),xs.max(),10)
    ax.plot(xx,np.polyval(b,xx),'-',color="#9a4a1e",lw=1.8)
ax.set_xlabel("ERA5 positive–degree–days over the SAME dates as the DEM window [°C·d] (~1250 m)")
ax.set_ylabel("observed summer surface lowering [m]")
ax.set_title(f"Summer melt vs reanalysis temperature (span–matched)\nPearson r = {r:+.2f} (p = {p:.3f}, n = {len(xs)})\n"
             "lowering from within–summer DEM series, main massif, outlier–cleaned")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(AOUT/"MELT_vs_PDD.png",dpi=600,bbox_inches="tight")
print(f"-> MELT_vs_PDD.png   r={r:+.2f} p={p:.3f} (span-matched, n={len(xs)})")

# ---- figure 2: per-year accumulation vs melt + climate panel ----
fig,(a1,a2)=plt.subplots(2,1,figsize=(12,8),sharex=True,gridspec_kw={'height_ratios':[1.2,0.8]})
xpos=np.arange(len(years)); wdt=0.38
av=[accum.get(y,np.nan) for y in years]
a1.bar(xpos-wdt/2,av,wdt,color="#1f77b4",label="apparent winter accumulation (DEM, biased low)")
a1.bar(xpos+wdt/2,[-melt[y] for y in years],wdt,color="#e0662a",label="summer surface change (− = lowering)")
a1.axhline(0,color='k',lw=0.7)
a1.set_ylabel("elevation change [m]")
a1.set_title("Winter accumulation vs following–summer lowering — Klyuchevskaya main massif\n"
             "(winter bars underestimate true accumulation by the X–band penetration depth)")
a1.legend(fontsize=9); a1.grid(alpha=0.25,axis='y')
a2.bar(xpos,[snow[y] for y in years],0.5,color="#8ab4d2",label="ERA5 snowfall Oct–Apr [cm]")
a2.set_ylabel("snowfall [cm]"); a2.legend(loc='upper left',fontsize=9); a2.grid(alpha=0.25,axis='y')
a2b=None  # no dual axis: PDD shown as text labels instead
for i,y in enumerate(years):
    a2.text(i,snow[y]+8,f"PDD\n{pdd[y]:.0f}",ha='center',fontsize=7.5,color="#7a3b12")
a2.set_xticks(xpos); a2.set_xticklabels([str(y) for y in years])
fig.tight_layout(); fig.savefig(AOUT/"ACCUM_vs_MELT_per_year.png",dpi=600,bbox_inches="tight")
print("-> ACCUM_vs_MELT_per_year.png")

# ---- MULTIVARIATE: which climate drivers correlate with melt / accumulation? ----
# Spearman rank correlation (robust for small n); per-day-normalised melt drivers.
ymv=[y for y in years if y in mvar]
melt_rate=np.array([melt[y]/max(1,(span[y][1]-span[y][0]).days) for y in ymv])*1000  # mm/day
M=dict(  # melt-season drivers, per day
    **{"PDD [°C/d]":np.array([mvar[y]["pdd"] for y in ymv]),
       "shortwave [MJ/m²/d]":np.array([mvar[y]["sw"] for y in ymv]),
       "rain [mm/d]":np.array([mvar[y]["rain"] for y in ymv]),
       "wind max [m/s]":np.array([mvar[y]["wind"] for y in ymv])})
yav=[y for y in years if y in accum]
acc=np.array([accum[y] for y in yav])
A=dict(**{"snowfall [cm]":np.array([avar[y]["snow_cm"] for y in yav]),
          "precip [mm]":np.array([avar[y]["precip_mm"] for y in yav]),
          "T mean [°C]":np.array([avar[y]["t_mean"] for y in yav]),
          "wind max [m/s]":np.array([avar[y]["wind_max"] for y in yav])})
print("\n=== Spearman correlations ===")
rows=[]
for lab,vars_,target,tn in [("summer melt rate [mm/d]",M,melt_rate,len(ymv)),
                            ("apparent accumulation [m]",A,acc,len(yav))]:
    rr=[]
    for k,v in vars_.items():
        r,p=stats.spearmanr(v,target)
        rr.append((k,r,p)); print(f"  {lab} vs {k}: rho={r:+.2f} p={p:.3f} (n={tn})")
    rows.append((lab,rr))
fig,axes=plt.subplots(2,1,figsize=(9,5.5))
for ax,(lab,rr) in zip(axes,rows):
    vals=[r for _,r,_ in rr]
    cols=['#2f7cb2' if v>0 else '#c0463a' for v in vals]
    ax.barh(range(len(rr)),vals,color=cols,height=0.55)
    ax.set_yticks(range(len(rr))); ax.set_yticklabels([k for k,_,_ in rr],fontsize=9)
    for i,(k,r,p) in enumerate(rr):
        ax.text(r+(0.03 if r>=0 else -0.03),i,f"ρ={r:+.2f}{' *' if p<0.05 else ''} (p={p:.2f})",
                va='center',ha='left' if r>=0 else 'right',fontsize=8.5)
    ax.axvline(0,color='k',lw=0.8); ax.set_xlim(-1.15,1.15); ax.grid(alpha=0.25,axis='x')
    ax.set_title(lab,fontsize=10,loc='left')
fig.suptitle("Climate drivers vs DEM–observed seasonal signals (Spearman rank)\n"
             "small n — treat as exploratory; * = p<0.05",fontsize=11)
fig.tight_layout(); fig.savefig(AOUT/"CLIMATE_correlations.png",dpi=600,bbox_inches="tight")
print("-> CLIMATE_correlations.png")
