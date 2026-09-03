#!/usr/bin/env python3
"""Timeline: TanDEM-X acquisition dates vs KVERT/GVP-verified Klyuchevskoy eruptions."""
import matplotlib, numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.dates import date2num
from datetime import date
from pathlib import Path
import csv

OUT = Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out")

# ── all DEM acquisition dates from the overview CSV (date col) ────────────────
acqs=[]
with open(OUT/"prc07_overview.csv") as f:
    for r in csv.DictReader(f):
        try: acqs.append(date.fromisoformat(r["date"]))
        except: pass
acqs=sorted(set(acqs))

# scenes actually used in the epochs (highlight)
USED = {
 date(2013,7,15):"S-early", date(2014,7,2):"S-early",
 date(2019,6,20):"S-late",  date(2019,7,1):"S-late",
 date(2012,2,13):"W-early", date(2017,2,12):"W-early",
 date(2024,2,8):"W-late",   date(2025,2,16):"W-late",
}

# ── eruptions (KVERT/GVP-verified) as (start,end,label,major) ────────────────
erupt=[
 (date(2010,2,27),date(2010,3,9),  "2010 expl/effus",False),
 (date(2012,10,15),date(2012,11,30),"2012 weak",False),
 (date(2013,1,25),date(2013,11,30),"2013 MAJOR (ash 10-12km)",True),
 (date(2015,1,2), date(2015,8,27),  "2015 Strombolian",False),
 (date(2019,10,25),date(2019,10,27),"2019 weak Strom.",False),
 (date(2020,12,9),date(2020,12,20),"2020 ash 7km",False),
 (date(2022,11,20),date(2022,12,15),"2022 onset",False),
 (date(2023,6,22),date(2024,1,15),  "2023-24 MAJOR + Nov23 paroxysm",True),
 (date(2025,7,30),date(2025,9,1),   "2025 summit/lava",False),
]

fig,ax=plt.subplots(figsize=(15,4.6))
# eruption spans
for s,e,lab,major in erupt:
    col="firebrick" if major else "darkorange"
    ax.axvspan(date2num(s),date2num(e),
               color=col, alpha=0.30 if major else 0.18, zorder=0)
    ax.text(date2num(s+(e-s)/2), 1.04, lab, rotation=35, ha="left",
            va="bottom", fontsize=7.5, color=col, transform=ax.get_xaxis_transform())
# all acquisitions
ax.scatter([date2num(d) for d in acqs],[0]*len(acqs),
           s=28, c="0.45", zorder=3, label=f"all DEMs (n={len(acqs)})")
# epoch-used scenes
cmap={"S-early":"#1f77b4","S-late":"#17becf","W-early":"#9467bd","W-late":"#e377c2"}
seen=set()
for d,grp in USED.items():
    ax.scatter(date2num(d),0,s=120,marker="^",color=cmap[grp],
               edgecolor="k",linewidth=0.6,zorder=5,
               label=(grp if grp not in seen else None)); seen.add(grp)
ax.set_yticks([]); ax.set_ylim(-0.5,0.6)
import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_xlim(date2num(date(2011,6,1)),date2num(date(2026,3,1)))
ax.set_title("Klyuchevskoy: TanDEM-X acquisitions vs eruptions (KVERT/GVP-verified)\n"
             "red span = major eruption, orange = minor; triangles = epoch scenes")
ax.legend(loc="lower center",ncol=6,fontsize=8,frameon=True)
ax.grid(axis="x",alpha=0.3)
out=OUT/"PREVIEW_acquisitions_vs_eruptions.png"
fig.savefig(out,dpi=140,bbox_inches="tight"); print("->",out.name)
