#!/usr/bin/env python3
"""
STEP 1: For each Klyuchevskaya eruption, find the best DEM + cloudless Sentinel-2
acquisition CLOSEST BEFORE and CLOSEST AFTER the eruption.

Outputs a match table (CSV) pairing:
  eruption | DEM_before | DEM_after | S2_before | S2_after
Later steps use S2 to map ash/lava extent and DEM dH to test elevation change.
"""
import glob, re, csv, json
from datetime import date, timedelta
from pathlib import Path
import rasterio
import pystac_client, planetary_computer

OUT = Path("/home/student/Desktop/_0_Korbinian_TANDEM-X/_code/prc07_overview_out/RESULTS_presentation/ash_analysis")
OUT.mkdir(parents=True, exist_ok=True)
DEMBASE = Path("/media/saturn/01_TDX_data/utm_CP30/10_NAS/reg_utm57_N_b/DEM-utm57_N_b")
BBOX = [159.889, 55.472, 161.359, 56.522]   # lon/lat study area
MAX_CLOUD = 5.0     # Sentinel-2 must be near-cloudless (<5%) — user wants completely cloudless
S2_WINDOW_DAYS = 120   # search this many days before/after eruption for a clear S2

# ── KVERT/GVP-verified eruptions (start dates; label; major?) ─────────────────
ERUPTIONS = [
    (date(2015,1,1),   "2015 Jan-Mar eruption",True),   # major summit+flank (KVERT/GVP)
    (date(2015,8,25),  "2015 Aug-Sep eruption",False),  # renewed Strombolian/lava
    (date(2019,10,25), "2019 Oct Strombolian", False),
    (date(2020,12,9),  "2020 Dec eruption",   False),
    (date(2022,11,20), "2022 Nov eruption",   False),
    (date(2023,11,1),  "2023 Nov PAROXYSM",   True),   # the big one
    (date(2024,8,15),  "2024 Aug-Oct eruption",False),
]

# ── available DEMs (any track; prefer 155 ascending, fall back 011) ───────────
def dem_list():
    out=[]
    for yd in sorted(DEMBASE.glob("20*")):
        for sd in sorted(yd.glob("20*")):
            nm=sd.name; m=re.match(r"(\d{4})-(\d{2})-(\d{2})",nm)
            if not m: continue
            for t in ("DEM_FNL","DEM_VER"):
                g=glob.glob(str(sd/"prc07"/f"{t}_*.tif"))
                if g:
                    track="155" if "155_0045" in nm else ("011" if "011_0045" in nm else "?")
                    out.append((date(*map(int,m.groups())), g[0], track, t[-3:]))
                    break
    return sorted(set(out))
DEMS=dem_list()
print(f"available DEMs: {len(DEMS)}")

def closest_dem(erupt, before=True):
    cand=[(d,f,tr,pr) for d,f,tr,pr in DEMS if (d<erupt if before else d>erupt)]
    if not cand: return None
    # closest in time; prefer 155 track if tie within 20 days
    cand.sort(key=lambda x: abs((x[0]-erupt).days))
    best=cand[0]
    for c in cand[:4]:
        if c[2]=="155" and abs((c[0]-best[0]).days)<=20: best=c; break
    return best

# ── Sentinel-2 search (Planetary Computer) ────────────────────────────────────
CAT = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1',
                                modifier=planetary_computer.sign_inplace)
def closest_s2(erupt, before=True):
    if before:
        dt=f"{(erupt-timedelta(days=S2_WINDOW_DAYS)).isoformat()}/{erupt.isoformat()}"
    else:
        dt=f"{erupt.isoformat()}/{(erupt+timedelta(days=S2_WINDOW_DAYS)).isoformat()}"
    s=CAT.search(collections=['sentinel-2-l2a'], bbox=BBOX, datetime=dt,
                 query={'eo:cloud_cover':{'lt':MAX_CLOUD}})
    items=list(s.items())
    if not items: return None
    # closest in time to eruption, then lowest cloud
    items.sort(key=lambda it:(abs((it.datetime.date()-erupt).days), it.properties['eo:cloud_cover']))
    it=items[0]
    return dict(id=it.id, date=it.datetime.date().isoformat(),
                cloud=round(it.properties['eo:cloud_cover'],1))

# ── build match table ─────────────────────────────────────────────────────────
rows=[]
for ed, lab, major in ERUPTIONS:
    db=closest_dem(ed,True); da=closest_dem(ed,False)
    sb=closest_s2(ed,True);  sa=closest_s2(ed,False)
    row=dict(eruption=lab, eruption_date=ed.isoformat(), major=major,
        dem_before=db[0].isoformat() if db else "", dem_before_track=db[2] if db else "",
        dem_before_dt=(ed-db[0]).days if db else "",
        dem_after=da[0].isoformat() if da else "", dem_after_track=da[2] if da else "",
        dem_after_dt=(da[0]-ed).days if da else "",
        s2_before=sb['date'] if sb else "", s2_before_cloud=sb['cloud'] if sb else "",
        s2_before_id=sb['id'] if sb else "",
        s2_after=sa['date'] if sa else "", s2_after_cloud=sa['cloud'] if sa else "",
        s2_after_id=sa['id'] if sa else "")
    rows.append(row)
    print(f"\n{lab} ({ed})")
    print(f"  DEM before: {row['dem_before']} ({row['dem_before_track']}, -{row['dem_before_dt']}d)")
    print(f"  DEM after : {row['dem_after']} ({row['dem_after_track']}, +{row['dem_after_dt']}d)")
    print(f"  S2  before: {row['s2_before']} (cloud {row['s2_before_cloud']}%)")
    print(f"  S2  after : {row['s2_after']} (cloud {row['s2_after_cloud']}%)")

with open(OUT/"eruption_matches.csv","w",newline="") as fh:
    w=csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with open(OUT/"eruption_matches.json","w") as fh: json.dump(rows,fh,indent=2,default=str)
print(f"\n-> {OUT}/eruption_matches.csv")
