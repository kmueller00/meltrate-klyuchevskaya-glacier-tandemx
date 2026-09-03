#!/usr/bin/env python3
# Klyuchevskoy eruptions — KVERT / Smithsonian GVP / Wikipedia cross-checked (2026-06).
# Dates verified against GVP weekly reports & Wikipedia "Klyuchevskaya Sopka".
epochs = {
 "Summer EARLY (2013-07-15, 2014-07-02)": (2013.54, 2014.50),
 "Summer LATE  (2019-06-20, 2019-07-01)": (2019.47, 2019.50),
 "Winter EARLY (2012-02-13, 2017-02-12)": (2012.12, 2017.12),
 "Winter LATE  (2024-02-08, 2025-02-16)": (2024.10, 2025.13),
}
# (start, end, label)  -- decimal years
eruptions = [
 (2010.16, 2010.19, "Feb27–Mar9 2010: explosive+effusive, ash to 6 km"),
 (2012.79, 2012.91, "Oct15 & Nov29 2012: weak (magma diverted to Tolbachik)"),
 (2013.07, 2013.93, "2013: Strombolian Jan25 & Aug15; MAJOR Oct–Nov explosions, ash 10–12 km (Nov19)"),
 (2015.00, 2015.23, "Jan2–16 & Mar10–24 2015: Strombolian + minor; Aug27 Strombolian"),
 (2019.82, 2019.83, "Oct25 2019: weak Strombolian (~30 h)"),
 (2020.94, 2020.96, "Dec9 2020: ash explosions to 7 km"),
 (2022.89, 2022.92, "Nov20 2022: eruption onset after regional quake"),
 (2023.47, 2024.05, "Jun22 2023 → early 2024: Strombolian; PAROXYSM Nov1 2023 ash to 13–14 km; lava flows; ash to 6 km Dec30"),
 (2025.58, 2025.62, "Jul30–Aug 2025: summit eruption, W-slope lava, lahars (after quake)"),
]
print("KLYUCHEVSKOY ERUPTIONS (KVERT/GVP-verified) vs DEM EPOCHS")
print("="*64)
for en,(a,b) in epochs.items():
    print(f"\n{en}:")
    hit=False
    for s,e,desc in eruptions:
        if e>=a and s<=b:
            print(f"   OVERLAPS  {desc}"); hit=True
    if not hit: print("   (no documented eruption within window)")
print("\n"+"="*64+"\nFull verified timeline:")
for s,e,desc in eruptions:
    print(f"   {s:.2f}-{e:.2f}: {desc}")
