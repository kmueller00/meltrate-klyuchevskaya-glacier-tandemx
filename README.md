# Klyuchevskaya Glacier — Volcanic Eruption Impact on Glacier Elevation (TanDEM-X)

**📄 [Read the full findings report](docs/index.html)** — or browse the source directly: [`prc07_overview_out/RESULTS_presentation/SUMMARY_findings.html`](prc07_overview_out/RESULTS_presentation/SUMMARY_findings.html).

## TL;DR

Do volcanic eruptions at Klyuchevskaya (Kamchatka, 56.0°N) change how fast its glaciers lose mass? Ash/tephra deposited on ice can go either way: thin ash lowers albedo and *accelerates* melt, thick ash *insulates* and slows it (the classic Østrem-curve effect). We tested this with ~300 TanDEM-X bistatic DEMs (2012–2025, ascending track 155) differenced summer-to-summer (so X-band radar's winter snow-penetration bias cancels out), cross-referenced against Sentinel-2/Landsat-derived ash masks for each of the last five eruptions.

**Findings:**
- **The 2023 paroxysm is the largest deposition signal on record**: +2.22 m of retained elevation on ash-covered ice the year after, reversing to net melt (−2.46 m) the year after that — the insulation-then-melt cycle predicted by the tephra critical-thickness mechanism, playing out in real time.
- Smaller eruptions show the same bidirectional pattern at lower magnitude: 2022 Nov +0.58 m, 2020 Dec −1.07 m (likely ash-driven excess melt), 2019 Oct not significant, 2015 Aug–Sep +0.08 m (Landsat-backfilled — Sentinel-2 didn't reliably cover this site yet).
- **Long-term thinning** (2012–2025, summer brackets): −0.38 ± 0.07 m/yr. Counter-intuitively, the *least* thinning is at the low, ash/debris-mantled tongue and the *most* is at mid-elevation clean ice — an inverted hypsometric gradient. A pixel-level, elevation-controlled partial correlation confirms ash cover has a real, independent protective association (r = +0.25) with slower thinning, not just a proxy for elevation.
- **The fastest-thinning "hotspot"** (top 5% of the glacier by thinning rate) sits unusually high (~3,241 m), steep, and SE-facing — tested against two alternative explanations: glacier surge (checked against ITS_LIVE velocity data — the hotspot flows *no faster* than the rest of the glacier, arguing against surge) and geothermal heating (Klyuchevskaya's well-documented Apakhonchich flank-vent system sits on the same SE flank, a real but not yet spatially-confirmed lead).
- A within-season melt-progression curve (spring accumulation → mid-summer turnover → ablation) is now built for the years with dense enough spring/summer coverage.

Full methodology, all figures, per-eruption breakdowns, and known limitations are in the [report](docs/index.html).

## Repository layout

This mirrors the actual working directory used to run the analysis (all scripts use absolute local paths to the source DEM archive and outputs — this is a working research repo, not a portable package; re-running anything requires the original TanDEM-X archive and matching local paths).

- **`ash_analysis/`** — the eruption/ash-deposition pipeline, numbered scripts `01`–`34` run roughly in order: acquisition matching → ash mask generation → per-eruption deposition/melt statistics → hotspot/coldspot/hypsometric spatial analysis → the within-season melt curve and surge-delineation checks. `s2_util.py` is the shared Sentinel-2/Landsat fetch utility.
- **Top-level `.py` scripts** — the broader elevation-change pipeline: full-archive robust rate maps (`dH_robust_all.py`), coregistration diagnostics, yearly/seasonal rate tifs, and presentation-figure generators.
- **`prc07_overview_out/RESULTS_presentation/`** — every figure (PNG), statistics table (CSV), and the flagship `SUMMARY_findings.html` report itself. `ash_analysis/` and `SUMMARY_figures_fullres/` subfolders hold eruption-specific and full-resolution copies respectively.
- **`docs/index.html`** — a copy of the report, present only so GitHub Pages can serve it as a live page.

Raw DEM rasters (`.tif`) are excluded from version control (see `.gitignore`) — they're large, fully regenerable from the scripts against the source archive, and not needed to read or understand the findings.
