# Klyuchevskaya Glacier — Volcanic Eruption Impact on Glacier Elevation (TanDEM-X)

**📄 [Read the full findings report, live](https://kmueller00.github.io/meltrate-klyuchevskaya-glacier-tandemx/)** — or browse the source directly: [`prc07_overview_out/RESULTS_presentation/SUMMARY_findings.html`](prc07_overview_out/RESULTS_presentation/SUMMARY_findings.html).

## TL;DR

Do volcanic eruptions at Klyuchevskaya (Kamchatka, 56.0°N) change how fast its glaciers lose elevation? Thin ash lowers albedo and can speed up melt, while thick ash insulates and slows it (the Østrem curve). We tested this with 308 usable TanDEM-X bistatic DEMs (2012–2025, ascending track 155) and Sentinel-2/Landsat ash masks.

**Correction, November 2026.** Earlier versions of this repo reported +2.22 m of retained elevation on ash-covered ice after the 2023 paroxysm, reversing to −2.46 m a year later. Those numbers reproduce, but they come from single DEM dates and compare low ash-covered ice with high clean ice. They do not survive a stricter test, specified in advance (`code/decisive_test/PROTOCOL.md`), using multi-date composites, comparison at equal elevation and pre-eruption control years.

| Target | Earlier report | Late-summer mask [95% CI] | Post-eruption mask [95% CI] | Outcome |
|---|---|---|---|---|
| 2019 Oct | −0.04 m | −0.21 [−1.57, +0.40] | not testable (0.0 km²) | Not detected |
| 2020 Dec | −1.07 m | −0.10 [−0.19, +0.14] | not testable (0.2 km²) | Not detected |
| 2022 Nov | +0.58 m | −1.51 [−2.15, −0.89] | +1.54 [−1.26, +2.85]; 0.9 km² | Late-summer: met the rules, judged an artefact of a faulty 2022 epoch. Post-eruption: not detected |
| 2022 Nov, skipping 2022 | | +0.08 [−0.20, +0.37] | +0.73 [−0.65, +1.74] | No signal |
| 2023 paroxysm, year 1 | +2.22 m | +0.13 [−0.28, +0.71] | −0.05 [−0.29, +0.21]; 25 km² | Not detected |
| 2023 paroxysm, year 2 | −2.46 m | −0.41 [−1.50, +0.23] | +0.10 [−0.73, +1.04]; 25 km² | Not detected |

**Ash-mask limitation.** The late-summer masks come from imagery about 10 months after each eruption, flag mainly low glacier tongues and can label buried ash as clean ice. The post-eruption masks (snow by NDSI before the eruption, no snow in the first clear image after it) map 2023 well, on the upper flanks, but almost nothing for the other eruptions, which remain poorly tested.

Smaller effects cannot be excluded, and the 2015 eruption was not part of this test.

**Other findings:**
- **Long-term thinning** (2012–2025): −0.36 ± 0.06 m/yr in summer, consistent with winter (−0.34 ± 0.17 m/yr).
- **Elevation pattern:** band-averaged thinning increases from the low, ash-mantled tongue (−0.17 m/yr at 1400 m) to about 3200 m (−0.95 m/yr). At pixel scale, ash cover does not predict thinning at equal elevation (partial r = +0.01).
- **The fastest-thinning 5% of pixels** were tested against glacier surge (ITS_LIVE: no faster than elsewhere) and geothermal heat (no overlap with the mapped flank vents; one small marginal exception). Neither explains the pattern systematically.
- **Ash mask limits:** about a fifth of the mapped "ash" is flagged in at least 3 of 4 eruptions, mostly low, smooth terrain, so the mask mixes tephra with early snow loss.

Full methodology, all figures, per-eruption breakdowns, and known limitations are in the [report](docs/index.html).

## Repository layout

This mirrors the actual working directory used to run the analysis (all scripts use absolute local paths to the source DEM archive and outputs — this is a working research repo, not a portable package; re-running anything requires the original TanDEM-X archive and matching local paths).

- **`code/`** — all analysis code.
  - **`code/ash_analysis/`** — the eruption/ash-deposition pipeline, numbered scripts `01`–`39` run roughly in order: acquisition matching → ash mask generation → per-eruption deposition/melt statistics → hotspot/coldspot/hypsometric spatial analysis → the within-season melt curve and surge-delineation checks. `s2_util.py` is the shared Sentinel-2/Landsat fetch utility.
    - **`code/decisive_test/`** — the pre-specified robustness test of November 2026: protocol, scripts, with results in `RESULTS_presentation/decisive_test/`.
- **`code/*.py`** — the broader elevation-change pipeline: full-archive robust rate maps (`dH_robust_all.py`), coregistration diagnostics, yearly/seasonal rate tifs, and presentation-figure generators.
- **`prc07_overview_out/RESULTS_presentation/`** — every figure (PNG), statistics table (CSV), and the flagship `SUMMARY_findings.html` report itself. `ash_analysis/` and `SUMMARY_figures_fullres/` subfolders hold eruption-specific and full-resolution copies respectively. This stays at the repo root rather than under `code/` — it's this project's *output*, not its code, and it's also the exact path every script writes to, so keeping it separate from `code/` avoids the two ever being confused.
- **`docs/index.html`** — a copy of the report, present only so GitHub Pages can serve it as a live page.

Raw DEM rasters (`.tif`) are excluded from version control (see `.gitignore`) — they're large, fully regenerable from the scripts against the source archive, and not needed to read or understand the findings.
