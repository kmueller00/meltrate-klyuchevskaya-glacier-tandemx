# Decisive test: multi-date composites, matched by elevation (fixed 2026-11-08, before computing any composite)

**Question:** does ash-covered ice change elevation differently from clean ice at the same elevation after each eruption, beyond what the same ash mask shows in years before the eruption?

**Epochs.** Per year, every August–September DEM of track 155_0045. Each DEM reprojected to the fixed grid of FULL_GRID_elevation_mosaic.tif, Nuth & Kääb co-registered to that mosaic on stable terrain (retry seeds; shifts over 60 m rejected), stable-terrain median removed. Scene gate (independent of the glacier signal): at least 3e5 valid pixels, median elevation 800–2500 m, stable-terrain NMAD at most 8 m. Same-date sub-frames are merged by median; the epoch is the per-pixel median over dates (at least 2 dates per pixel where an epoch has 2 or more).

**Stable terrain.** Outside all inventoried glaciers + 300 m, slope 2–40°.

**Statistic.** Saved strict ash masks. D = ash minus clean median dH within 200 m elevation bands (elevation from the fixed mosaic), weighted by ash pixels, at least 30 pixels per class and band. CI: 1.5 km tile bootstrap, 1000 resamples.

**Targets.** 2019 Oct (2019→20), 2020 Dec (2020→21), 2022 Nov (2022→23), 2023 paroxysm year 1 (2023→24) and year 2 (2024→25).

**A target is CONFIRMED only if all four hold:**
1. the 95% CI of D excludes zero;
2. D has the same sign in all five variants: primary, residual-plane removal, only DEM dates passing the trend gate, moderate ash mask, and 100 m elevation bands;
3. leaving out any single DEM date never changes the sign;
4. D lies outside the full range of the placebo values: the same mask and method applied to every consecutive year pair ending before the eruption.

**SUGGESTIVE:** criterion 1 holds and at least two of 2–4. **NOT DETECTED:** otherwise.

**The reversal hypothesis** is confirmed only if 2023 year 1 is CONFIRMED positive and year 2 is CONFIRMED negative.

Reported secondary, not part of the decision: raw D, elevation × aspect matching.
Not tested: 2015 (no saved strict mask), which also has the unexplained coherence problem.
