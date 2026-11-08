# Decisive test v2: post-eruption ash masks (fixed 2026-11-08, before computing any height change with these masks)

**Why v2.** The masks of v1 were built from imagery about 10 months after each eruption. They flag mostly low glacier tongues, repeat across eruptions and call possibly buried ash "clean".

**New masks.** Sentinel-2 L2A, 30 m, B03 and B11, scene cloud cover < 60%, SCL classes 0, 1, 3, 8, 9, 10 excluded, pixels with cos(local solar incidence) < 0.15 excluded. For each glacier pixel: PRE = last valid observation before the eruption; POST = first valid observation after it, within these windows:

| Eruption | PRE window | POST window |
|---|---|---|
| 2019 Oct | 2019-09-25 to 10-24 | 2019-10-28 to 11-30 |
| 2020 Dec | 2020-11-09 to 12-08 | 2020-12-10 to 2021-01-15 |
| 2022 Nov | 2022-10-16 to 11-15 | 2022-11-27 to 12-31 |
| 2023 paroxysm | 2023-10-01 to 10-30 | 2023-11-03 to 12-05 |

- **Strict:** ash = NDSI_PRE ≥ 0.4 and NDSI_POST < 0.4; clean = NDSI_PRE ≥ 0.4 and NDSI_POST ≥ 0.4.
- **Loose (replaces the "moderate" variant):** ash = NDSI_PRE ≥ 0.4 and (NDSI_POST < 0.6 or NDSI drop ≥ 0.2); clean = the remaining pixels with NDSI_PRE ≥ 0.4.
- All other pixels (bare before the eruption, no valid data) are excluded from both classes. One morphological opening of the ash class.

Masks may be revised after visual QC of the imagery, but only before any height change is computed with them.

**Epoch rule added from the v1 diagnostic.** A DEM date enters an epoch only if its largest-footprint sub-frame passes the scene gate. All other epoch settings are as in PROTOCOL.md.

**Statistic, targets, placebo years and the four criteria:** unchanged from PROTOCOL.md (the 2023 mask is used for 2023→24 and 2024→25). In addition, the two-year pair 2021→2023 is reported for the 2022 mask.

**Limit.** Ash that fell after the POST observation (for example later phases of the 2020–21 eruption) is not mapped.
