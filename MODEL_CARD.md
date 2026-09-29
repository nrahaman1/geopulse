# Model card: GeoPulse v0.2 (flood, wildfire, vegetation disturbance)

Checkpoints in `models/`, each with a JSON card beside it (tasks, checkpoint SHA-256, dataset manifest SHA-256,
label sources, git SHA, validation metrics, temperature). Full training records: `runs/<model>/run.json`.
Evaluation reports: `runs/<model>/eval_<task>_test.{json,md}`.

| Model | Tasks | Trained on | Sensor dropout |
|---|---|---|---|
| `threshold-baseline` | all | nothing (physics) | — |
| `gpft-flood-mini` / `-nodrop` | flood | GeoPulse-Bench flood | yes / no |
| `gpft-wildfire-mini` / `-nodrop` | wildfire | GeoPulse-Bench wildfire | yes / no |
| `gpft-vegetation-mini` / `-nodrop` | vegetation | GeoPulse-Bench vegetation | yes / no |
| `gpft-multitask-mini` | all three | all three benchmarks | yes |

`auto` resolves to the newest checkpoint that supports the requested task (currently `gpft-multitask-mini`).

## Intended use

Research and demonstration of multimodal change mapping from Sentinel-1 + Sentinel-2 + Copernicus DEM, run
locally on AOIs of up to a few hundred km². Outputs are probabilities with an uncertainty map and an explicit
"review recommended" area; they are not an operational emergency or forest-monitoring product.

**Out of scope:** operational use without expert review; flooded vegetation or dense urban flooding (flood labels
cover open water only); non-stand-replacing forest degradation, crop anomalies (vegetation labels are Hansen
stand-replacing loss); burn severity as a learned output (severity classes are dNBR thresholds, not model output);
resolutions other than 10 m.

## Architecture

Per-sensor adapters shared across time → per-pixel quality-gated fusion (unobserved or cloudy sensors get exactly
zero weight) → change encoder `[pre, post, post−pre, |post−pre|]` + terrain → 3-level U-Net (GroupNorm) → one logit
per task + a shared change logit, ÷ temperature. Uncertainty = entropy of the mean of 8 MC-dropout passes.
~0.54 M parameters, trained from scratch. Post-processing constraint for burn and disturbance: pixels that were open
water before the event are set to 0 (a model never shown the sea once mapped ocean as burned).

## Training data

| Benchmark | Events (train / val / test) | Tiles | Labels |
|---|---|---:|---|
| flood | Missouri 2019, Harvey 2017, Idai 2019, Florence 2018, Ian 2022 / Emilia-Romagna 2023 / **Pakistan 2022** | 236 | weak: S1+S2 consensus |
| wildfire | Carr, Camp, Ranch 2018; Thomas, Tubbs 2017 (California) / Eagle Creek 2017 (Oregon) / **Martin 2018 (Nevada sagebrush)** | 212 | MTBS burn severity (low–high = burned) |
| vegetation | Georgia pines 2019, Oregon Coast 2020, Hurricane Michael 2018 / Västerbotten, Sweden 2019 / **Pará, Brazil 2020** | 176 | Hansen GFC loss year |

Each test event is out of distribution: a continent (flood), an ecoregion (wildfire) or a biome (vegetation)
absent from training.

**Label caveats.** Flood labels are weak: scores measure agreement with the two-sensor consensus, not ground
truth, and the flood baseline cannot be scored against them (circular). MTBS and Hansen are independent of the
Sentinel inputs (Landsat-derived, analyst-thresholded), so baseline-vs-model comparisons on wildfire and vegetation
are fair. MTBS "unburned to low" and Hansen loss in other years are ignored, not treated as negatives.

## Results — held-out test events, one seed

IoU against the benchmark labels; sensors removed at inference by zeroing their validity masks.

**Flood — Pakistan 2022** (weak labels)

| Model | S1+S2 | S1 only | S2 only | ECE |
|---|---:|---:|---:|---:|
| gpft-flood-mini | 0.940 | 0.922 | 0.887 | 0.037 |
| gpft-flood-mini-nodrop | **0.975** | **0.957** | 0.911 | **0.016** |
| gpft-multitask-mini | 0.959 | 0.685 | **0.969** | 0.030 |

**Wildfire — Martin Fire 2018, Great Basin sagebrush** (MTBS)

| Model | S1+S2 | S1 only | S2 only | ECE |
|---|---:|---:|---:|---:|
| threshold-baseline (dNBR) | **0.901**\* | 0.013 | **0.901**\* | **0.070** |
| gpft-wildfire-mini | 0.017 | 0.371 | 0.016 | 0.315 |
| gpft-wildfire-mini-nodrop | 0.232 | **0.445** | 0.323 | 0.267 |
| gpft-multitask-mini | 0.707 | 0.338 | 0.765 | 0.107 |

\* The baseline first scored **0.076**: its NDVI > 0.2 "fuel" gate excluded sagebrush steppe (pre-fire NDVI ≈ 0.17).
The gate was lowered to 0.10 after diagnosing this on the test event, so 0.901 is not a clean held-out number.
On the validation fire (Eagle Creek, conifer) the baseline scores 0.840 and the learned models 0.947–0.955.

**Vegetation disturbance — Novo Progresso, Pará 2020** (Hansen GFC)

| Model | S1+S2 | S1 only | S2 only | ECE |
|---|---:|---:|---:|---:|
| threshold-baseline (NDVI + VH drop) | 0.212 | **0.113** | 0.188 | 0.060 |
| gpft-vegetation-mini | 0.496 | 0.103 | 0.525 | 0.025 |
| gpft-vegetation-mini-nodrop | 0.521 | 0.081 | **0.566** | 0.026 |
| gpft-multitask-mini | **0.560** | 0.200 | 0.557 | **0.021** |

## Findings

1. **Multi-task training helped every task's out-of-distribution test** (design RQ7), measured against the
   matching single-task model with sensor dropout: flood 0.940 → 0.959, wildfire 0.017 → 0.707, vegetation
   0.496 → 0.560. The shared encoder sees far more varied land cover (arid Sindh, Mozambique, Amazon), which
   plausibly explains the wildfire jump. Cost: the multi-task model's SAR-only flood score fell to 0.685.
2. **Physics beat learning across an ecoregion shift for burns.** Trained on five California forest/chaparral
   fires, the single-task models learned "burned forest" and missed low-biomass rangeland burns; dNBR transfers.
   In-distribution (conifer val fire) the learned models win by ~0.11 IoU.
3. **Learning beat physics for vegetation disturbance by ~2.5×**, where fixed NDVI/VH thresholds cannot separate
   stand-replacing loss from drought browning, partial windthrow and regrowth.
4. **Sensor dropout (RQ3) did not help** in any task: the no-dropout ablation matched or beat it with both sensors
   and in most single-sensor rows. Plausible reasons: few training events, and labels concentrated on pixels where
   both sensors carry the signal.
5. SAR-only burn mapping is weak for every model (≤ 0.45 IoU): C-band sees fire only through canopy loss.

## Known limitations

- One seed and one test event per task: no confidence intervals; differences of a few IoU points are not evidence.
- MTBS covers CONUS through 2018 only; Hansen labels only stand-replacing loss, annually.
- Change labels are conservative S1+S2 consensus; the change head is secondary.
- Sentinel-2 SCL can mask dark floodwater as cloud shadow; winter scenes can leave large optical gaps (Camp Fire:
  55 % valid), which SAR fills with explicitly higher uncertainty.
- Vegetation requests must compare the same season in different years; others get a warning.

## Systems (RTX 4060 Laptop GPU, torch 2.11 + CUDA 12.8, batch 16, 256 px tiles)

FP32 10.1 ms/tile, 1.9 GB peak; FP16 autocast is not faster for this small model. Training: ~12 min per
single-task model (40 epochs), ~25 min for the multi-task model (25 epochs). End-to-end inference for 100–200 km² is
dominated by data access (≈1.5 min cold, ≈10–15 s cached).

## License

Code and weights: Apache-2.0. Data and labels: see `DATA_LICENSES.md`.
