# Singapore virtual case: NUS Kent Ridge–NUH pilot

This is a **geometry and data-readiness pilot**, not a claim that the agents improve Singapore traffic lights. A small public map extract and public signal-hardware locations are included so the case can be inspected without a large download. The protected simplified corridor has synthetic controller runs, but there are **no calibrated Singapore-controller results**, real traffic or pedestrian counts, or actual LTA signal timings.

## What is in the workspace

| Item | Source and meaning | Size |
| --- | --- | ---: |
| NUS-area OSM snapshot | [OpenStreetMap bbox API](https://wiki.openstreetmap.org/wiki/Api06#Retrieving_map_data_by_bounding_box:_GET_/api/0.6/map), captured 2026-09-28; roads, tagged footways, signals, and crossings | 424 kB compressed |
| Clipped LTA signal aspects | [data.gov.sg LTA Traffic Signal Aspect](https://data.gov.sg/datasets/d_44cc2bba6131f9057fb92f367eba69ed/view), fetched and clipped to the same box; **hardware points, not phase plans** | 39 kB |
| Config | `config/singapore.json`: bbox, source URLs, hashes, 8 MB OSM cap and 22 MB LTA cap | <2 kB |
| Optional observations | `config/singapore_counts.example.csv` has a header only; there are **no invented counts** | <1 kB |

The full LTA signal-aspect GeoJSON is read in memory and only its local clip is retained. The 518 MB [LTA turning-count archive](https://datamall.lta.gov.sg/content/datamall/en/static-data.html) is deliberately not downloaded. LTA describes its counts as mostly two-hour junction observations in 15-minute intervals in its [traffic-count note](https://datamall.lta.gov.sg/content/dam/datamall/datasets/Geospatial/Description%20for%20Trip%20Count.pdf). Actual [junction phase timings and detector counts are on-request](https://datamall.lta.gov.sg/content/datamall/en/on-request_datasets.html).

## Run the small virtual test

From the repository root, make the bundled SUMO Python tools visible if using the portable installation:

```powershell
$env:PYTHONPATH = (Resolve-Path .local\sumo-1.27.1\tools).Path
python -m src.traffic_control.singapore --smoke
```

This reads the versioned OSM snapshot, runs `netconvert` with `--lefthand true --osm.sidewalks true --osm.crossings true`, checks which signal sites and pedestrian crossings survived import, places seeded **synthetic** cars and walkers on routable edges, and runs 450 seconds of SUMO. The audit, SUMO import warnings, generated routes, trip output and GUI configuration are stored under Git-ignored `runs/singapore-case/`.

To watch the geometry smoke in the SUMO GUI:

```powershell
& .local\sumo-1.27.1\bin\sumo-gui.exe -c runs\singapore-case\geometry-smoke.sumocfg --start --delay 250
```

Zoom to the crossing named in `runs/singapore-case/geometry-smoke.json`. The route file and GUI use the same saved synthetic demand. `--seed 38` is the default; another integer changes departures and route choices. This demonstration uses **SUMO's imported default signal plan**, never the fixed-time, actuated, IPPO, or MAPPO comparison controllers.

The completed smoke with seed 38 scheduled 24 vehicles and 12 walkers; 24 vehicles departed and arrived, all 12 walkers were seen, and 9 entered mapped crossing edges. The imported map has 7 SUMO crossing edges but **0 signal-controlled crossing edges**, while the clipped LTA layer contains 19 pedestrian signal-aspect points. The import emitted 93 warnings, including missing OSM route references and one invalid pedestrian topology. These counts are diagnostics, not research outcomes. The uncorrected import **fails the protected-crossing prerequisite** for any fair controller comparison.

SUMO's automatic `--crossings.guess true` produced 132 crossing edges but only **one** signal-controlled crossing; adding `--tls.guess-signals true --tls.join true` still controlled only one. The imported OSM signal heads reduce from 20 separate signal programs to 8 joined programs, with a large cluster representing one physical junction. They cannot be treated as 20 independent junctions. Candidate-road lanes also need explicit pedestrian restrictions and sidewalks before adding protected crossings. Use a corrected plain-XML/network source and audit every conflict; do not directly edit a compiled `.net.xml`.

## Run a protected, NUS-derived *simplified* corridor

For a virtual safety and controller-interface check while exact OSM crossings are being repaired:

```powershell
python -m src.traffic_control.singapore --derived-baselines
```

This builds a **new three-junction plain-XML network** through approximate centroids of three candidate OSM signal-site groups. In the imported SUMO coordinate system, `C0 = (1415.52, 3757.59)`, `C1 = (1301.21, 3871.48)`, and `C2 = (1175.20, 4021.76)` metres; `config/singapore_derived.json` freezes these values and checks them against the selected candidate in the case audit. The centre-to-centre distances are about **161 m and 196 m**. The connectors are straight segments between these points, with synthetic endpoints extended along the corridor and synthetic perpendicular 110 m side arms. They **do not reproduce the intervening road shapes or actual junction layout**. The candidate's reported road path is 247.4 m while the centroid distances sum to 357.5 m; site identities, grouping, connectivity, and physical road distances need manual verification. One lane per approach, straight vehicle movements, 2 m sidewalks, four protected crossings per junction, and all traffic and walking demand remain project assumptions. It uses left-hand traffic and the project's shared minimum-green, yellow, all-red, pedestrian-clearance and conflict checks. Fixed-time, pedestrian-actuated, and queue-pressure baselines receive identical demand; each produces metrics and a phase replay under `runs/singapore-case/derived/`.

The seed-61 smoke completed 192 scheduled vehicles and recorded 160 pedestrian crossing entries per baseline. It checks movement, the presence of all 12 protected crossing edges, and runtime phase-safety assertions on this **simplified geometry**. Crossing coverage is uneven: `:C0_c3` has only 2 entries with fixed-time control, 0 with pedestrian-actuated control, and 1 with queue-pressure control. Its actuated mean and p95 waits are therefore undefined. The output reports entry counts and lists zero-entry crossings explicitly; sparse results cannot support a crossing-level performance claim. This is not an estimate of real NUS delay or a ranking of controllers. The actual OSM import above remains unsuitable for controller comparison.

The earlier replica-zero models can run on the same corridor, seed, demand, and revised safe phase machine for an **exploratory candidate-centroid and left-hand-drive interface pilot**:

```powershell
python -m src.traffic_control.singapore --derived-baselines --model-dir models\v2 --output runs\singapore-transfer-v3-pilot
```

The runner selects IPPO and MAPPO at the vehicle reward weight and the first positive pedestrian reward weight from `training_manifest.json`. It checks model version, training episodes and junction IDs, then logs model hashes, vehicle time loss, completed and unserved trips, and per-crossing mean/95th-percentile wait beside the paired conventional runs in `derived-baselines.json`. Detailed metrics and graphical phase replays for this invocation are under `runs/singapore-transfer-v3-pilot/derived/<controller>/`. The v2 models were trained under the **superseded clearance rule**, even though the revised runtime replay passed its live safety assertions. This single-seed test is an interface pilot, **not valid final transfer evidence or a claim about real Singapore traffic**.

A completed replica-zero pilot under the revised three-empty-step clearance is saved at `runs/singapore-transfer-v3-pilot/derived-baselines.json` (seed 61, identical demand hash `f8387720da31a838c40f568442471cc614d4a92af9d2e0ba347b5a26916b4494`). All seven controller runs passed the shared live safety check, but the four v2 policies were trained under the older phase rule; their rounded figures below are shown only to make this **exploratory** virtual run inspectable:

| Controller | Time loss (s) | Completed / scheduled | Crossing entries | Crossing with no entries |
| --- | ---: | ---: | ---: | --- |
| Fixed | 40.0 | 192 / 192 | 160 | None |
| Pedestrian-actuated | 42.2 | 192 / 192 | 160 | `:C0_c3` |
| Pressure-inspired | 39.9 | 192 / 192 | 160 | None |
| IPPO, weight 0, replica 0 | 45.3 | 192 / 192 | 160 | None |
| IPPO, weight 3, replica 0 | 45.3 | 192 / 192 | 160 | None |
| MAPPO, weight 0, replica 0 | 43.7 | 192 / 192 | 160 | `:C1_c3` |
| MAPPO, weight 3, replica 0 | 45.3 | 192 / 192 | 160 | None |

All seven had zero unfinished and uninserted vehicles and zero pending observed crossing requests. The actuated p95 for `:C0_c3` is undefined; other crossings may have only one or two entries. These single-run values are **not** a ranking of methods or a Singapore traffic-performance estimate. They show that the learned controller interface and safety machine operate on the new approximate layout; a crossing-level comparison needs enough observed requests at every crossing, repeated held-out seeds and real demand calibration.

The existing older policies can be checked separately with an explicit legacy label:

```powershell
python -m src.traffic_control.singapore --derived-baselines --model-dir models --legacy-models
```

Those runs are tagged `legacy-*` and `EXPLORATORY LEGACY POLICY PILOT`; do not merge their numbers into the final research comparison. The GUI replay command below also works for a learned controller by replacing `fixed` in all three paths with its label.

To see the derived fixed-time run, replay its recorded phases in SUMO GUI:

```powershell
& .local\sumo-1.27.1\bin\sumo-gui.exe --net-file runs\singapore-transfer-v3-pilot\derived\fixed\connected.net.xml --route-files runs\singapore-transfer-v3-pilot\derived\fixed\demand.rou.xml --additional-files runs\singapore-transfer-v3-pilot\derived\fixed\replay\replay.add.xml --begin 0 --end 600 --seed 61 --pedestrian.striping.dawdling 0 --time-to-teleport -1 --start --delay 250
```

Replace `fixed` with `actuated` or `pressure` in all three paths to view the other baseline on the **same seeded demand**. A headless check of the fixed-time replay reproduced all 192 original vehicle trip records exactly.

## Repeated uncalibrated geometry transfer

`config/singapore_transfer.json` freezes ten separate transfer seeds, **9101–9110**, which were not used for training, validation, the exploratory seed-61 Singapore pilot, or the main locked synthetic evaluation. The repeated transfer uses all **30 models**: IPPO and MAPPO at reward weights 0, 3, and 10, each with five independently trained replicas. Fixed-time, pedestrian-actuated, and pressure-inspired controls run on the same demand per seed. The full run is accepted only after `runs/training-v3/extension_decision.json` selects the training budget and the chosen model manifest contains every matching replica. Run it **after the main locked synthetic evaluation completes**, with the model set selected by the recorded decision:

```powershell
python -m src.traffic_control.singapore_transfer --model-dir models/v3 --output results/singapore-transfer
# If the decision selected 300 episodes, wait for all models/v4 replicas instead:
python -m src.traffic_control.singapore_transfer --model-dir models/v4 --output results/singapore-transfer
```

The output includes `REPORT.md`, `summary.json`, 330 per-run records, all 12 crossings' entry denominators, p95 waits and pending requests, paired confidence intervals against pedestrian-actuated control, and compressed crossing events. Each seed's network and routes are generated once and shared across controllers; completed groups can resume only with an identical protocol and model fingerprint. Temporary per-controller SUMO files are removed after extraction to respect the storage limit. The already inspected seed 61 may be used only for an exploratory code smoke check with `--smoke --model-dir models/v2 --output runs/singapore-transfer-smoke-dev`.

This is a **Singapore-derived geometry stress test with synthetic demand**, not a calibrated NUS corridor evaluation. The raw OSM import still has zero protected signal-controlled crossings; the tested safe corridor uses approximate candidate centroids and spacings, straight synthetic links, synthetic side arms and crossings, and the team's phase table. The site identities and actual road connectivity remain unverified. Sparse crossing entry counts or an incomplete p95 bootstrap make the every-crossing target inconclusive even with repeated seeds.

## Turn this into a defensible Singapore evaluation

1. **Choose three distinct connected signal sites.** `case-audit.json` lists candidate groups. The grouping treats signal heads within 45 m as one site so three heads at one large junction cannot masquerade as three junctions. Inspect the candidates in SUMO GUI and against the OSM and LTA layers. The listed IDs and road distances are suggestions, not verified junction identities.
2. **Repair and verify pedestrian infrastructure.** Use LTA's public [Road Crossing, Footpath, and Road Section Line layers](https://datamall.lta.gov.sg/content/datamall/en/static-data.html), the signal-aspect clip, and a site visit or independently permissible map observations to verify crossing location and lane movements. Edit a small SUMO network source, add explicit protected crossings where verified, resolve relevant import warnings, and manually define a conflict-free phase table with minimum green, yellow, all-red, and slow-walker clearance. Apply that *same* table to all six controllers. Hardware points alone do not tell us whether a crossing is signalised or how it is phased.
3. **Collect matched demand without the large archive.** Ask LTA for a small extract or count three junctions in 15-minute intervals. Record pedestrian arrivals by crossing separately. If recording locally, use aggregate counts only; the CSV requires `collection_date,site_id,start_s,end_s,mode,vehicle_class,movement_id,count,source,source_reference`. Use `field_observation` or `lta_traffic_count` as the source only when accurate. Set `vehicle_class` empty for pedestrian counts. Movement IDs should identify a vehicle approach and turn or a pedestrian crossing, then be mapped explicitly to SUMO edge IDs. The intake command accepts `--counts path\to\counts.csv` and reports a hash, totals, site dates, and common observed intervals; it does **not** silently transform counts into routes.
4. **Calibrate, validate, then compare.** SUMO's [routeSampler](https://sumo.dlr.de/docs/Demand/Routes_from_Observation_Points.html) can use turning counts and pedestrian counts to form plausible routes, but counts do not uniquely determine origin–destination trips. Fit demand on some intervals and check simulated counts, queues, travel time and pedestrian arrivals on held-out intervals. Only after the geometry, legal phases, crossing events, and demand validation pass should the evaluated controller configurations be run with identical Singapore demand and the project's per-crossing and unserved-trip metrics. A transfer test (train on the synthetic corridor, evaluate on a **fully audited and calibrated** Singapore network) would be stronger than the current approximate-centroid interface pilot.

If counts or actual signal timings cannot be obtained, report this as a **Singapore-geometry stress test with synthetic demand**. Do not call it a calibrated digital twin or an improvement to deployed traffic lights. The current project's EW/NS/all-walk phase table and fixed network assumptions cannot simply be attached to this OSM import: local turns, multiple signal heads, and verified crossing links require an explicit adapter and safety test first.

If a team member obtains LTA's large count ZIP on a machine with enough storage, inspect its member index without extracting it:

```powershell
python -m src.traffic_control.singapore --count-zip D:\Traffic_Count_Data.zip --count-filter junction_name
```

If it contains a small `.csv`, `.txt`, `.xls`, or `.xlsx` member, extract **one exact member** with `--count-member "folder/member.csv"`; a 20 MB limit and path checks apply. An inspection of the current public archive's ZIP index found a roughly **530.6 MB** archive. Its per-junction spreadsheet attachments sit inside one roughly **529.9 MB compressed geodatabase table member**, so ZIP-member selection cannot retrieve just one site's counts. The small location table identifies candidate junction IDs J980, J981, J984 and J515, but the observation dates vary across sites (including 2017, 2018, 2019, 2022 and 2023); matched same-day observations have not been established. These are archive-structure findings, **not an intake of count values**. Use QGIS/GDAL on a machine with adequate space to export **only the selected junction and observed intervals** to a compact CSV, and inspect date and movement metadata before combining sites. Map turn counts into the example CSV schema, preserve provenance, and then use `--counts`. No local GDB-to-CSV conversion or archive-wide extraction is attempted here.

## Reproducibility and attribution

The compressed OSM file and clipped LTA GeoJSON are snapshots, with source hashes in `config/singapore.json` and fresh output hashes in each `case-audit.json`. To inspect current public data instead, use `--fetch-osm --fetch-lta-signals` (bounded downloads). Public datasets can change, so a refreshed audit is a new case version; do not overwrite the saved snapshots without recording date and hashes.

Map data: © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), ODbL; follow the [OSM Foundation attribution guidance](https://osmfoundation.org/wiki/Licence/Attribution_Guidelines) in figures and demonstrations. LTA signal-aspect data: [Land Transport Authority via data.gov.sg](https://data.gov.sg/datasets/d_44cc2bba6131f9057fb92f367eba69ed/view), Singapore Open Data Licence. The SUMO import flags and limitations follow the official [SUMO OpenStreetMap import documentation](https://sumo.dlr.de/docs/Networks/Import/OpenStreetMap.html).
