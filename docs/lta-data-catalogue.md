# LTA data acquisition review - 1 November 2026

Sources reviewed:
- https://datamall.lta.gov.sg/content/datamall/en/static-data.html
- https://datamall.lta.gov.sg/content/datamall/en/dynamic-data.html
- https://datamall.lta.gov.sg/content/dam/datamall/datasets/LTA_DataMall_API_User_Guide.pdf (version 6.11, 30 October 2026)

## Acquisition status

The catalogues were read successfully. Direct downloads of the six selected files returned HTTP 403 from this environment. **No new dataset was acquired.** Machine-readable attempt records are in `data/lta-public/2026-11-01/manifest.json`. The bounded downloader is `scripts/fetch_lta_public.py`: maximum 16 MiB per file, 40 MiB total; ZIP contents are inventoried without extraction. Successful files receive SHA-256 checksums. The small TrafficFlow ZIP is an uninspected catalogue package, not verified traffic observations.

Existing September OSM and clipped signal-aspect snapshots remain in `scenario/singapore/`. They have not been overwritten and do not supply measured demand or signal timings. New acquisitions must not alter the currently frozen cluster experiment.

## Public downloads to obtain

| Dataset | Catalogue size | Use and limitations |
| --- | --- | --- |
| [Road Crossing](https://datamall.lta.gov.sg/content/dam/datamall/datasets/Geospatial/RoadCrossing_Sep2026.zip) | 431 kB | Crossing geometry; does not establish signal phase or arrival counts. |
| [Footpath](https://datamall.lta.gov.sg/content/dam/datamall/datasets/Geospatial/Footpath_Sep2026.zip) | 7.8 MB | Check pedestrian connectivity against OSM. |
| [Detector Loop](https://datamall.lta.gov.sg/content/dam/datamall/datasets/Geospatial/DetectorLoop_Sep2026.zip) | 1.3 MB | Detector geometry; verify IDs/mappings against requested count records. |
| [Traffic Signal Aspect](https://datamall.lta.gov.sg/content/dam/datamall/datasets/Geospatial/TrafficLight_Sep2026.zip) | 1.5 MB | Signal hardware locations; not timing plans. Existing local clip already available. |
| [Traffic-count description](https://datamall.lta.gov.sg/content/dam/datamall/datasets/Geospatial/Description%20for%20Trip%20Count.pdf) | 84 kB | Explains geodatabase packaging and mostly two-hour, 15-minute classified movement counts. |
| [Traffic Count archive](https://datamall.lta.gov.sg/content/dam/datamall/datasets/Geospatial/Traffic_Count_Data.zip) | 518 MB | Highest-value public demand source, but deferred to protect local storage. Inspect contents on cluster before extraction; find sites with matching collection dates. Do not assume all junctions are synchronised. |

If automated retrieval remains blocked, download the first four files through the static catalogue in a normal browser into `data/lta-public/2026-11-01/`. Keep original archives until checksums and compact local extracts are verified. Do not install a full GIS suite locally merely to inspect these files. Verify each layer's CRS before clipping to the NUS study window; a bounding box is not sufficient to validate crossing topology or permitted movements.

## Dynamic catalogue: useful optional sources

An LTA DataMall AccountKey is needed for authenticated API calls; use the catalogue's Request for API Access link. No account was created and no credential was requested or used during this review. Never put the key in source control or chat.

- **Traffic Flow:** quarterly release of hourly average flow for 07:00-09:00 in a representative month. Useful for plausible demand scales where coverage matches, not same-day junction turn calibration.
- **Traffic Speed Bands:** current arterial/expressway speed bands. Useful for contemporaneous congestion checks; a current snapshot cannot validate historical count periods.
- **Passenger Volume by Bus Stops:** aggregate boarding/alighting activity can motivate pedestrian sensitivity ranges near stops. It does not identify which crossing passengers use or the fraction who cross.
- **Bus Routes / Bus Services / Bus Stops:** useful if adding explicit bus demand and stop locations; not required for the current car/pedestrian study.
- **Approved Road Works / Traffic Incidents / Faulty Traffic Lights:** useful context for observations collected concurrently, not automatic historical exclusion records.
- **Estimated Travel Times:** expressway segments; poor match for a short signalised NUS corridor.
- **Traffic Images:** primarily expressways/checkpoints; do not collect image archives for this study without a concrete relevant site and measurement protocol.

No crossing-level pedestrian arrival series was identified in these reviewed catalogues. Retain synthetic pedestrian demand labels unless observations are obtained.

## Still request from LTA

The prepared form requests matched junction loop counts and Traffic Lights Traffic Plans, with detector-to-lane/movement mapping, quality flags, effective dates, actuation rules and pedestrian clearance details if available. Optional aggregate pedestrian counts are explicitly availability-dependent.

## Before claiming calibration

1. Select three connected junctions based on overlapping usable observations, not just proximity to NUS.
2. Check geometry, left-hand movement permissions, pedestrian connectivity and every signal conflict.
3. Preserve dates, units, interval duration, source IDs and missing values. Distinguish vehicles from PCU; do not treat missing counts as zero.
4. Fit demand using calibration periods and assess held-out periods. Short within-session holdouts support only limited validation.
5. Compare simulated turning counts and relevant speeds with observations; report discrepancies and routing uncertainty.
6. Without matched pedestrian arrivals and signal operation, describe the case as real-geometry, partially calibrated simulation, not a validated reconstruction of deployed LTA performance.
