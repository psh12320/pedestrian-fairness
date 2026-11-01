# CS4246 Team 38 presentation guide

This 13-slide deck is paced for a **15-minute talk plus 5 minutes of Q&A**; adjust the pacing if the course sets a different slot. Build it with `node deliverables/presentation/build_team38_deck.mjs`. The completed, rendered and visually checked deck is `deliverables/presentation/output/team38-research-talk-v2f-final.pptx`. It contains the 4,290-run main study and the separate 330-run uncalibrated transfer case. The older preflight previews are superseded. The builder preserves prior files, so choose a new filename with `TEAM38_PRESENTATION_OUTPUT` if you need another revision. Slide 9 intentionally keeps the earlier exploratory result, which used the stopped-approach crossing-wait definition and five inspected evaluation seeds. Do not combine those waits with the final event-based study.

## Pacing

| Slides | Time | Speaker focus |
|---|---:|---|
| 1–2 | 1.5 min | Question and the strict crossing-level target |
| 3–5 | 3.5 min | Synthetic environment, controllers, common safety rules |
| 6–8 | 4 min | Event measurement, repeated-seed protocol, decision rule |
| 9–10 | 2.5 min | Explain the exploratory failure, then the locked final result |
| 11–12 | 2 min | Singapore pilot limitations and interpretation |
| 13 | 1.5 min | Team work, LLM disclosure, invite questions |

## Before presenting

1. Slide 3 uses final-protocol SUMO stills from junction C1 at simulation time 147 s, with the same demand seed 6001. Fixed time appears beside **IPPO pedestrian weight 3, training replica zero**, chosen as the illustration before inspecting final results. Pedestrian icons are enlarged threefold for visibility; simulation geometry and movement are unchanged. Refresh the files with the commands below after the validation-selected model budget is complete. The old files outside `assets/final-protocol/` remain exploratory and must not be presented as the final demonstration.
2. Slide 10 updates automatically only when `results/final-study/summary.json` and `crossing_comparison.csv` show the complete 30-model, ten-locked-seed, 13-scenario protocol. The delivered deck passed this gate and slide 10 has been rendered and inspected. A later incomplete dataset will produce a conspicuous placeholder instead.
3. The slide 10 illustration is **IPPO with pedestrian weight 3**, fixed in the builder before inspecting final results. It shows paired vehicle and worst-crossing p95 differences with 95% hierarchical bootstrap intervals, the all-crossing decision, and service counts. Its editable chart displays all 12 crossing p95 differences, paired intervals, and entry counts as `actuated / smallest IPPO training replica`. The full nine-controller comparison belongs in `results/final-study/REPORT.md`. State any failed or uncertain criterion directly. A positive crossing difference means longer p95 wait; sparse entry counts make tail estimates less convincing.
4. Keep the NUS-area corridor labeled **estimated-spacing and synthetic-demand**. The candidate signal-group identities and spacings need manual verification. Public LTA signal points are not timing plans. The raw OSM import did not yield protected crossing control, so the safe left-hand corridor uses project assumptions.
5. Slide 12 updates its six-setting pass counts when final data are complete; its first sentence still reports the **earlier exploratory** outcome. Keep that distinction clear when speaking. Slide 13 contains **provisional individual contributions** because the team has not supplied a verified work log; replace every assignment with actual work before submission. Check the exact course disclosure requirement and state specifically how LLM assistance was used, then have the team verify the final slides, plots, and text.

## Demonstration and likely questions

- Once the validation-only training-budget decision and all 30 matching replicas are complete, open both windows with `python -m src.traffic_control.viewer --compare --model-dir models/v3 --seed 6001 --pedestrian-count 120` if the selected budget is 150 episodes. Use `models/v4` if the decision extends all replicas to 300 episodes. The viewer now keys cached source runs by the safety configuration and simulation code, so the fixed-time and learned phases are generated under the same rule. Seed 6001 is exploratory; the final evaluation uses locked seeds 9001–9010.
- The final moving comparison is `deliverables/presentation/assets/final-protocol/sumo-fixed-vs-ippo-pedestrian-seed-6001.mp4`, fixed time on the left and IPPO weight 3 on the right. A 20-second simulation segment plays over 10 seconds. This demonstrates movements, not average performance. Record from a normal interactive desktop: a hidden sandbox desktop can produce one-pixel screenshots.
### Refresh the final-protocol visuals after training

Run these PowerShell commands from the repository root after `models/v3/training_manifest.json` says `complete_protocol: true` and the validation decision is recorded. If it selects 300 episodes, wait for `models/v4/training_manifest.json` to complete and change `$modelDir` to `models/v4`. SUMO GUI, TraCI, and `ffmpeg` must be available. The stills and MP4s are visual replays of measured controller phases; they are not evidence of average performance.

```powershell
$modelDir = 'models/v3'
$assetDir = 'deliverables/presentation/assets/final-protocol'
python -m src.traffic_control.capture_gui --controllers fixed ippo-pedestrian --model-dir $modelDir --output $assetDir --seed 6001 --second 147 --boundary 240 70 320 150
python -m src.traffic_control.capture_clip --controllers fixed ippo-pedestrian --model-dir $modelDir --output $assetDir --seed 6001 --start-second 138 --frames 20 --fps 2 --isolated-frames
ffmpeg -y -i "$assetDir/sumo-fixed-seed-6001-t138.mp4" -i "$assetDir/sumo-ippo-pedestrian-seed-6001-t138.mp4" -filter_complex 'hstack=inputs=2,scale=1920:-2' -c:v libx264 -pix_fmt yuv420p -crf 22 "$assetDir/sumo-fixed-vs-ippo-pedestrian-seed-6001.mp4"
```

The slide builder already points to these final-protocol image filenames. Rebuild it after refreshing the captures. The screenshots share the same demand hash, but are from one illustrative seed and one replica; present the 10-seed, five-replica results separately.
- **How is crossing wait measured?** Arrival is first observed near the crossing endpoint or at a stopped approach queue; entry is first step on the crossing edge. The one-second samples and direct-entry flags are in the raw events. Downsampling one shared 0.1-second trajectory gave 0.49 s mean absolute error and 1.30 s maximum error over 30 entered events; two pending requests remained in both trackers. This is one development fixture, not a universal bound. SUMO's `waitingTime` alone is not the measure.
- **Can a fast controller hide vehicles?** Report scheduled, completed, unfinished, and uninserted trips beside completed-trip time loss.
- **Why not claim field validation?** The Singapore pilot lacks observed turning counts, pedestrian arrivals, and actual timing plans; only signal spacing comes from local map geometry.
- **Why compare MAPPO with IPPO?** MAPPO permits a network-wide value estimator in training while actors still see local junction observations. The short-spacing case probes whether that difference changes choices under coupling.
- **Why did the final seeds change?** An 80-person burst exposed a pedestrian-on-crossing safety failure in the initial 7001–7010 run. The common clearance rule was strengthened and the burst reduced to 20 people; all 330 development controller runs on those exposed seeds passed. A partial 8001–8010 evaluation was then stopped when we noticed its policies had been trained under the earlier clearance. All replicas were retrained, and the untouched 9001–9010 seeds are the final test. Keep both older runs out of performance comparisons; see `docs/safety_revision_v3.md`.

The final logs contain 28 pedestrian-jam warning lines in the short-spacing study and two in the separate transfer case. SUMO jam recovery can ignore obstacles. A saved replay verifies red vehicle links throughout the observed crossing occupancy, but signal checks do not establish collision-free trajectories; pre-entry wait excludes crossing traversal jams. The warning and position audits are in `results/reproducibility/`. Explain this limitation in Q&A.

Speaker notes inside the PowerPoint file give slide-specific caveats and source links.
