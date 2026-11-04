# Pedestrian-focused MAPPO fine-tuning

The starting policies are vehicle-efficient MAPPO-control checkpoints trained with pedestrian reward weight 3. This is a full model warm start, including actor, critic and value-normalisation state; these are not vehicle-only-reward checkpoints.

All planned starting replicas and final-budget checkpoints are included. This experiment supplements the original proposal comparison. It does not replace earlier failures or use a best-run headline.

Vehicle loss is completed-vehicle SUMO time loss. Count tables show means per simulation, not pooled totals. Vehicle service counts and insertion delay remain visible. Crossing tails pool demand seeds within each training replica and then average replica tails. Intervals are exploratory paired bootstrap intervals over whole replicas/demand seeds. Per-crossing intervals are pointwise, not simultaneous guarantees.

A result such as vehicle improvement with 6/12 crossing p95s no higher is a partial trade-off result. The original target still requires all 12 crossing p95s no higher and no strict service deterioration. Sparse estimates (<100 entries in any pooled replica/reference crossing) are inconclusive.

## vmedium-pmedium

| Controller | Vehicle loss s | Insertion delay s | Scheduled | Completed | Unfinished | Uninserted | Pending cutoff | Unresolved followup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed | 33.14 | 0.33 | 963.75 | 933.50 | 30.00 | 0.25 | 13.25 | 0.00 |
| actuated | 50.59 | 0.32 | 963.75 | 929.00 | 34.50 | 0.25 | 6.75 | 0.00 |
| pressure | 39.93 | 0.33 | 963.75 | 935.75 | 27.75 | 0.25 | 7.75 | 0.00 |
| mappo-original | 40.99 | 0.34 | 963.75 | 935.25 | 28.25 | 0.25 | 6.67 | 0.00 |
| mappo-continue | 38.10 | 0.33 | 963.75 | 936.25 | 27.25 | 0.25 | 7.50 | 0.00 |
| mappo-tail | 41.59 | 0.34 | 963.75 | 932.75 | 30.75 | 0.25 | 6.33 | 0.00 |
| mappo-guided | 46.59 | 0.34 | 963.75 | 931.92 | 31.50 | 0.33 | 5.00 | 0.00 |

All-horizon vehicle service below includes warmup and is a different cohort from the primary measurements above.

| Controller | Horizon scheduled | Horizon completed | Horizon unfinished | Horizon uninserted |
|---|---:|---:|---:|---:|
| fixed | 1115.75 | 1115.75 | 0.00 | 0.00 |
| actuated | 1115.75 | 1115.75 | 0.00 | 0.00 |
| pressure | 1115.75 | 1115.75 | 0.00 | 0.00 |
| mappo-original | 1115.75 | 1115.75 | 0.00 | 0.00 |
| mappo-continue | 1115.75 | 1115.67 | 0.08 | 0.00 |
| mappo-tail | 1115.75 | 1115.75 | 0.00 | 0.00 |
| mappo-guided | 1115.75 | 1115.67 | 0.08 | 0.00 |

| Controller | Persons scheduled | Persons completed | Persons unfinished | Persons uninserted | Entered crossing without observed exit (all horizon) |
|---|---:|---:|---:|---:|---:|
| fixed | 681.25 | 681.25 | 0.00 | 0.00 | 0.00 |
| actuated | 681.25 | 681.25 | 0.00 | 0.00 | 0.00 |
| pressure | 681.25 | 681.25 | 0.00 | 0.00 | 0.00 |
| mappo-original | 681.25 | 681.25 | 0.00 | 0.00 | 0.00 |
| mappo-continue | 681.25 | 681.25 | 0.00 | 0.00 | 0.00 |
| mappo-tail | 681.25 | 681.25 | 0.00 | 0.00 | 0.00 |
| mappo-guided | 681.25 | 681.25 | 0.00 | 0.00 | 0.00 |

| Policy / reference | Vehicle delta s (exploratory 95% interval) | Crossings no worse | Worst crossing delta s | Strict service failures | Interpretation |
|---|---|---:|---:|---:|---|
| mappo-original / actuated | -9.60 ([-11.72, -7.82]) | 0/12 | 18.13 | 12/12 | vehicle gain with all 12 crossing p95 point estimates worse; original target unmet |
| mappo-original / mappo-original | 0.00 ([0.00, 0.00]) | 12/12 | 0.00 | 0/12 | no vehicle time-loss improvement; joint comparison gate unmet; original target is assessed against actuated only |
| mappo-continue / actuated | -12.50 ([-13.99, -11.01]) | 0/12 | 16.60 | 12/12 | vehicle gain with all 12 crossing p95 point estimates worse; original target unmet |
| mappo-continue / mappo-original | -2.89 ([-4.90, -0.94]) | 4/12 | 5.00 | 12/12 | vehicle gain with 4/12 crossing p95 point estimates no higher; joint comparison gate unmet; strict service deterioration also present; original target is assessed against actuated only |
| mappo-tail / actuated | -9.00 ([-10.63, -7.56]) | 0/12 | 14.53 | 9/12 | vehicle gain with all 12 crossing p95 point estimates worse; original target unmet |
| mappo-tail / mappo-original | 0.60 ([-1.06, 2.25]) | 10/12 | 1.47 | 11/12 | no vehicle time-loss improvement; joint comparison gate unmet; original target is assessed against actuated only |
| mappo-guided / actuated | -4.00 ([-5.36, -3.16]) | 0/12 | 10.67 | 10/12 | vehicle gain with all 12 crossing p95 point estimates worse; original target unmet |
| mappo-guided / mappo-original | 5.60 ([3.87, 7.68]) | 12/12 | -3.93 | 9/12 | no vehicle time-loss improvement; joint comparison gate unmet; original target is assessed against actuated only |

### mappo-original: all crossings versus actuated

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | 18.13 | [15.00, 21.80] | [189, 189, 189] | [189, 189, 189] | 0.00/0.00 |
| :C0_c1 | 16.47 | [13.20, 20.67] | [376, 376, 380] | [373, 373, 373] | 0.00/0.00 |
| :C0_c2 | 9.40 | [0.87, 15.20] | [177, 178, 178] | [176, 176, 176] | 0.00/0.00 |
| :C0_c3 | 16.33 | [15.33, 19.13] | [191, 191, 191] | [191, 191, 191] | 0.00/0.00 |
| :C1_c0 | 10.20 | [2.80, 17.40] | [190, 190, 191] | [192, 192, 192] | 0.00/0.00 |
| :C1_c1 | 12.67 | [9.47, 15.40] | [416, 418, 401] | [407, 407, 407] | 0.00/0.00 |
| :C1_c2 | 4.60 | [-0.73, 12.47] | [223, 221, 223] | [223, 223, 223] | 0.00/0.00 |
| :C1_c3 | 9.27 | [3.20, 14.53] | [365, 371, 370] | [373, 373, 373] | 0.00/0.00 |
| :C2_c0 | 12.60 | [2.53, 15.73] | [195, 197, 196] | [194, 194, 194] | 0.00/0.00 |
| :C2_c1 | 16.00 | [10.87, 26.00] | [179, 179, 179] | [179, 179, 179] | 0.00/0.00 |
| :C2_c2 | 13.33 | [9.27, 17.13] | [210, 210, 210] | [210, 210, 210] | 0.00/0.00 |
| :C2_c3 | 16.33 | [14.00, 17.93] | [380, 377, 390] | [387, 387, 387] | 0.00/0.00 |

### mappo-original: all crossings versus mappo-original

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | 0.00 | [0.00, 0.00] | [189, 189, 189] | [189, 189, 189] | 0.00/0.00 |
| :C0_c1 | 0.00 | [0.00, 0.00] | [376, 376, 380] | [376, 376, 380] | 0.00/0.00 |
| :C0_c2 | 0.00 | [0.00, 0.00] | [177, 178, 178] | [177, 178, 178] | 0.00/0.00 |
| :C0_c3 | 0.00 | [0.00, 0.00] | [191, 191, 191] | [191, 191, 191] | 0.00/0.00 |
| :C1_c0 | 0.00 | [0.00, 0.00] | [190, 190, 191] | [190, 190, 191] | 0.00/0.00 |
| :C1_c1 | 0.00 | [0.00, 0.00] | [416, 418, 401] | [416, 418, 401] | 0.00/0.00 |
| :C1_c2 | 0.00 | [0.00, 0.00] | [223, 221, 223] | [223, 221, 223] | 0.00/0.00 |
| :C1_c3 | 0.00 | [0.00, 0.00] | [365, 371, 370] | [365, 371, 370] | 0.00/0.00 |
| :C2_c0 | 0.00 | [0.00, 0.00] | [195, 197, 196] | [195, 197, 196] | 0.00/0.00 |
| :C2_c1 | 0.00 | [0.00, 0.00] | [179, 179, 179] | [179, 179, 179] | 0.00/0.00 |
| :C2_c2 | 0.00 | [0.00, 0.00] | [210, 210, 210] | [210, 210, 210] | 0.00/0.00 |
| :C2_c3 | 0.00 | [0.00, 0.00] | [380, 377, 390] | [380, 377, 390] | 0.00/0.00 |

### mappo-continue: all crossings versus actuated

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | 16.60 | [13.07, 18.67] | [189, 189, 189] | [189, 189, 189] | 0.00/0.00 |
| :C0_c1 | 14.40 | [12.53, 16.80] | [380, 357, 375] | [373, 373, 373] | 0.00/0.00 |
| :C0_c2 | 14.40 | [7.73, 17.20] | [179, 180, 181] | [176, 176, 176] | 0.00/0.00 |
| :C0_c3 | 13.47 | [10.80, 16.93] | [191, 191, 191] | [191, 191, 191] | 0.00/0.00 |
| :C1_c0 | 14.27 | [3.40, 16.40] | [191, 191, 192] | [192, 192, 192] | 0.00/0.00 |
| :C1_c1 | 14.13 | [11.40, 15.53] | [408, 409, 428] | [407, 407, 407] | 0.00/0.00 |
| :C1_c2 | 7.87 | [1.00, 13.60] | [221, 223, 223] | [223, 223, 223] | 0.00/0.00 |
| :C1_c3 | 12.93 | [9.73, 14.80] | [366, 389, 371] | [373, 373, 373] | 0.00/0.00 |
| :C2_c0 | 14.00 | [9.07, 16.60] | [194, 195, 194] | [194, 194, 194] | 0.00/0.00 |
| :C2_c1 | 16.33 | [12.47, 23.80] | [179, 179, 179] | [179, 179, 179] | 0.00/0.00 |
| :C2_c2 | 14.07 | [12.07, 16.40] | [210, 210, 210] | [210, 210, 210] | 0.00/0.00 |
| :C2_c3 | 14.87 | [13.00, 17.00] | [385, 387, 368] | [387, 387, 387] | 0.00/0.00 |

### mappo-continue: all crossings versus mappo-original

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | -1.53 | [-6.67, 1.47] | [189, 189, 189] | [189, 189, 189] | 0.00/0.00 |
| :C0_c1 | -2.07 | [-6.80, 1.40] | [380, 357, 375] | [376, 376, 380] | 0.00/0.00 |
| :C0_c2 | 5.00 | [-1.87, 13.27] | [179, 180, 181] | [177, 178, 178] | 0.00/0.00 |
| :C0_c3 | -2.87 | [-6.73, -0.40] | [191, 191, 191] | [191, 191, 191] | 0.00/0.00 |
| :C1_c0 | 4.07 | [-6.93, 9.87] | [191, 191, 192] | [190, 190, 191] | 0.00/0.00 |
| :C1_c1 | 1.47 | [-3.27, 4.20] | [408, 409, 428] | [416, 418, 401] | 0.00/0.00 |
| :C1_c2 | 3.27 | [-4.93, 11.00] | [221, 223, 223] | [223, 221, 223] | 0.00/0.00 |
| :C1_c3 | 3.67 | [-2.07, 8.53] | [366, 389, 371] | [365, 371, 370] | 0.00/0.00 |
| :C2_c0 | 1.40 | [-1.67, 10.40] | [194, 195, 194] | [195, 197, 196] | 0.00/0.00 |
| :C2_c1 | 0.33 | [-6.20, 5.13] | [179, 179, 179] | [179, 179, 179] | 0.00/0.00 |
| :C2_c2 | 0.73 | [-3.00, 4.27] | [210, 210, 210] | [210, 210, 210] | 0.00/0.00 |
| :C2_c3 | -1.47 | [-3.73, 0.73] | [385, 387, 368] | [380, 377, 390] | 0.00/0.00 |

### mappo-tail: all crossings versus actuated

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | 14.13 | [8.20, 17.80] | [189, 189, 189] | [189, 189, 189] | 0.00/0.00 |
| :C0_c1 | 14.53 | [12.27, 16.87] | [383, 363, 376] | [373, 373, 373] | 0.00/0.00 |
| :C0_c2 | 9.67 | [0.73, 13.07] | [176, 179, 176] | [176, 176, 176] | 0.00/0.00 |
| :C0_c3 | 13.40 | [9.13, 16.93] | [191, 191, 191] | [191, 191, 191] | 0.00/0.00 |
| :C1_c0 | 5.07 | [0.47, 12.67] | [189, 191, 192] | [192, 192, 192] | 0.00/0.00 |
| :C1_c1 | 11.47 | [7.67, 14.33] | [419, 417, 418] | [407, 407, 407] | 0.00/0.00 |
| :C1_c2 | 4.60 | [0.20, 12.87] | [224, 225, 224] | [223, 223, 223] | 0.00/0.00 |
| :C1_c3 | 10.73 | [6.53, 14.07] | [365, 384, 370] | [373, 373, 373] | 0.00/0.00 |
| :C2_c0 | 7.00 | [-2.93, 13.13] | [195, 195, 197] | [194, 194, 194] | 0.00/0.00 |
| :C2_c1 | 14.53 | [9.20, 17.80] | [179, 179, 179] | [179, 179, 179] | 0.00/0.00 |
| :C2_c2 | 12.53 | [8.07, 15.73] | [210, 210, 210] | [210, 210, 210] | 0.00/0.00 |
| :C2_c3 | 11.13 | [6.67, 15.33] | [376, 384, 373] | [387, 387, 387] | 0.00/0.00 |

### mappo-tail: all crossings versus mappo-original

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | -4.00 | [-9.13, -1.20] | [189, 189, 189] | [189, 189, 189] | 0.00/0.00 |
| :C0_c1 | -1.93 | [-5.60, 1.00] | [383, 363, 376] | [376, 376, 380] | 0.00/0.00 |
| :C0_c2 | 0.27 | [-10.00, 6.53] | [176, 179, 176] | [177, 178, 178] | 0.00/0.00 |
| :C0_c3 | -2.93 | [-7.40, 0.00] | [191, 191, 191] | [191, 191, 191] | 0.00/0.00 |
| :C1_c0 | -5.13 | [-15.33, 6.47] | [189, 191, 192] | [190, 190, 191] | 0.00/0.00 |
| :C1_c1 | -1.20 | [-6.80, 1.93] | [419, 417, 418] | [416, 418, 401] | 0.00/0.00 |
| :C1_c2 | 0.00 | [-8.33, 10.40] | [224, 225, 224] | [223, 221, 223] | 0.00/0.00 |
| :C1_c3 | 1.47 | [-4.20, 6.60] | [365, 384, 370] | [365, 371, 370] | 0.00/0.00 |
| :C2_c0 | -5.60 | [-13.20, 4.73] | [195, 195, 197] | [195, 197, 196] | 0.00/0.00 |
| :C2_c1 | -1.47 | [-13.33, 4.60] | [179, 179, 179] | [179, 179, 179] | 0.00/0.00 |
| :C2_c2 | -0.80 | [-5.73, 4.07] | [210, 210, 210] | [210, 210, 210] | 0.00/0.00 |
| :C2_c3 | -5.20 | [-9.13, -1.60] | [376, 384, 373] | [380, 377, 390] | 0.00/0.00 |

### mappo-guided: all crossings versus actuated

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | 9.60 | [6.13, 13.47] | [189, 189, 189] | [189, 189, 189] | 0.00/0.00 |
| :C0_c1 | 8.80 | [3.73, 13.00] | [378, 387, 384] | [373, 373, 373] | 0.00/0.00 |
| :C0_c2 | 1.80 | [-1.80, 10.33] | [179, 179, 175] | [176, 176, 176] | 0.00/0.00 |
| :C0_c3 | 10.67 | [6.20, 13.87] | [191, 191, 191] | [191, 191, 191] | 0.00/0.00 |
| :C1_c0 | 1.93 | [0.07, 10.80] | [191, 189, 190] | [192, 192, 192] | 0.00/0.00 |
| :C1_c1 | 3.93 | [0.87, 8.67] | [420, 431, 418] | [407, 407, 407] | 0.00/0.00 |
| :C1_c2 | 0.67 | [-2.40, 5.80] | [225, 221, 223] | [223, 223, 223] | 0.00/0.00 |
| :C1_c3 | 5.20 | [1.20, 10.33] | [370, 365, 364] | [373, 373, 373] | 0.00/0.00 |
| :C2_c0 | 7.20 | [0.20, 13.80] | [192, 193, 195] | [194, 194, 194] | 0.00/0.00 |
| :C2_c1 | 10.60 | [3.00, 13.80] | [179, 179, 179] | [179, 179, 179] | 0.00/0.00 |
| :C2_c2 | 7.27 | [2.40, 12.53] | [210, 210, 210] | [210, 210, 210] | 0.00/0.00 |
| :C2_c3 | 5.93 | [1.27, 11.47] | [373, 368, 377] | [387, 387, 387] | 0.00/0.00 |

### mappo-guided: all crossings versus mappo-original

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | -8.53 | [-12.80, -4.13] | [189, 189, 189] | [189, 189, 189] | 0.00/0.00 |
| :C0_c1 | -7.67 | [-14.00, -2.00] | [378, 387, 384] | [376, 376, 380] | 0.00/0.00 |
| :C0_c2 | -7.60 | [-13.27, 1.00] | [179, 179, 175] | [177, 178, 178] | 0.00/0.00 |
| :C0_c3 | -5.67 | [-9.93, -2.87] | [191, 191, 191] | [191, 191, 191] | 0.00/0.00 |
| :C1_c0 | -8.27 | [-14.40, 0.13] | [191, 189, 190] | [190, 190, 191] | 0.00/0.00 |
| :C1_c1 | -8.73 | [-12.53, -4.40] | [420, 431, 418] | [416, 418, 401] | 0.00/0.00 |
| :C1_c2 | -3.93 | [-13.00, 2.00] | [225, 221, 223] | [223, 221, 223] | 0.00/0.00 |
| :C1_c3 | -4.07 | [-9.40, 2.20] | [370, 365, 364] | [365, 371, 370] | 0.00/0.00 |
| :C2_c0 | -5.40 | [-14.40, 7.20] | [192, 193, 195] | [195, 197, 196] | 0.00/0.00 |
| :C2_c1 | -5.40 | [-19.27, 1.53] | [179, 179, 179] | [179, 179, 179] | 0.00/0.00 |
| :C2_c2 | -6.07 | [-10.27, 0.00] | [210, 210, 210] | [210, 210, 210] | 0.00/0.00 |
| :C2_c3 | -10.40 | [-16.07, -4.00] | [373, 368, 377] | [380, 377, 390] | 0.00/0.00 |

The full JSON also retains person-trip service, slower-walker groups where recorded, zero-entry runs, cutoff-only failures, whole-horizon vehicle counts, and entered-but-not-exited diagnostics where available. Missing diagnostics remain unavailable rather than zero.

## pedestrian-burst

| Controller | Vehicle loss s | Insertion delay s | Scheduled | Completed | Unfinished | Uninserted | Pending cutoff | Unresolved followup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed | 33.14 | 0.33 | 963.75 | 933.50 | 30.00 | 0.25 | 13.25 | 0.00 |
| actuated | 50.59 | 0.32 | 963.75 | 929.00 | 34.50 | 0.25 | 6.75 | 0.00 |
| pressure | 40.79 | 0.33 | 963.75 | 933.00 | 30.50 | 0.25 | 7.25 | 0.00 |
| mappo-original | 40.92 | 0.34 | 963.75 | 935.42 | 28.08 | 0.25 | 6.50 | 0.00 |
| mappo-continue | 38.30 | 0.33 | 963.75 | 937.33 | 26.17 | 0.25 | 7.25 | 0.00 |
| mappo-tail | 41.74 | 0.34 | 963.75 | 934.33 | 29.17 | 0.25 | 6.17 | 0.00 |
| mappo-guided | 46.62 | 0.34 | 963.75 | 933.75 | 29.67 | 0.33 | 4.67 | 0.00 |

All-horizon vehicle service below includes warmup and is a different cohort from the primary measurements above.

| Controller | Horizon scheduled | Horizon completed | Horizon unfinished | Horizon uninserted |
|---|---:|---:|---:|---:|
| fixed | 1115.75 | 1115.75 | 0.00 | 0.00 |
| actuated | 1115.75 | 1115.75 | 0.00 | 0.00 |
| pressure | 1115.75 | 1115.75 | 0.00 | 0.00 |
| mappo-original | 1115.75 | 1115.75 | 0.00 | 0.00 |
| mappo-continue | 1115.75 | 1115.67 | 0.08 | 0.00 |
| mappo-tail | 1115.75 | 1115.75 | 0.00 | 0.00 |
| mappo-guided | 1115.75 | 1115.75 | 0.00 | 0.00 |

| Controller | Persons scheduled | Persons completed | Persons unfinished | Persons uninserted | Entered crossing without observed exit (all horizon) |
|---|---:|---:|---:|---:|---:|
| fixed | 701.25 | 701.25 | 0.00 | 0.00 | 0.00 |
| actuated | 701.25 | 701.25 | 0.00 | 0.00 | 0.00 |
| pressure | 701.25 | 701.25 | 0.00 | 0.00 | 0.00 |
| mappo-original | 701.25 | 701.25 | 0.00 | 0.00 | 0.00 |
| mappo-continue | 701.25 | 701.25 | 0.00 | 0.00 | 0.00 |
| mappo-tail | 701.25 | 701.25 | 0.00 | 0.00 | 0.00 |
| mappo-guided | 701.25 | 701.25 | 0.00 | 0.00 | 0.00 |

| Policy / reference | Vehicle delta s (exploratory 95% interval) | Crossings no worse | Worst crossing delta s | Strict service failures | Interpretation |
|---|---|---:|---:|---:|---|
| mappo-original / actuated | -9.67 ([-11.73, -7.94]) | 0/12 | 16.60 | 12/12 | vehicle gain with all 12 crossing p95 point estimates worse; original target unmet |
| mappo-original / mappo-original | 0.00 ([0.00, 0.00]) | 12/12 | 0.00 | 0/12 | no vehicle time-loss improvement; joint comparison gate unmet; original target is assessed against actuated only |
| mappo-continue / actuated | -12.29 ([-13.99, -10.48]) | 0/12 | 17.87 | 11/12 | vehicle gain with all 12 crossing p95 point estimates worse; original target unmet |
| mappo-continue / mappo-original | -2.62 ([-4.95, -0.60]) | 5/12 | 5.27 | 12/12 | vehicle gain with 5/12 crossing p95 point estimates no higher; joint comparison gate unmet; strict service deterioration also present; original target is assessed against actuated only |
| mappo-tail / actuated | -8.85 ([-10.43, -7.62]) | 0/12 | 14.40 | 9/12 | vehicle gain with all 12 crossing p95 point estimates worse; original target unmet |
| mappo-tail / mappo-original | 0.82 ([-0.65, 2.54]) | 11/12 | 0.27 | 11/12 | no vehicle time-loss improvement; joint comparison gate unmet; original target is assessed against actuated only |
| mappo-guided / actuated | -3.97 ([-5.29, -2.85]) | 0/12 | 10.80 | 9/12 | vehicle gain with all 12 crossing p95 point estimates worse; original target unmet |
| mappo-guided / mappo-original | 5.70 ([3.90, 7.95]) | 12/12 | -2.40 | 10/12 | no vehicle time-loss improvement; joint comparison gate unmet; original target is assessed against actuated only |

### mappo-original: all crossings versus actuated

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | 16.40 | [13.53, 17.87] | [269, 269, 269] | [269, 269, 269] | 0.00/0.00 |
| :C0_c1 | 16.60 | [13.20, 20.93] | [376, 376, 371] | [373, 373, 373] | 0.00/0.00 |
| :C0_c2 | 9.40 | [0.60, 15.13] | [177, 178, 177] | [176, 176, 176] | 0.00/0.00 |
| :C0_c3 | 16.47 | [15.47, 19.80] | [191, 191, 191] | [191, 191, 191] | 0.00/0.00 |
| :C1_c0 | 10.20 | [2.60, 17.40] | [190, 190, 192] | [192, 192, 192] | 0.00/0.00 |
| :C1_c1 | 13.07 | [10.00, 15.40] | [416, 418, 400] | [407, 407, 407] | 0.00/0.00 |
| :C1_c2 | 5.20 | [-0.73, 12.47] | [223, 221, 223] | [223, 223, 223] | 0.00/0.00 |
| :C1_c3 | 10.07 | [4.20, 14.73] | [365, 371, 378] | [373, 373, 373] | 0.00/0.00 |
| :C2_c0 | 13.00 | [2.53, 16.67] | [195, 197, 196] | [194, 194, 194] | 0.00/0.00 |
| :C2_c1 | 15.87 | [10.87, 23.60] | [179, 179, 179] | [179, 179, 179] | 0.00/0.00 |
| :C2_c2 | 13.87 | [9.27, 20.80] | [210, 210, 210] | [210, 210, 210] | 0.00/0.00 |
| :C2_c3 | 16.47 | [14.07, 18.00] | [380, 377, 391] | [387, 387, 387] | 0.00/0.00 |

### mappo-original: all crossings versus mappo-original

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | 0.00 | [0.00, 0.00] | [269, 269, 269] | [269, 269, 269] | 0.00/0.00 |
| :C0_c1 | 0.00 | [0.00, 0.00] | [376, 376, 371] | [376, 376, 371] | 0.00/0.00 |
| :C0_c2 | 0.00 | [0.00, 0.00] | [177, 178, 177] | [177, 178, 177] | 0.00/0.00 |
| :C0_c3 | 0.00 | [0.00, 0.00] | [191, 191, 191] | [191, 191, 191] | 0.00/0.00 |
| :C1_c0 | 0.00 | [0.00, 0.00] | [190, 190, 192] | [190, 190, 192] | 0.00/0.00 |
| :C1_c1 | 0.00 | [0.00, 0.00] | [416, 418, 400] | [416, 418, 400] | 0.00/0.00 |
| :C1_c2 | 0.00 | [0.00, 0.00] | [223, 221, 223] | [223, 221, 223] | 0.00/0.00 |
| :C1_c3 | 0.00 | [0.00, 0.00] | [365, 371, 378] | [365, 371, 378] | 0.00/0.00 |
| :C2_c0 | 0.00 | [0.00, 0.00] | [195, 197, 196] | [195, 197, 196] | 0.00/0.00 |
| :C2_c1 | 0.00 | [0.00, 0.00] | [179, 179, 179] | [179, 179, 179] | 0.00/0.00 |
| :C2_c2 | 0.00 | [0.00, 0.00] | [210, 210, 210] | [210, 210, 210] | 0.00/0.00 |
| :C2_c3 | 0.00 | [0.00, 0.00] | [380, 377, 391] | [380, 377, 391] | 0.00/0.00 |

### mappo-continue: all crossings versus actuated

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | 15.07 | [11.80, 18.33] | [269, 269, 269] | [269, 269, 269] | 0.00/0.00 |
| :C0_c1 | 14.20 | [12.80, 16.33] | [383, 363, 378] | [373, 373, 373] | 0.00/0.00 |
| :C0_c2 | 14.67 | [6.80, 17.20] | [179, 179, 180] | [176, 176, 176] | 0.00/0.00 |
| :C0_c3 | 13.67 | [11.20, 17.80] | [191, 191, 191] | [191, 191, 191] | 0.00/0.00 |
| :C1_c0 | 12.27 | [2.40, 16.40] | [190, 190, 192] | [192, 192, 192] | 0.00/0.00 |
| :C1_c1 | 13.60 | [11.33, 15.47] | [410, 405, 421] | [407, 407, 407] | 0.00/0.00 |
| :C1_c2 | 4.93 | [-0.27, 13.60] | [221, 223, 223] | [223, 223, 223] | 0.00/0.00 |
| :C1_c3 | 13.20 | [9.67, 14.60] | [364, 384, 370] | [373, 373, 373] | 0.00/0.00 |
| :C2_c0 | 14.67 | [9.33, 19.27] | [194, 196, 194] | [194, 194, 194] | 0.00/0.00 |
| :C2_c1 | 17.87 | [13.00, 24.00] | [179, 179, 179] | [179, 179, 179] | 0.00/0.00 |
| :C2_c2 | 15.20 | [12.27, 23.13] | [210, 210, 210] | [210, 210, 210] | 0.00/0.00 |
| :C2_c3 | 15.20 | [13.00, 17.80] | [384, 389, 373] | [387, 387, 387] | 0.00/0.00 |

### mappo-continue: all crossings versus mappo-original

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | -1.33 | [-4.07, 2.73] | [269, 269, 269] | [269, 269, 269] | 0.00/0.00 |
| :C0_c1 | -2.40 | [-6.20, 0.60] | [383, 363, 378] | [376, 376, 371] | 0.00/0.00 |
| :C0_c2 | 5.27 | [-0.80, 13.60] | [179, 179, 180] | [177, 178, 177] | 0.00/0.00 |
| :C0_c3 | -2.80 | [-6.73, 0.73] | [191, 191, 191] | [191, 191, 191] | 0.00/0.00 |
| :C1_c0 | 2.07 | [-8.00, 8.93] | [190, 190, 192] | [190, 190, 192] | 0.00/0.00 |
| :C1_c1 | 0.53 | [-3.40, 4.00] | [410, 405, 421] | [416, 418, 400] | 0.00/0.00 |
| :C1_c2 | -0.27 | [-7.33, 10.47] | [221, 223, 223] | [223, 221, 223] | 0.00/0.00 |
| :C1_c3 | 3.13 | [-2.40, 8.33] | [364, 384, 370] | [365, 371, 378] | 0.00/0.00 |
| :C2_c0 | 1.67 | [-2.87, 10.00] | [194, 196, 194] | [195, 197, 196] | 0.00/0.00 |
| :C2_c1 | 2.00 | [-6.93, 10.27] | [179, 179, 179] | [179, 179, 179] | 0.00/0.00 |
| :C2_c2 | 1.33 | [-5.00, 7.93] | [210, 210, 210] | [210, 210, 210] | 0.00/0.00 |
| :C2_c3 | -1.27 | [-3.87, 1.20] | [384, 389, 373] | [380, 377, 391] | 0.00/0.00 |

### mappo-tail: all crossings versus actuated

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | 11.07 | [2.80, 15.33] | [269, 269, 269] | [269, 269, 269] | 0.00/0.00 |
| :C0_c1 | 14.40 | [12.53, 17.07] | [382, 363, 372] | [373, 373, 373] | 0.00/0.00 |
| :C0_c2 | 9.13 | [1.00, 12.60] | [178, 179, 176] | [176, 176, 176] | 0.00/0.00 |
| :C0_c3 | 13.00 | [8.80, 16.80] | [191, 191, 191] | [191, 191, 191] | 0.00/0.00 |
| :C1_c0 | 5.53 | [0.47, 13.40] | [189, 191, 193] | [192, 192, 192] | 0.00/0.00 |
| :C1_c1 | 11.60 | [6.00, 14.80] | [414, 417, 411] | [407, 407, 407] | 0.00/0.00 |
| :C1_c2 | 4.53 | [0.20, 12.93] | [223, 225, 225] | [223, 223, 223] | 0.00/0.00 |
| :C1_c3 | 10.33 | [4.07, 13.93] | [366, 384, 376] | [373, 373, 373] | 0.00/0.00 |
| :C2_c0 | 5.33 | [-2.60, 11.40] | [195, 195, 197] | [194, 194, 194] | 0.00/0.00 |
| :C2_c1 | 14.40 | [8.13, 17.20] | [179, 179, 179] | [179, 179, 179] | 0.00/0.00 |
| :C2_c2 | 12.20 | [7.67, 14.40] | [210, 210, 210] | [210, 210, 210] | 0.00/0.00 |
| :C2_c3 | 11.13 | [7.87, 15.53] | [381, 384, 379] | [387, 387, 387] | 0.00/0.00 |

### mappo-tail: all crossings versus mappo-original

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | -5.33 | [-11.87, -1.20] | [269, 269, 269] | [269, 269, 269] | 0.00/0.00 |
| :C0_c1 | -2.20 | [-5.80, 1.60] | [382, 363, 372] | [376, 376, 371] | 0.00/0.00 |
| :C0_c2 | -0.27 | [-10.60, 6.47] | [178, 179, 176] | [177, 178, 177] | 0.00/0.00 |
| :C0_c3 | -3.47 | [-9.67, -1.20] | [191, 191, 191] | [191, 191, 191] | 0.00/0.00 |
| :C1_c0 | -4.67 | [-14.67, 6.47] | [189, 191, 193] | [190, 190, 192] | 0.00/0.00 |
| :C1_c1 | -1.47 | [-8.20, 2.20] | [414, 417, 411] | [416, 418, 400] | 0.00/0.00 |
| :C1_c2 | -0.67 | [-9.20, 9.40] | [223, 225, 225] | [223, 221, 223] | 0.00/0.00 |
| :C1_c3 | 0.27 | [-7.20, 5.93] | [366, 384, 376] | [365, 371, 378] | 0.00/0.00 |
| :C2_c0 | -7.67 | [-15.00, 3.00] | [195, 195, 197] | [195, 197, 196] | 0.00/0.00 |
| :C2_c1 | -1.47 | [-12.13, 4.87] | [179, 179, 179] | [179, 179, 179] | 0.00/0.00 |
| :C2_c2 | -1.67 | [-7.47, 2.53] | [210, 210, 210] | [210, 210, 210] | 0.00/0.00 |
| :C2_c3 | -5.33 | [-7.80, -1.33] | [381, 384, 379] | [380, 377, 391] | 0.00/0.00 |

### mappo-guided: all crossings versus actuated

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | 6.33 | [3.00, 8.80] | [269, 269, 269] | [269, 269, 269] | 0.00/0.00 |
| :C0_c1 | 7.93 | [2.67, 12.53] | [378, 373, 386] | [373, 373, 373] | 0.00/0.00 |
| :C0_c2 | 5.67 | [-1.07, 12.00] | [179, 178, 175] | [176, 176, 176] | 0.00/0.00 |
| :C0_c3 | 10.80 | [6.13, 14.80] | [191, 191, 191] | [191, 191, 191] | 0.00/0.00 |
| :C1_c0 | 1.67 | [-0.40, 6.67] | [191, 190, 189] | [192, 192, 192] | 0.00/0.00 |
| :C1_c1 | 3.80 | [0.60, 9.20] | [420, 428, 412] | [407, 407, 407] | 0.00/0.00 |
| :C1_c2 | 0.87 | [-2.20, 5.60] | [225, 223, 223] | [223, 223, 223] | 0.00/0.00 |
| :C1_c3 | 7.67 | [1.20, 10.53] | [370, 374, 362] | [373, 373, 373] | 0.00/0.00 |
| :C2_c0 | 3.60 | [0.20, 13.27] | [192, 193, 195] | [194, 194, 194] | 0.00/0.00 |
| :C2_c1 | 10.60 | [5.40, 13.80] | [179, 179, 179] | [179, 179, 179] | 0.00/0.00 |
| :C2_c2 | 7.20 | [2.33, 12.40] | [210, 210, 210] | [210, 210, 210] | 0.00/0.00 |
| :C2_c3 | 5.93 | [2.00, 10.73] | [373, 370, 385] | [387, 387, 387] | 0.00/0.00 |

### mappo-guided: all crossings versus mappo-original

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | -10.07 | [-12.93, -6.47] | [269, 269, 269] | [269, 269, 269] | 0.00/0.00 |
| :C0_c1 | -8.67 | [-14.60, -2.47] | [378, 373, 386] | [376, 376, 371] | 0.00/0.00 |
| :C0_c2 | -3.73 | [-13.80, 4.20] | [179, 178, 175] | [177, 178, 177] | 0.00/0.00 |
| :C0_c3 | -5.67 | [-10.40, -2.33] | [191, 191, 191] | [191, 191, 191] | 0.00/0.00 |
| :C1_c0 | -8.53 | [-15.20, -0.40] | [191, 190, 189] | [190, 190, 192] | 0.00/0.00 |
| :C1_c1 | -9.27 | [-12.80, -4.33] | [420, 428, 412] | [416, 418, 400] | 0.00/0.00 |
| :C1_c2 | -4.33 | [-12.73, 2.33] | [225, 223, 223] | [223, 221, 223] | 0.00/0.00 |
| :C1_c3 | -2.40 | [-10.33, 4.20] | [370, 374, 362] | [365, 371, 378] | 0.00/0.00 |
| :C2_c0 | -9.40 | [-15.00, 5.00] | [192, 193, 195] | [195, 197, 196] | 0.00/0.00 |
| :C2_c1 | -5.27 | [-16.33, 1.53] | [179, 179, 179] | [179, 179, 179] | 0.00/0.00 |
| :C2_c2 | -6.67 | [-15.20, -1.40] | [210, 210, 210] | [210, 210, 210] | 0.00/0.00 |
| :C2_c3 | -10.53 | [-15.00, -5.07] | [373, 370, 385] | [380, 377, 391] | 0.00/0.00 |

The full JSON also retains person-trip service, slower-walker groups where recorded, zero-entry runs, cutoff-only failures, whole-horizon vehicle counts, and entered-but-not-exited diagnostics where available. Missing diagnostics remain unavailable rather than zero.

## short-spacing

| Controller | Vehicle loss s | Insertion delay s | Scheduled | Completed | Unfinished | Uninserted | Pending cutoff | Unresolved followup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed | 37.11 | 8.02 | 2391.25 | 2288.50 | 86.75 | 16.00 | 11.50 | 0.00 |
| actuated | 39.39 | 13.33 | 2391.25 | 729.25 | 67.75 | 1594.25 | 9.00 | 0.00 |
| pressure | 52.24 | 78.73 | 2391.25 | 1923.75 | 86.50 | 381.00 | 7.25 | 0.00 |
| mappo-original | 65.86 | 105.29 | 2391.25 | 1741.25 | 93.17 | 556.83 | 6.08 | 0.00 |
| mappo-continue | 61.50 | 95.87 | 2391.25 | 1810.50 | 87.50 | 493.25 | 7.50 | 0.00 |
| mappo-tail | 62.44 | 107.31 | 2391.25 | 1704.50 | 90.08 | 596.67 | 4.17 | 0.00 |
| mappo-guided | 66.24 | 102.31 | 2391.25 | 1544.92 | 87.08 | 759.25 | 3.58 | 0.00 |

All-horizon vehicle service below includes warmup and is a different cohort from the primary measurements above.

| Controller | Horizon scheduled | Horizon completed | Horizon unfinished | Horizon uninserted |
|---|---:|---:|---:|---:|
| fixed | 2786.75 | 2786.75 | 0.00 | 0.00 |
| actuated | 2786.75 | 1285.50 | 82.25 | 1419.00 |
| pressure | 2786.75 | 2526.25 | 22.25 | 238.25 |
| mappo-original | 2786.75 | 2328.50 | 47.58 | 410.67 |
| mappo-continue | 2786.75 | 2395.67 | 38.58 | 352.50 |
| mappo-tail | 2786.75 | 2292.42 | 47.42 | 446.92 |
| mappo-guided | 2786.75 | 2122.17 | 51.75 | 612.83 |

| Controller | Persons scheduled | Persons completed | Persons unfinished | Persons uninserted | Entered crossing without observed exit (all horizon) |
|---|---:|---:|---:|---:|---:|
| fixed | 692.00 | 692.00 | 0.00 | 0.00 | 0.00 |
| actuated | 692.00 | 692.00 | 0.00 | 0.00 | 0.00 |
| pressure | 692.00 | 692.00 | 0.00 | 0.00 | 0.00 |
| mappo-original | 692.00 | 692.00 | 0.00 | 0.00 | 0.00 |
| mappo-continue | 692.00 | 692.00 | 0.00 | 0.00 | 0.00 |
| mappo-tail | 692.00 | 692.00 | 0.00 | 0.00 | 0.00 |
| mappo-guided | 692.00 | 692.00 | 0.00 | 0.00 | 0.00 |

| Policy / reference | Vehicle delta s (exploratory 95% interval) | Crossings no worse | Worst crossing delta s | Strict service failures | Interpretation |
|---|---|---:|---:|---:|---|
| mappo-original / actuated | 26.47 ([15.62, 37.40]) | 0/12 | 33.13 | 12/12 | no vehicle time-loss improvement; original target unmet |
| mappo-original / mappo-original | 0.00 ([0.00, 0.00]) | 12/12 | 0.00 | 0/12 | no vehicle time-loss improvement; joint comparison gate unmet; original target is assessed against actuated only |
| mappo-continue / actuated | 22.11 ([10.62, 32.26]) | 0/12 | 34.33 | 12/12 | no vehicle time-loss improvement; original target unmet |
| mappo-continue / mappo-original | -4.36 ([-7.52, -0.76]) | 4/12 | 10.87 | 11/12 | vehicle gain with 4/12 crossing p95 point estimates no higher; joint comparison gate unmet; strict service deterioration also present; original target is assessed against actuated only |
| mappo-tail / actuated | 23.05 ([10.92, 34.64]) | 0/12 | 28.07 | 12/12 | no vehicle time-loss improvement; original target unmet |
| mappo-tail / mappo-original | -3.42 ([-8.77, 1.41]) | 12/12 | -1.33 | 12/12 | vehicle gain with 12/12 crossing p95 point estimates no higher; joint comparison gate unmet; strict service deterioration also present; original target is assessed against actuated only |
| mappo-guided / actuated | 26.85 ([16.28, 37.57]) | 0/12 | 16.40 | 12/12 | no vehicle time-loss improvement; original target unmet |
| mappo-guided / mappo-original | 0.38 ([-6.31, 5.85]) | 12/12 | -9.27 | 12/12 | no vehicle time-loss improvement; joint comparison gate unmet; original target is assessed against actuated only |

### mappo-original: all crossings versus actuated

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | 28.73 | [24.67, 36.27] | [190, 190, 190] | [190, 190, 190] | 0.00/0.00 |
| :C0_c1 | 30.33 | [21.67, 39.20] | [403, 399, 410] | [388, 388, 388] | 0.00/0.00 |
| :C0_c2 | 20.13 | [10.60, 30.33] | [212, 207, 213] | [210, 210, 210] | 0.00/0.00 |
| :C0_c3 | 33.13 | [22.00, 38.40] | [212, 212, 212] | [212, 212, 212] | 0.00/0.00 |
| :C1_c0 | 10.60 | [1.00, 18.00] | [200, 199, 202] | [196, 196, 196] | 0.00/0.00 |
| :C1_c1 | 21.80 | [-95.07, 27.87] | [397, 386, 394] | [391, 391, 391] | 0.00/0.00 |
| :C1_c2 | 14.40 | [2.60, 22.87] | [197, 195, 198] | [193, 193, 193] | 0.00/0.00 |
| :C1_c3 | 19.40 | [-119.27, 24.00] | [415, 416, 407] | [427, 427, 427] | 0.00/0.00 |
| :C2_c0 | 20.93 | [6.73, 32.80] | [192, 193, 190] | [191, 191, 191] | 0.00/0.00 |
| :C2_c1 | 31.67 | [23.60, 34.13] | [191, 191, 191] | [191, 191, 191] | 0.00/0.00 |
| :C2_c2 | 29.20 | [22.67, 32.40] | [200, 200, 200] | [200, 200, 200] | 0.00/0.00 |
| :C2_c3 | 28.00 | [22.80, 36.40] | [372, 382, 374] | [377, 377, 377] | 0.00/0.00 |

### mappo-original: all crossings versus mappo-original

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | 0.00 | [0.00, 0.00] | [190, 190, 190] | [190, 190, 190] | 0.00/0.00 |
| :C0_c1 | 0.00 | [0.00, 0.00] | [403, 399, 410] | [403, 399, 410] | 0.00/0.00 |
| :C0_c2 | 0.00 | [0.00, 0.00] | [212, 207, 213] | [212, 207, 213] | 0.00/0.00 |
| :C0_c3 | 0.00 | [0.00, 0.00] | [212, 212, 212] | [212, 212, 212] | 0.00/0.00 |
| :C1_c0 | 0.00 | [0.00, 0.00] | [200, 199, 202] | [200, 199, 202] | 0.00/0.00 |
| :C1_c1 | 0.00 | [0.00, 0.00] | [397, 386, 394] | [397, 386, 394] | 0.00/0.00 |
| :C1_c2 | 0.00 | [0.00, 0.00] | [197, 195, 198] | [197, 195, 198] | 0.00/0.00 |
| :C1_c3 | 0.00 | [0.00, 0.00] | [415, 416, 407] | [415, 416, 407] | 0.00/0.00 |
| :C2_c0 | 0.00 | [0.00, 0.00] | [192, 193, 190] | [192, 193, 190] | 0.00/0.00 |
| :C2_c1 | 0.00 | [0.00, 0.00] | [191, 191, 191] | [191, 191, 191] | 0.00/0.00 |
| :C2_c2 | 0.00 | [0.00, 0.00] | [200, 200, 200] | [200, 200, 200] | 0.00/0.00 |
| :C2_c3 | 0.00 | [0.00, 0.00] | [372, 382, 374] | [372, 382, 374] | 0.00/0.00 |

### mappo-continue: all crossings versus actuated

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | 34.33 | [20.93, 44.33] | [190, 190, 190] | [190, 190, 190] | 0.00/0.00 |
| :C0_c1 | 30.07 | [23.47, 39.60] | [402, 407, 407] | [388, 388, 388] | 0.00/0.00 |
| :C0_c2 | 21.53 | [5.20, 29.53] | [210, 209, 209] | [210, 210, 210] | 0.00/0.00 |
| :C0_c3 | 26.13 | [20.47, 34.93] | [212, 212, 212] | [212, 212, 212] | 0.00/0.00 |
| :C1_c0 | 21.47 | [14.40, 30.93] | [201, 198, 204] | [196, 196, 196] | 0.00/0.00 |
| :C1_c1 | 23.00 | [-99.60, 28.13] | [390, 389, 388] | [391, 391, 391] | 0.00/0.00 |
| :C1_c2 | 17.20 | [3.00, 19.40] | [197, 199, 197] | [193, 193, 193] | 0.00/0.00 |
| :C1_c3 | 23.27 | [-113.07, 28.60] | [414, 409, 409] | [427, 427, 427] | 0.00/0.00 |
| :C2_c0 | 16.27 | [3.87, 28.27] | [195, 192, 190] | [191, 191, 191] | 0.00/0.00 |
| :C2_c1 | 32.53 | [28.40, 34.40] | [191, 191, 191] | [191, 191, 191] | 0.00/0.00 |
| :C2_c2 | 28.13 | [23.07, 33.87] | [200, 200, 200] | [200, 200, 200] | 0.00/0.00 |
| :C2_c3 | 28.47 | [25.27, 31.47] | [378, 379, 380] | [377, 377, 377] | 0.00/0.00 |

### mappo-continue: all crossings versus mappo-original

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | 5.60 | [-8.20, 14.53] | [190, 190, 190] | [190, 190, 190] | 0.00/0.00 |
| :C0_c1 | -0.27 | [-10.67, 11.40] | [402, 407, 407] | [403, 399, 410] | 0.00/0.00 |
| :C0_c2 | 1.40 | [-21.93, 12.67] | [210, 209, 209] | [212, 207, 213] | 0.00/0.00 |
| :C0_c3 | -7.00 | [-15.80, 6.87] | [212, 212, 212] | [212, 212, 212] | 0.00/0.00 |
| :C1_c0 | 10.87 | [-1.73, 25.27] | [201, 198, 204] | [200, 199, 202] | 0.00/0.00 |
| :C1_c1 | 1.20 | [-8.67, 7.93] | [390, 389, 388] | [397, 386, 394] | 0.00/0.00 |
| :C1_c2 | 2.80 | [-8.13, 13.93] | [197, 199, 197] | [197, 195, 198] | 0.00/0.00 |
| :C1_c3 | 3.87 | [-2.20, 10.20] | [414, 409, 409] | [415, 416, 407] | 0.00/0.00 |
| :C2_c0 | -4.67 | [-20.07, 11.73] | [195, 192, 190] | [192, 193, 190] | 0.00/0.00 |
| :C2_c1 | 0.87 | [-4.20, 9.73] | [191, 191, 191] | [191, 191, 191] | 0.00/0.00 |
| :C2_c2 | -1.07 | [-4.27, 5.80] | [200, 200, 200] | [200, 200, 200] | 0.00/0.00 |
| :C2_c3 | 0.47 | [-7.13, 5.13] | [378, 379, 380] | [372, 382, 374] | 0.00/0.00 |

### mappo-tail: all crossings versus actuated

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | 20.27 | [13.33, 26.67] | [190, 190, 190] | [190, 190, 190] | 0.00/0.00 |
| :C0_c1 | 21.53 | [16.40, 26.07] | [405, 406, 403] | [388, 388, 388] | 0.00/0.00 |
| :C0_c2 | 10.27 | [0.40, 18.80] | [210, 209, 211] | [210, 210, 210] | 0.00/0.00 |
| :C0_c3 | 25.07 | [19.67, 28.40] | [212, 212, 212] | [212, 212, 212] | 0.00/0.00 |
| :C1_c0 | 7.40 | [-0.13, 22.53] | [201, 199, 203] | [196, 196, 196] | 0.00/0.00 |
| :C1_c1 | 17.33 | [-103.53, 22.20] | [393, 394, 390] | [391, 391, 391] | 0.00/0.00 |
| :C1_c2 | 12.33 | [2.20, 20.33] | [199, 197, 197] | [193, 193, 193] | 0.00/0.00 |
| :C1_c3 | 16.27 | [-120.67, 21.27] | [411, 411, 413] | [427, 427, 427] | 0.00/0.00 |
| :C2_c0 | 19.60 | [5.13, 29.60] | [192, 192, 187] | [191, 191, 191] | 0.00/0.00 |
| :C2_c1 | 28.07 | [21.60, 33.40] | [191, 191, 191] | [191, 191, 191] | 0.00/0.00 |
| :C2_c2 | 22.53 | [16.33, 29.20] | [200, 200, 200] | [200, 200, 200] | 0.00/0.00 |
| :C2_c3 | 21.33 | [17.13, 24.27] | [375, 374, 378] | [377, 377, 377] | 0.00/0.00 |

### mappo-tail: all crossings versus mappo-original

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | -8.47 | [-18.40, -1.20] | [190, 190, 190] | [190, 190, 190] | 0.00/0.00 |
| :C0_c1 | -8.80 | [-16.67, -1.73] | [405, 406, 403] | [403, 399, 410] | 0.00/0.00 |
| :C0_c2 | -9.87 | [-29.80, 6.53] | [210, 209, 211] | [212, 207, 213] | 0.00/0.00 |
| :C0_c3 | -8.07 | [-17.33, 2.33] | [212, 212, 212] | [212, 212, 212] | 0.00/0.00 |
| :C1_c0 | -3.20 | [-13.27, 12.40] | [201, 199, 203] | [200, 199, 202] | 0.00/0.00 |
| :C1_c1 | -4.47 | [-12.60, 2.80] | [393, 394, 390] | [397, 386, 394] | 0.00/0.00 |
| :C1_c2 | -2.07 | [-15.93, 16.00] | [199, 197, 197] | [197, 195, 198] | 0.00/0.00 |
| :C1_c3 | -3.13 | [-9.47, 4.47] | [411, 411, 413] | [415, 416, 407] | 0.00/0.00 |
| :C2_c0 | -1.33 | [-20.07, 16.00] | [192, 192, 187] | [192, 193, 190] | 0.00/0.00 |
| :C2_c1 | -3.60 | [-9.80, 5.60] | [191, 191, 191] | [191, 191, 191] | 0.00/0.00 |
| :C2_c2 | -6.67 | [-12.00, 2.33] | [200, 200, 200] | [200, 200, 200] | 0.00/0.00 |
| :C2_c3 | -6.67 | [-14.73, -1.73] | [375, 374, 378] | [372, 382, 374] | 0.00/0.00 |

### mappo-guided: all crossings versus actuated

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | 14.33 | [8.87, 25.80] | [190, 190, 190] | [190, 190, 190] | 0.00/0.00 |
| :C0_c1 | 15.87 | [11.40, 24.20] | [402, 404, 396] | [388, 388, 388] | 0.00/0.00 |
| :C0_c2 | 4.67 | [-0.53, 12.80] | [211, 208, 211] | [210, 210, 210] | 0.00/0.00 |
| :C0_c3 | 16.40 | [8.53, 21.13] | [212, 212, 212] | [212, 212, 212] | 0.00/0.00 |
| :C1_c0 | 1.33 | [-1.60, 12.00] | [199, 203, 202] | [196, 196, 196] | 0.00/0.00 |
| :C1_c1 | 6.67 | [-111.00, 10.80] | [396, 391, 393] | [391, 391, 391] | 0.00/0.00 |
| :C1_c2 | 3.73 | [-0.40, 12.40] | [199, 198, 195] | [193, 193, 193] | 0.00/0.00 |
| :C1_c3 | 7.33 | [-126.87, 12.80] | [416, 412, 420] | [427, 427, 427] | 0.00/0.00 |
| :C2_c0 | 5.47 | [0.47, 13.80] | [191, 190, 189] | [191, 191, 191] | 0.00/0.00 |
| :C2_c1 | 13.80 | [6.00, 20.00] | [191, 191, 191] | [191, 191, 191] | 0.00/0.00 |
| :C2_c2 | 13.93 | [9.20, 22.87] | [200, 200, 200] | [200, 200, 200] | 0.00/0.00 |
| :C2_c3 | 14.60 | [9.73, 17.60] | [372, 376, 374] | [377, 377, 377] | 0.00/0.00 |

### mappo-guided: all crossings versus mappo-original

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | -14.40 | [-22.73, -5.13] | [190, 190, 190] | [190, 190, 190] | 0.00/0.00 |
| :C0_c1 | -14.47 | [-21.13, -7.87] | [402, 404, 396] | [403, 399, 410] | 0.00/0.00 |
| :C0_c2 | -15.47 | [-30.60, -3.40] | [211, 208, 211] | [212, 207, 213] | 0.00/0.00 |
| :C0_c3 | -16.73 | [-24.47, -9.20] | [212, 212, 212] | [212, 212, 212] | 0.00/0.00 |
| :C1_c0 | -9.27 | [-17.33, 5.93] | [199, 203, 202] | [200, 199, 202] | 0.00/0.00 |
| :C1_c1 | -15.13 | [-23.07, -9.00] | [396, 391, 393] | [397, 386, 394] | 0.00/0.00 |
| :C1_c2 | -10.67 | [-21.47, 0.47] | [199, 198, 195] | [197, 195, 198] | 0.00/0.00 |
| :C1_c3 | -12.07 | [-17.27, -5.27] | [416, 412, 420] | [415, 416, 407] | 0.00/0.00 |
| :C2_c0 | -15.47 | [-28.60, 0.87] | [191, 190, 189] | [192, 193, 190] | 0.00/0.00 |
| :C2_c1 | -17.87 | [-23.93, -6.27] | [191, 191, 191] | [191, 191, 191] | 0.00/0.00 |
| :C2_c2 | -15.27 | [-20.80, -4.80] | [200, 200, 200] | [200, 200, 200] | 0.00/0.00 |
| :C2_c3 | -13.40 | [-20.60, -8.40] | [372, 376, 374] | [372, 382, 374] | 0.00/0.00 |

The full JSON also retains person-trip service, slower-walker groups where recorded, zero-entry runs, cutoff-only failures, whole-horizon vehicle counts, and entered-but-not-exited diagnostics where available. Missing diagnostics remain unavailable rather than zero.

## slow-walkers

| Controller | Vehicle loss s | Insertion delay s | Scheduled | Completed | Unfinished | Uninserted | Pending cutoff | Unresolved followup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed | 33.14 | 0.33 | 963.75 | 933.50 | 30.00 | 0.25 | 21.25 | 0.00 |
| actuated | 53.09 | 0.33 | 963.75 | 927.50 | 36.00 | 0.25 | 4.25 | 0.00 |
| pressure | 43.99 | 0.33 | 963.75 | 932.00 | 31.50 | 0.25 | 5.75 | 0.00 |
| mappo-original | 43.26 | 0.34 | 963.75 | 933.42 | 30.08 | 0.25 | 6.33 | 0.00 |
| mappo-continue | 41.09 | 0.33 | 963.75 | 935.25 | 28.25 | 0.25 | 8.58 | 0.00 |
| mappo-tail | 44.39 | 0.35 | 963.75 | 934.58 | 28.92 | 0.25 | 6.75 | 0.00 |
| mappo-guided | 49.37 | 0.34 | 963.75 | 931.58 | 31.92 | 0.25 | 5.75 | 0.00 |

All-horizon vehicle service below includes warmup and is a different cohort from the primary measurements above.

| Controller | Horizon scheduled | Horizon completed | Horizon unfinished | Horizon uninserted |
|---|---:|---:|---:|---:|
| fixed | 1115.75 | 1115.75 | 0.00 | 0.00 |
| actuated | 1115.75 | 1115.75 | 0.00 | 0.00 |
| pressure | 1115.75 | 1115.75 | 0.00 | 0.00 |
| mappo-original | 1115.75 | 1115.67 | 0.08 | 0.00 |
| mappo-continue | 1115.75 | 1115.58 | 0.17 | 0.00 |
| mappo-tail | 1115.75 | 1115.75 | 0.00 | 0.00 |
| mappo-guided | 1115.75 | 1115.75 | 0.00 | 0.00 |

| Controller | Persons scheduled | Persons completed | Persons unfinished | Persons uninserted | Entered crossing without observed exit (all horizon) |
|---|---:|---:|---:|---:|---:|
| fixed | 681.25 | 677.25 | 4.00 | 0.00 | 0.00 |
| actuated | 681.25 | 681.25 | 0.00 | 0.00 | 0.00 |
| pressure | 681.25 | 681.25 | 0.00 | 0.00 | 0.00 |
| mappo-original | 681.25 | 681.08 | 0.17 | 0.00 | 0.00 |
| mappo-continue | 681.25 | 681.25 | 0.00 | 0.00 | 0.00 |
| mappo-tail | 681.25 | 681.25 | 0.00 | 0.00 | 0.00 |
| mappo-guided | 681.25 | 681.25 | 0.00 | 0.00 | 0.00 |

| Policy / reference | Vehicle delta s (exploratory 95% interval) | Crossings no worse | Worst crossing delta s | Strict service failures | Interpretation |
|---|---|---:|---:|---:|---|
| mappo-original / actuated | -9.83 ([-13.15, -6.53]) | 0/12 | 19.87 | 12/12 | vehicle gain with all 12 crossing p95 point estimates worse; original target unmet |
| mappo-original / mappo-original | 0.00 ([0.00, 0.00]) | 12/12 | 0.00 | 0/12 | no vehicle time-loss improvement; joint comparison gate unmet; original target is assessed against actuated only |
| mappo-continue / actuated | -12.00 ([-15.15, -9.10]) | 0/12 | 22.67 | 12/12 | vehicle gain with all 12 crossing p95 point estimates worse; original target unmet |
| mappo-continue / mappo-original | -2.17 ([-3.69, -0.72]) | 3/12 | 4.47 | 9/12 | vehicle gain with 3/12 crossing p95 point estimates no higher; joint comparison gate unmet; strict service deterioration also present; original target is assessed against actuated only |
| mappo-tail / actuated | -8.70 ([-12.21, -5.14]) | 0/12 | 17.13 | 12/12 | vehicle gain with all 12 crossing p95 point estimates worse; original target unmet |
| mappo-tail / mappo-original | 1.13 ([-0.09, 2.54]) | 9/12 | 2.27 | 12/12 | no vehicle time-loss improvement; joint comparison gate unmet; original target is assessed against actuated only |
| mappo-guided / actuated | -3.72 ([-7.35, -0.76]) | 0/12 | 17.00 | 11/12 | vehicle gain with all 12 crossing p95 point estimates worse; original target unmet |
| mappo-guided / mappo-original | 6.11 ([5.15, 7.35]) | 12/12 | -1.07 | 11/12 | no vehicle time-loss improvement; joint comparison gate unmet; original target is assessed against actuated only |

### mappo-original: all crossings versus actuated

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | 14.47 | [12.13, 20.07] | [187, 187, 187] | [187, 187, 187] | 0.00/0.00 |
| :C0_c1 | 13.40 | [10.80, 16.60] | [357, 383, 359] | [373, 373, 373] | 0.00/0.00 |
| :C0_c2 | 18.20 | [17.00, 28.40] | [177, 177, 179] | [178, 178, 178] | 0.00/0.00 |
| :C0_c3 | 14.67 | [12.40, 16.80] | [197, 197, 197] | [197, 197, 197] | 0.00/0.00 |
| :C1_c0 | 17.07 | [15.53, 18.27] | [194, 192, 194] | [191, 191, 191] | 0.00/0.00 |
| :C1_c1 | 10.13 | [5.93, 13.07] | [400, 396, 399] | [408, 408, 408] | 0.00/0.00 |
| :C1_c2 | 16.60 | [15.20, 18.07] | [229, 229, 230] | [228, 228, 228] | 0.00/0.00 |
| :C1_c3 | 8.00 | [2.27, 10.80] | [386, 366, 387] | [374, 374, 374] | 0.00/0.00 |
| :C2_c0 | 19.87 | [16.40, 28.00] | [200, 197, 198] | [197, 197, 197] | 0.00/0.00 |
| :C2_c1 | 12.53 | [7.73, 17.60] | [177, 177, 177] | [177, 177, 177] | 0.00/0.00 |
| :C2_c2 | 14.07 | [12.00, 16.67] | [204, 204, 204] | [204, 204, 204] | 0.00/0.00 |
| :C2_c3 | 12.33 | [9.07, 15.47] | [391, 397, 393] | [383, 383, 383] | 0.00/0.00 |

### mappo-original: all crossings versus mappo-original

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | 0.00 | [0.00, 0.00] | [187, 187, 187] | [187, 187, 187] | 0.00/0.00 |
| :C0_c1 | 0.00 | [0.00, 0.00] | [357, 383, 359] | [357, 383, 359] | 0.00/0.00 |
| :C0_c2 | 0.00 | [0.00, 0.00] | [177, 177, 179] | [177, 177, 179] | 0.00/0.00 |
| :C0_c3 | 0.00 | [0.00, 0.00] | [197, 197, 197] | [197, 197, 197] | 0.00/0.00 |
| :C1_c0 | 0.00 | [0.00, 0.00] | [194, 192, 194] | [194, 192, 194] | 0.00/0.00 |
| :C1_c1 | 0.00 | [0.00, 0.00] | [400, 396, 399] | [400, 396, 399] | 0.00/0.00 |
| :C1_c2 | 0.00 | [0.00, 0.00] | [229, 229, 230] | [229, 229, 230] | 0.00/0.00 |
| :C1_c3 | 0.00 | [0.00, 0.00] | [386, 366, 387] | [386, 366, 387] | 0.00/0.00 |
| :C2_c0 | 0.00 | [0.00, 0.00] | [200, 197, 198] | [200, 197, 198] | 0.00/0.00 |
| :C2_c1 | 0.00 | [0.00, 0.00] | [177, 177, 177] | [177, 177, 177] | 0.00/0.00 |
| :C2_c2 | 0.00 | [0.00, 0.00] | [204, 204, 204] | [204, 204, 204] | 0.00/0.00 |
| :C2_c3 | 0.00 | [0.00, 0.00] | [391, 397, 393] | [391, 397, 393] | 0.00/0.00 |

### mappo-continue: all crossings versus actuated

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | 15.93 | [14.20, 17.87] | [187, 187, 187] | [187, 187, 187] | 0.00/0.00 |
| :C0_c1 | 15.47 | [11.87, 18.33] | [375, 380, 370] | [373, 373, 373] | 0.00/0.00 |
| :C0_c2 | 22.67 | [17.07, 31.20] | [179, 177, 176] | [178, 178, 178] | 0.00/0.00 |
| :C0_c3 | 15.20 | [10.33, 18.00] | [197, 197, 197] | [197, 197, 197] | 0.00/0.00 |
| :C1_c0 | 17.00 | [16.47, 18.20] | [195, 193, 196] | [191, 191, 191] | 0.00/0.00 |
| :C1_c1 | 11.53 | [8.93, 13.80] | [407, 410, 404] | [408, 408, 408] | 0.00/0.00 |
| :C1_c2 | 16.60 | [15.20, 17.00] | [229, 227, 227] | [228, 228, 228] | 0.00/0.00 |
| :C1_c3 | 10.53 | [8.80, 13.20] | [374, 364, 373] | [374, 374, 374] | 0.00/0.00 |
| :C2_c0 | 17.53 | [15.33, 24.40] | [196, 199, 197] | [197, 197, 197] | 0.00/0.00 |
| :C2_c1 | 13.60 | [7.73, 16.33] | [177, 177, 177] | [177, 177, 177] | 0.00/0.00 |
| :C2_c2 | 13.13 | [8.80, 16.07] | [204, 204, 204] | [204, 204, 204] | 0.00/0.00 |
| :C2_c3 | 13.13 | [10.13, 16.40] | [386, 381, 391] | [383, 383, 383] | 0.00/0.00 |

### mappo-continue: all crossings versus mappo-original

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | 1.47 | [-3.40, 4.53] | [187, 187, 187] | [187, 187, 187] | 0.00/0.00 |
| :C0_c1 | 2.07 | [-1.60, 6.93] | [375, 380, 370] | [357, 383, 359] | 0.00/0.00 |
| :C0_c2 | 4.47 | [-1.40, 10.80] | [179, 177, 176] | [177, 177, 179] | 0.00/0.00 |
| :C0_c3 | 0.53 | [-3.73, 3.13] | [197, 197, 197] | [197, 197, 197] | 0.00/0.00 |
| :C1_c0 | -0.07 | [-1.27, 1.47] | [195, 193, 196] | [194, 192, 194] | 0.00/0.00 |
| :C1_c1 | 1.40 | [-2.33, 6.00] | [407, 410, 404] | [400, 396, 399] | 0.00/0.00 |
| :C1_c2 | 0.00 | [-1.87, 0.53] | [229, 227, 227] | [229, 229, 230] | 0.00/0.00 |
| :C1_c3 | 2.53 | [-0.73, 7.87] | [374, 364, 373] | [386, 366, 387] | 0.00/0.00 |
| :C2_c0 | -2.33 | [-10.93, 5.33] | [196, 199, 197] | [200, 197, 198] | 0.00/0.00 |
| :C2_c1 | 1.07 | [-5.80, 5.80] | [177, 177, 177] | [177, 177, 177] | 0.00/0.00 |
| :C2_c2 | -0.93 | [-6.47, 2.40] | [204, 204, 204] | [204, 204, 204] | 0.00/0.00 |
| :C2_c3 | 0.80 | [-1.73, 3.40] | [386, 381, 391] | [391, 397, 393] | 0.00/0.00 |

### mappo-tail: all crossings versus actuated

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | 9.00 | [4.67, 14.60] | [187, 187, 187] | [187, 187, 187] | 0.00/0.00 |
| :C0_c1 | 13.13 | [9.07, 16.13] | [382, 386, 376] | [373, 373, 373] | 0.00/0.00 |
| :C0_c2 | 17.13 | [17.00, 27.80] | [178, 179, 179] | [178, 178, 178] | 0.00/0.00 |
| :C0_c3 | 12.60 | [8.00, 14.80] | [197, 197, 197] | [197, 197, 197] | 0.00/0.00 |
| :C1_c0 | 17.00 | [15.00, 17.00] | [197, 193, 194] | [191, 191, 191] | 0.00/0.00 |
| :C1_c1 | 9.60 | [4.00, 13.80] | [392, 420, 406] | [408, 408, 408] | 0.00/0.00 |
| :C1_c2 | 16.60 | [14.33, 17.00] | [230, 234, 227] | [228, 228, 228] | 0.00/0.00 |
| :C1_c3 | 10.27 | [4.93, 13.53] | [362, 359, 376] | [374, 374, 374] | 0.00/0.00 |
| :C2_c0 | 16.67 | [15.27, 19.80] | [197, 198, 198] | [197, 197, 197] | 0.00/0.00 |
| :C2_c1 | 9.47 | [1.87, 17.40] | [177, 177, 177] | [177, 177, 177] | 0.00/0.00 |
| :C2_c2 | 14.47 | [10.33, 21.53] | [204, 204, 204] | [204, 204, 204] | 0.00/0.00 |
| :C2_c3 | 11.53 | [7.47, 16.40] | [399, 376, 385] | [383, 383, 383] | 0.00/0.00 |

### mappo-tail: all crossings versus mappo-original

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | -5.47 | [-13.00, 1.80] | [187, 187, 187] | [187, 187, 187] | 0.00/0.00 |
| :C0_c1 | -0.27 | [-5.07, 2.20] | [382, 386, 376] | [357, 383, 359] | 0.00/0.00 |
| :C0_c2 | -1.07 | [-10.80, 9.60] | [178, 179, 179] | [177, 177, 179] | 0.00/0.00 |
| :C0_c3 | -2.07 | [-6.80, 0.47] | [197, 197, 197] | [197, 197, 197] | 0.00/0.00 |
| :C1_c0 | -0.07 | [-2.27, 0.47] | [197, 193, 194] | [194, 192, 194] | 0.00/0.00 |
| :C1_c1 | -0.53 | [-5.93, 3.53] | [392, 420, 406] | [400, 396, 399] | 0.00/0.00 |
| :C1_c2 | 0.00 | [-4.27, 0.27] | [230, 234, 227] | [229, 229, 230] | 0.00/0.00 |
| :C1_c3 | 2.27 | [-1.20, 6.53] | [362, 359, 376] | [386, 366, 387] | 0.00/0.00 |
| :C2_c0 | -3.20 | [-11.33, 1.00] | [197, 198, 198] | [200, 197, 198] | 0.00/0.00 |
| :C2_c1 | -3.07 | [-11.87, 5.67] | [177, 177, 177] | [177, 177, 177] | 0.00/0.00 |
| :C2_c2 | 0.40 | [-3.80, 7.13] | [204, 204, 204] | [204, 204, 204] | 0.00/0.00 |
| :C2_c3 | -0.80 | [-4.73, 2.67] | [399, 376, 385] | [391, 397, 393] | 0.00/0.00 |

### mappo-guided: all crossings versus actuated

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | 5.27 | [1.20, 7.27] | [187, 187, 187] | [187, 187, 187] | 0.00/0.00 |
| :C0_c1 | 9.53 | [3.40, 15.07] | [385, 386, 383] | [373, 373, 373] | 0.00/0.00 |
| :C0_c2 | 17.00 | [16.20, 17.67] | [179, 178, 177] | [178, 178, 178] | 0.00/0.00 |
| :C0_c3 | 3.73 | [0.87, 9.07] | [197, 197, 197] | [197, 197, 197] | 0.00/0.00 |
| :C1_c0 | 11.67 | [1.80, 17.00] | [192, 193, 193] | [191, 191, 191] | 0.00/0.00 |
| :C1_c1 | 1.73 | [0.67, 9.27] | [393, 428, 409] | [408, 408, 408] | 0.00/0.00 |
| :C1_c2 | 15.53 | [5.20, 16.20] | [228, 230, 228] | [228, 228, 228] | 0.00/0.00 |
| :C1_c3 | 1.07 | [0.47, 6.00] | [363, 370, 364] | [374, 374, 374] | 0.00/0.00 |
| :C2_c0 | 16.33 | [10.80, 17.00] | [197, 197, 198] | [197, 197, 197] | 0.00/0.00 |
| :C2_c1 | 1.60 | [-0.27, 2.20] | [177, 177, 177] | [177, 177, 177] | 0.00/0.00 |
| :C2_c2 | 4.60 | [0.87, 11.40] | [204, 204, 204] | [204, 204, 204] | 0.00/0.00 |
| :C2_c3 | 4.67 | [2.00, 10.20] | [399, 365, 388] | [383, 383, 383] | 0.00/0.00 |

### mappo-guided: all crossings versus mappo-original

| Crossing | p95 delta s | Pointwise exploratory 95% interval | Model entries by replica | Reference entries by replica | Unresolved requests: model/reference |
|---|---:|---|---|---|---|
| :C0_c0 | -9.20 | [-16.40, -6.00] | [187, 187, 187] | [187, 187, 187] | 0.00/0.00 |
| :C0_c1 | -3.87 | [-9.93, 1.87] | [385, 386, 383] | [357, 383, 359] | 0.00/0.00 |
| :C0_c2 | -1.20 | [-11.40, -0.00] | [179, 178, 177] | [177, 177, 179] | 0.00/0.00 |
| :C0_c3 | -10.93 | [-14.40, -7.20] | [197, 197, 197] | [197, 197, 197] | 0.00/0.00 |
| :C1_c0 | -5.40 | [-15.20, 0.00] | [192, 193, 193] | [194, 192, 194] | 0.00/0.00 |
| :C1_c1 | -8.40 | [-10.47, -1.53] | [393, 428, 409] | [400, 396, 399] | 0.00/0.00 |
| :C1_c2 | -1.07 | [-11.53, -0.40] | [228, 230, 228] | [229, 229, 230] | 0.00/0.00 |
| :C1_c3 | -6.93 | [-9.60, -0.60] | [363, 370, 364] | [386, 366, 387] | 0.00/0.00 |
| :C2_c0 | -3.53 | [-14.20, -0.80] | [197, 197, 198] | [200, 197, 198] | 0.00/0.00 |
| :C2_c1 | -10.93 | [-15.60, -6.00] | [177, 177, 177] | [177, 177, 177] | 0.00/0.00 |
| :C2_c2 | -9.47 | [-13.20, -3.00] | [204, 204, 204] | [204, 204, 204] | 0.00/0.00 |
| :C2_c3 | -7.67 | [-9.87, -2.80] | [399, 365, 388] | [391, 397, 393] | 0.00/0.00 |

The full JSON also retains person-trip service, slower-walker groups where recorded, zero-entry runs, cutoff-only failures, whole-horizon vehicle counts, and entered-but-not-exited diagnostics where available. Missing diagnostics remain unavailable rather than zero.

## Limits

These results concern one synthetic corridor and the frozen demands/budgets. Teacher guidance, reward shaping and extra experience must be interpreted through their matched contrasts. A vehicle gain with some protected crossings does not imply fairness at other crossings, successful service of unserved users, or an improvement to deployed traffic lights.
