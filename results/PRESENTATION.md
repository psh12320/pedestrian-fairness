# CS4246 Team 38 — presentation outline

## 1. Research question
Can multi-agent control improve vehicle flow without imposing long waits at specific crossings?

## 2. Experimental setup
New synthetic three-junction SUMO corridor; six controllers; matched held-out demand and seeds.

## 3. Safe action boundary
Show EW, NS and all-walk states with minimum green, yellow, all-red and occupancy-extended clearance.

## 4. Measurement
Explain crossing approach/entry event timestamps, per-crossing p95 and unserved requests; show vehicle trip accounting.

## 5. IPPO versus MAPPO
Local actors in both methods. IPPO local critic; MAPPO training critic sees the corridor. Vehicle and pedestrian rewards give four learned controllers.

## 6. Main comparison
Use the medium-demand table and `tradeoff.svg` from `REPORT.md`; read service counts beside time loss.

## 7. Stress tests
Use the demand grid, slow-walker table and worst-crossing table; explain seed uncertainty.

## 8. Conclusion
State only what the simulated paired results support. Highlight limits of the synthetic network, linear policies and five evaluation seeds.
