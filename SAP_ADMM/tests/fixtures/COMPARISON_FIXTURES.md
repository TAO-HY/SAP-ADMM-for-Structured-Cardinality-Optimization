# Comparison solver regression fixtures

`comparison_reference_outputs.npz` contains reference outputs for eight accepted SDCAM updates at a single smoothing stage for each loss and forty pADMM updates. The companion JSON records the parameters, random seed and source hashes.

Tests use these deterministic arrays to check the update formulas. Full SDCAM runs evaluate the feasible reference at each stage's current smoothing value. Convex experiment runs check all six variable blocks for stopping. These fixtures are numerical regression checks, not full experiment statistics.
