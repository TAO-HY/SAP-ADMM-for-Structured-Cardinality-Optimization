# Convex solver regression fixtures

`l1_reference_outputs.npz` contains reference outputs for deterministic signal and image inputs. The companion JSON records the input seed, parameters, update counts and source hashes.

Tests use these arrays to check convex solver updates. The signal fixture includes the auxiliary jump vector; the image fixture includes the clipped restoration. These files support numerical regression checks and are not benchmark measurements.
