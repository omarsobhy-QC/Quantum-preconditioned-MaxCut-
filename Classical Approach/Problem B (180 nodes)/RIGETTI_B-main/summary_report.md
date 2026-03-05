# LoopEi Problem B — Summary Report

- Nodes (N): **180**
- Edges (M): **226**
- Total weight: **7465.707486**

## Result
- Cut: **7099.571726**
- MPES: **0.950958**
- Runtime (sec): **16.58**
- Seed: **7**

## Comparisons (baselines)

| Method | MPES | Cut | Runtime (s) |
|---|---:|---:|---:|
| Random best-of-2000 | 0.623202 | 4652.644 | 0.02 |
| Greedy hillclimb (24 starts) | 0.901841 | 6732.877 | 0.00 |
| LoopEi (Tabu+Schedule+Relink+MicroLNS) | 0.950958 | 7099.572 | 16.58 |

## Method flags
- Dynamic smoothing schedule: **True**
- Elite consensus recombination: **True**
- Guided path relinking: **True**
- 2-flip micro-LNS (sampled): **True**
- Adaptive stop: **True**
- Baselines included: **True**
- Figures included: **True**

## Environment
- Python: `3.13.7 (tags/v3.13.7:bcee1c3, Aug 14 2025, 14:15:11) [MSC v.1944 64 bit (AMD64)]`
- Platform: `Windows-11-10.0.26200-SP0`
- CPU cores (reported): `8`
- Numba: `0.62.1`
- Numpy: `2.3.3`
- Pandas: `3.0.1`

