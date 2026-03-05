# Hardware Validation – Rigetti Ankaa-3 QPU

## Overview

This section documents the real quantum hardware execution of our QAOA circuit for the weighted Max-Cut problem.

We executed a 20-qubit QAOA instance on Rigetti's Ankaa-3 QPU to experimentally evaluate NISQ performance limitations.

---

## Circuit Configuration

- Platform: Rigetti Ankaa-3
- Qubits used: 20
- QAOA depth: p = 1
- Problem type: Weighted Max-Cut (subgraph of 180-node graph)
- ZZ terms: 24 weighted edges
- Approximate two-qubit gates: ~48 CNOT
- Mixer: RX layer
- Initialization: |+> state

Optimized parameters (from classical simulation):

- γ = 3.20712571
- β = 5.37330497

---

## Execution Strategy

To distinguish between sampling noise and gate noise, we performed a shot-scaling experiment.

The same compiled circuit was executed with:

- 1,000 shots
- 5,000 shots
- 10,000 shots

---

## Measured Energies

| Shots  | Energy |
|---------|----------|
| 1000    | -5.51    |
| 5000    | -4.72    |
| 10000   | -13.90   |

---

## Readout Calibration

Single-qubit readout characterization revealed:

- P(1|0) ≈ 1–4%
- P(1|1) ≈ 88–98%

Measurement error was moderate and not the dominant error source.

---

## Observations

- Energy does not monotonically improve with increased shots.
- Significant fluctuations persist at higher shot counts.
- This indicates gate noise (particularly two-qubit infidelity) dominates over sampling noise.

---

## Conclusion

Direct large-scale QAOA on current NISQ hardware is limited by accumulated two-qubit errors.

However, this experiment validates:

- Successful hardware compilation
- Real QPU execution
- Empirical noise characterization
- Hardware-aware algorithm design

Our results support the use of hybrid quantum-classical strategies for scalable optimization.