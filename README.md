# Hybrid Quantum-Classical Solver for Large-Scale Weighted Max-Cut  
## Rigetti Qvolution Hackathon 2026

---

## Overview
This repository proposes a scalable hybrid quantum-classical framework for solving large weighted Max-Cut instances under realistic NISQ hardware constraints.

Our approach combines:

- Shallow-depth QAOA on carefully selected subgraphs
- Classical refinement and local search
- Real hardware validation on Rigetti Ankaa-3

The project is organized into three major components:

1. Problem A – QAOA baseline study
2. Problem B – Hybrid quantum preconditioning strategy
3. Hardware validation – Execution on real QPU

---

# Problem A – QAOA Baseline (21 nodes)

### Objective
Evaluate QAOA performance on a small weighted Max-Cut instance where brute-force optimal value is computable.

### Implemented
- Hamiltonian construction using ZZ terms
- QAOA with p = 1 and p = 2
- Multi-start classical optimization
- Exact brute-force validation

### Result
Approximation ratio ≈ 0.58

### Insight
Shallow QAOA struggles on irregular weighted graphs.
Increasing depth improves quality but increases circuit noise.

---

# Problem B – Scalable Hybrid Strategy (180 nodes)

Direct QAOA on 180 qubits is infeasible on current NISQ devices.

### Our Strategy

1. Graph structural analysis (degree & weight distribution)
2. Extraction of a 20-node dense subgraph
3. QAOA (p=1) applied to subgraph
4. Quantum-informed partition used as preconditioning
5. Classical local improvement on full graph

---

### Final Result (Problem B)

Total graph weight: 7465.707  
Final cut: 6538.127  

MPES Ratio:

0.8758

This result is stable across multiple random seeds.

---

# Hardware Validation – Rigetti Ankaa-3

To evaluate real NISQ performance, we executed the 20-qubit QAOA instance on Rigetti's Ankaa-3 QPU.

### Circuit Details
- 20 qubits
- 24 weighted ZZ interactions
- ~48 CNOT gates
- p = 1
- Mixer RX layer

### Shot Scaling Study

| Shots  | Energy |
|---------|----------|
| 1000    | -5.51    |
| 5000    | -4.72    |
| 10000   | -13.90   |

### Observations

- Energy fluctuations persist with increased shots.
- Readout error is moderate (1–4%).
- Two-qubit gate noise dominates performance degradation.

### Conclusion

Our experiment demonstrates the practical limits of deep QAOA circuits on current NISQ hardware and justifies hybrid approaches.

---

# Repository Structure

problem_A/ QAOA baseline implementation
problem_B/ Hybrid quantum preconditioning solver
hardware/ Real QPU execution code and analysis
classical/ Classical benchmarking implementations

---

# How to Run

Each notebook is self-contained and can be executed independently.

Required packages:
- Qiskit
- PyQuil
- NumPy
- SciPy
- NetworkX
- Pandas

Hardware execution requires Rigetti QPU access credentials.

---

# Team Contributions

- Classical benchmarking and refinement

---

# Key Takeaway

Instead of scaling QAOA depth beyond hardware limits, we demonstrate that:

Hybrid quantum preconditioning + classical refinement  
provides scalable performance while remaining hardware-aware.

