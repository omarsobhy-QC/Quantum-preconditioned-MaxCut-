# ============================================================
# LoopEi Max-Cut — Problem B (180) — DO-ALL + FIGURES + COMPARISONS
#
# Produces submission bundle:
#   outputs/loopei_result_B.json
#   outputs/partition.csv
#   outputs/convergence.csv
#   outputs/summary_report.md
#   outputs/publication_figure.png
#
# PLUS plots + comparisons:
#   outputs/comparisons.csv
#   outputs/comparisons.json
#   outputs/figure_baseline_comparison.png
#   outputs/figure_weight_hist.png
#   outputs/figure_degree_hist.png
#   outputs/figure_cut_edge_weight_hist.png
#
# Engine:
#   - multi-start tabu on sqrt-smoothed weights
#   - smoothing schedule sqrt -> w^0.75 -> full
#   - elite consensus recombination
#   - guided path relinking
#   - sampled 2-flip micro-LNS
#   - adaptive stop
#
# Baselines:
#   - Random (best-of-K)
#   - Greedy 1-flip hillclimb (multi-start)
# ============================================================

import os
import sys
import time
import json
import platform
import argparse
import numpy as np
import pandas as pd
from numba import njit
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import matplotlib.pyplot as plt

# ----------------------------
# Data loading
# ----------------------------

def infer_cols(df: pd.DataFrame):
    cols = list(df.columns)
    if {"node_1", "node_2", "weight"}.issubset(cols):
        return "node_1", "node_2", "weight"
    return cols[0], cols[1], cols[2]

def load_df(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    # auto try parquet then csv
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_csv(path)
    raise FileNotFoundError(f"Data file not found: {path}")

def compute_n_m_totalw(df: pd.DataFrame):
    a, b, wcol = infer_cols(df)
    n = int(max(df[a].max(), df[b].max()) + 1)
    m = int(len(df))
    total_w = float(df[wcol].sum())
    return n, m, total_w, a, b, wcol

def to_csr(df: pd.DataFrame, n: int, power: float):
    a, b, wcol = infer_cols(df)
    adj = [[] for _ in range(n)]
    for r in df.itertuples(index=False):
        u = int(getattr(r, a))
        v = int(getattr(r, b))
        w = float(getattr(r, wcol))
        tw = w ** power
        adj[u].append((v, tw))
        adj[v].append((u, tw))

    indices = []
    weights = []
    ptrs = [0]
    for u in range(n):
        for v, w in adj[u]:
            indices.append(v)
            weights.append(w)
        ptrs.append(len(indices))

    return (
        np.asarray(indices, dtype=np.int64),
        np.asarray(ptrs, dtype=np.int64),
        np.asarray(weights, dtype=np.float64),
    )

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# ----------------------------
# NUMBA core
# ----------------------------

@njit
def calculate_full_cut(part, adj_indices, adj_ptrs, adj_weights):
    total = 0.0
    n = len(part)
    for u in range(n):
        pu = part[u]
        for idx in range(adj_ptrs[u], adj_ptrs[u + 1]):
            v = adj_indices[idx]
            if u < v and pu != part[v]:
                total += adj_weights[idx]
    return total

@njit
def _init_gains(part, adj_indices, adj_ptrs, adj_weights):
    n = len(part)
    gains = np.zeros(n, dtype=np.float64)
    for u in range(n):
        pu = part[u]
        s = 0.0
        for idx in range(adj_ptrs[u], adj_ptrs[u + 1]):
            v = adj_indices[idx]
            w = adj_weights[idx]
            if pu == part[v]:
                s += w
            else:
                s -= w
        gains[u] = s
    return gains

@njit
def _lcg_next(x):
    return (1103515245 * x + 12345) & 0x7fffffff

@njit
def fast_tabu_kernel(part, iters, stall_limit,
                     adj_indices, adj_ptrs, adj_weights,
                     base_tenure, seed):
    """
    1-flip tabu with aspiration + adaptive stop.
    No Python random inside @njit.
    """
    n = len(part)
    gains = _init_gains(part, adj_indices, adj_ptrs, adj_weights)
    tabu_until = np.zeros(n, dtype=np.int64)

    curr_cut = calculate_full_cut(part, adj_indices, adj_ptrs, adj_weights)
    best_cut = curr_cut
    best_part = part.copy()
    last_improve = 0

    rng = seed

    for i in range(iters):
        if i - last_improve > stall_limit:
            break

        best_move = -1
        max_gain = -1e18

        for u in range(n):
            g = gains[u]
            if tabu_until[u] <= i:
                if g > max_gain:
                    max_gain = g
                    best_move = u
            else:
                if curr_cut + g > best_cut + 1e-9:
                    best_move = u
                    max_gain = g
                    break

        if best_move == -1:
            break

        part[best_move] ^= 1
        curr_cut += gains[best_move]

        pb = part[best_move]
        for idx in range(adj_ptrs[best_move], adj_ptrs[best_move + 1]):
            v = adj_indices[idx]
            w = adj_weights[idx]
            if pb == part[v]:
                gains[v] += 2.0 * w
            else:
                gains[v] -= 2.0 * w

        gains[best_move] = -gains[best_move]

        rng = _lcg_next(rng ^ (best_move + 17))
        jitter = 1 + (rng % 11)  # 1..11
        tabu_until[best_move] = i + base_tenure + jitter

        if curr_cut > best_cut + 1e-12:
            best_cut = curr_cut
            best_part = part.copy()
            last_improve = i

    return best_part, best_cut

@njit
def path_relink_guided(a_part, b_part, max_steps,
                       adj_indices, adj_ptrs, adj_weights):
    """
    Guided relinking from A toward B:
    flip one differing node per step with best immediate gain.
    Returns best solution along the path.
    """
    part = a_part.copy()
    n = len(part)
    gains = _init_gains(part, adj_indices, adj_ptrs, adj_weights)

    curr_cut = calculate_full_cut(part, adj_indices, adj_ptrs, adj_weights)
    best_cut = curr_cut
    best_part = part.copy()

    for _ in range(max_steps):
        best_move = -1
        best_gain = -1e18

        for u in range(n):
            if part[u] != b_part[u]:
                g = gains[u]
                if g > best_gain:
                    best_gain = g
                    best_move = u

        if best_move == -1:
            break

        part[best_move] ^= 1
        curr_cut += gains[best_move]

        pb = part[best_move]
        for idx in range(adj_ptrs[best_move], adj_ptrs[best_move + 1]):
            v = adj_indices[idx]
            w = adj_weights[idx]
            if pb == part[v]:
                gains[v] += 2.0 * w
            else:
                gains[v] -= 2.0 * w

        gains[best_move] = -gains[best_move]

        if curr_cut > best_cut + 1e-12:
            best_cut = curr_cut
            best_part = part.copy()

    return best_part, best_cut

@njit
def micro_2flip_sample(part, topk, pair_samples, seed,
                       adj_indices, adj_ptrs, adj_weights):
    """
    Sampled 2-flip improvement among top-K single-flip candidates.
    """
    n = len(part)
    gains = _init_gains(part, adj_indices, adj_ptrs, adj_weights)

    cand = np.empty(topk, dtype=np.int64)
    used = np.zeros(n, dtype=np.uint8)

    for i in range(topk):
        best_u = -1
        best_g = -1e18
        for u in range(n):
            if used[u] == 0:
                g = gains[u]
                if g > best_g:
                    best_g = g
                    best_u = u
        cand[i] = best_u
        used[best_u] = 1

    best_cut = calculate_full_cut(part, adj_indices, adj_ptrs, adj_weights)
    best_part = part.copy()

    rng = seed
    for _ in range(pair_samples):
        rng = _lcg_next(rng + 101)
        i = rng % topk
        rng = _lcg_next(rng + 313)
        j = rng % topk
        if i == j:
            continue
        u = cand[i]
        v = cand[j]
        if u == v:
            continue

        tmp = part.copy()
        tmp[u] ^= 1
        tmp[v] ^= 1
        c = calculate_full_cut(tmp, adj_indices, adj_ptrs, adj_weights)
        if c > best_cut + 1e-12:
            best_cut = c
            best_part = tmp

    return best_part, best_cut

# ----------------------------
# Baselines (non-numba for clarity; still fast at N=180)
# ----------------------------

def baseline_random_bestof(N: int, csr_full, total_w: float, seed: int, trials: int):
    rs = np.random.RandomState(seed)
    best_cut = -1.0
    t0 = time.time()
    for _ in range(trials):
        part = rs.randint(0, 2, N).astype(np.int64)
        c = float(calculate_full_cut(part, *csr_full))
        if c > best_cut:
            best_cut = c
    rt = time.time() - t0
    return {"name": f"Random best-of-{trials}", "cut": best_cut, "ratio": best_cut / total_w, "runtime_sec": rt}

def baseline_greedy_hillclimb(N: int, csr_full, total_w: float, seed: int, starts: int, max_passes: int):
    """
    Greedy 1-flip hillclimb (multi-start).
    """
    rs = np.random.RandomState(seed)
    t0 = time.time()
    best_cut = -1.0

    adj_indices, adj_ptrs, adj_weights = csr_full

    for _ in range(starts):
        part = rs.randint(0, 2, N).astype(np.int64)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            gains = _init_gains(part, adj_indices, adj_ptrs, adj_weights)  # numba
            while True:
                u = int(np.argmax(gains))
                g = float(gains[u])
                if g <= 1e-12:
                    break
                part[u] ^= 1
                pb = part[u]
                for idx in range(adj_ptrs[u], adj_ptrs[u + 1]):
                    v = int(adj_indices[idx])
                    w = float(adj_weights[idx])
                    if pb == int(part[v]):
                        gains[v] += 2.0 * w
                    else:
                        gains[v] -= 2.0 * w
                gains[u] = -gains[u]
                improved = True

        c = float(calculate_full_cut(part, *csr_full))
        if c > best_cut:
            best_cut = c

    rt = time.time() - t0
    return {"name": f"Greedy hillclimb ({starts} starts)", "cut": best_cut, "ratio": best_cut / total_w, "runtime_sec": rt}

# ----------------------------
# Output writers
# ----------------------------

def save_partition_csv(path: str, part: np.ndarray):
    df = pd.DataFrame({"node": np.arange(len(part), dtype=int), "side": part.astype(int)})
    df.to_csv(path, index=False)

def save_convergence_csv(path: str, rows):
    df = pd.DataFrame(rows, columns=["cycle", "best_cut", "best_ratio"])
    df.to_csv(path, index=False)

def write_summary_report(path: str, info: dict, comparisons: list):
    lines = []
    lines.append("# LoopEi Problem B — Summary Report\n")
    lines.append(f"- Nodes (N): **{info['N']}**")
    lines.append(f"- Edges (M): **{info['M']}**")
    lines.append(f"- Total weight: **{info['total_weight']:.6f}**\n")
    lines.append("## Result")
    lines.append(f"- Cut: **{info['cut']:.6f}**")
    lines.append(f"- MPES: **{info['cut_ratio']:.6f}**")
    lines.append(f"- Runtime (sec): **{info['runtime_sec']:.2f}**")
    lines.append(f"- Seed: **{info['seed']}**\n")

    lines.append("## Comparisons (baselines)\n")
    lines.append("| Method | MPES | Cut | Runtime (s) |")
    lines.append("|---|---:|---:|---:|")
    for r in comparisons:
        lines.append(f"| {r['name']} | {r['ratio']:.6f} | {r['cut']:.3f} | {r['runtime_sec']:.2f} |")
    lines.append("")

    lines.append("## Method flags")
    mf = info["meta"]["method_flags"]
    lines.append(f"- Dynamic smoothing schedule: **{mf['dynamic_smoothing']}**")
    lines.append(f"- Elite consensus recombination: **{mf['elite_consensus']}**")
    lines.append(f"- Guided path relinking: **{mf['path_relinking']}**")
    lines.append(f"- 2-flip micro-LNS (sampled): **{mf['micro_lns_2flip_sampled']}**")
    lines.append(f"- Adaptive stop: **{mf['adaptive_stop']}**")
    lines.append(f"- Baselines included: **{mf['baselines_included']}**")
    lines.append(f"- Figures included: **{mf['figures_included']}**\n")

    lines.append("## Environment")
    env = info["meta"]["env"]
    lines.append(f"- Python: `{env['python']}`")
    lines.append(f"- Platform: `{env['platform']}`")
    lines.append(f"- CPU cores (reported): `{env['cpu_count']}`")
    lines.append(f"- Numba: `{env['numba']}`")
    lines.append(f"- Numpy: `{env['numpy']}`")
    lines.append(f"- Pandas: `{env['pandas']}`\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def make_convergence_figure(path: str, conv_rows):
    cycles = [int(r[0]) for r in conv_rows]
    ratios = [float(r[2]) for r in conv_rows]
    plt.figure(figsize=(9, 5))
    plt.plot(cycles, ratios, marker="o", linewidth=1.5)
    plt.xlabel("Cycle")
    plt.ylabel("Best MPES (cut / total weight)")
    plt.title("Problem B — Convergence (Best MPES)")
    plt.grid(True, linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def make_baseline_comparison_figure(path: str, rows):
    names = [r["name"] for r in rows]
    ratios = [float(r["ratio"]) for r in rows]
    plt.figure(figsize=(10, 5))
    plt.bar(names, ratios)
    plt.ylabel("MPES (cut / total weight)")
    plt.title("Problem B — Baseline Comparison (MPES)")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def make_weight_hist(path: str, weights: np.ndarray):
    plt.figure(figsize=(9, 5))
    plt.hist(weights, bins=40)
    plt.xlabel("Edge weight")
    plt.ylabel("Count")
    plt.title("Problem B — Edge Weight Distribution")
    plt.grid(True, linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def make_degree_hist(path: str, degrees: np.ndarray):
    plt.figure(figsize=(9, 5))
    plt.hist(degrees, bins=30)
    plt.xlabel("Node degree")
    plt.ylabel("Count")
    plt.title("Problem B — Node Degree Distribution")
    plt.grid(True, linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def make_cut_edge_weight_hist(path: str, df: pd.DataFrame, part: np.ndarray):
    a, b, wcol = infer_cols(df)
    u = df[a].to_numpy(dtype=np.int64)
    v = df[b].to_numpy(dtype=np.int64)
    w = df[wcol].to_numpy(dtype=np.float64)
    mask = part[u] != part[v]
    cut_w = w[mask]
    plt.figure(figsize=(9, 5))
    plt.hist(cut_w, bins=40)
    plt.xlabel("Edge weight (only cut edges)")
    plt.ylabel("Count")
    plt.title("Problem B — Distribution of Cut Edge Weights")
    plt.grid(True, linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

# ----------------------------
# Worker for parallel starts
# ----------------------------

def worker_run(seed, n, csr_indices, csr_ptrs, csr_weights,
               iters, stall, base_tenure):
    rs = np.random.RandomState(seed)
    part0 = rs.randint(0, 2, n).astype(np.int64)
    best_p, best_c = fast_tabu_kernel(part0, iters, stall,
                                      csr_indices, csr_ptrs, csr_weights,
                                      base_tenure, int(seed))
    return best_p, float(best_c)

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="problemb.parquet", help="problemb.parquet or problemb.csv")
    ap.add_argument("--outdir", default="outputs", help="Output directory")
    ap.add_argument("--seed", type=int, default=7, help="Global seed for determinism")
    ap.add_argument("--starts", type=int, default=12, help="Parallel starts (capped by CPU count)")
    ap.add_argument("--cycles", type=int, default=22, help="Elite cycles")
    ap.add_argument("--baseline_random_trials", type=int, default=2000, help="Random best-of-K trials")
    ap.add_argument("--baseline_greedy_starts", type=int, default=24, help="Greedy hillclimb multi-starts")
    ap.add_argument("--baseline_greedy_passes", type=int, default=40, help="Max passes per greedy start")
    args = ap.parse_args()

    np.random.seed(args.seed)

    df = load_df(args.data)
    N, M, TOTAL_W, a, b, wcol = compute_n_m_totalw(df)

    csr_sqrt = to_csr(df, N, power=0.5)
    csr_mid  = to_csr(df, N, power=0.75)
    csr_full = to_csr(df, N, power=1.0)

    ensure_dir(args.outdir)

    out_json = os.path.join(args.outdir, "loopei_result_B.json")
    out_part = os.path.join(args.outdir, "partition.csv")
    out_conv = os.path.join(args.outdir, "convergence.csv")
    out_rep  = os.path.join(args.outdir, "summary_report.md")

    fig_conv = os.path.join(args.outdir, "publication_figure.png")
    fig_base = os.path.join(args.outdir, "figure_baseline_comparison.png")
    fig_w    = os.path.join(args.outdir, "figure_weight_hist.png")
    fig_deg  = os.path.join(args.outdir, "figure_degree_hist.png")
    fig_cutw = os.path.join(args.outdir, "figure_cut_edge_weight_hist.png")

    out_comp_csv = os.path.join(args.outdir, "comparisons.csv")
    out_comp_json = os.path.join(args.outdir, "comparisons.json")

    # budgets
    IT_SMOOTH = 550_000
    IT_MID = 550_000
    IT_FULL = 700_000
    STALL_SMOOTH = 45_000
    STALL_MID = 50_000
    STALL_FULL = 60_000

    POP_KEEP = 12
    ELITES = 4
    RELINK_STEPS = 60

    MICRO_TOPK = 34
    MICRO_PAIR_SAMPLES = 900

    starts = min(int(args.starts), max(1, os.cpu_count() or 1))
    cycles = int(args.cycles)

    print(f"--- LoopEi Problem B DO-ALL+FIGURES (N={N}) ---")
    print(f"Edges={M} TotalW={TOTAL_W:.6f}")
    t0 = time.time()

    # warmup compile
    _ = calculate_full_cut(np.zeros(N, dtype=np.int64), *csr_full)
    ptmp = np.zeros(N, dtype=np.int64)
    _ = fast_tabu_kernel(ptmp, 10, 5, *csr_sqrt, 12, args.seed)

    # ----------------------------
    # Baselines
    # ----------------------------
    print("[Baselines] Running random + greedy hillclimb")
    base_rows = []
    base_rows.append(baseline_random_bestof(
        N, csr_full, TOTAL_W, seed=args.seed + 11, trials=int(args.baseline_random_trials)
    ))
    base_rows.append(baseline_greedy_hillclimb(
        N, csr_full, TOTAL_W, seed=args.seed + 23,
        starts=int(args.baseline_greedy_starts), max_passes=int(args.baseline_greedy_passes)
    ))

    # ----------------------------
    # Phase 1: parallel init on sqrt-smoothed weights
    # ----------------------------
    print("[Phase 1] Parallel init on sqrt-smoothed weights")
    rs = np.random.RandomState(args.seed)
    seeds = rs.randint(1, 2_000_000_000, size=starts).tolist()

    base_tenure_s = int(8 + 0.15 * N)

    population = []
    with ProcessPoolExecutor(max_workers=starts) as ex:
        futs = []
        for sd in seeds:
            futs.append(ex.submit(worker_run, int(sd), N,
                                  csr_sqrt[0], csr_sqrt[1], csr_sqrt[2],
                                  IT_SMOOTH, STALL_SMOOTH, base_tenure_s))
        for f in futs:
            p_s, _ = f.result()
            c_full = float(calculate_full_cut(p_s, *csr_full))
            population.append((p_s, c_full))

    population.sort(key=lambda x: x[1], reverse=True)
    population = population[:POP_KEEP]
    best_part = population[0][0].copy()
    best_cut = float(population[0][1])

    convergence_rows = [(0, best_cut, best_cut / TOTAL_W)]
    print(f"Init best MPES={best_cut/TOTAL_W:.6f}")

    # ----------------------------
    # Elite cycles
    # ----------------------------
    for cycle in range(cycles):
        if cycle < 6:
            csr = csr_mid
            iters = IT_MID
            stall = STALL_MID
            base_tenure = int(9 + 0.16 * N)
        else:
            csr = csr_full
            iters = IT_FULL
            stall = STALL_FULL
            base_tenure = int(10 + 0.18 * N)

        elites = [p for (p, _) in population[:min(ELITES, len(population))]]
        if len(elites) < 2:
            elites = [best_part.copy(), best_part.copy()]

        stack = np.stack(elites, axis=0)
        consensus = np.sum(stack, axis=0)
        unstable = np.where((consensus != 0) & (consensus != len(elites)))[0]

        rng = np.random.RandomState(args.seed + 1000 + cycle)

        child = best_part.copy()
        if unstable.size > 0:
            flips = rng.rand(unstable.size) > 0.5
            child[unstable[flips]] ^= 1

        pr_part, _ = path_relink_guided(elites[0], elites[1], RELINK_STEPS, *csr)

        merged = child.copy()
        diff = np.where(child != pr_part)[0]
        if diff.size > 0:
            pick = rng.rand(diff.size) < 0.35
            merged[diff[pick]] ^= 1

        seed_pol = int(rng.randint(1, 2_000_000_000))
        pol_part, _ = fast_tabu_kernel(merged.astype(np.int64), iters, stall,
                                       *csr, base_tenure, seed_pol)
        pol_cut_full = float(calculate_full_cut(pol_part, *csr_full))

        seed_micro = int(rng.randint(1, 2_000_000_000))
        micro_part, micro_cut = micro_2flip_sample(
            pol_part.astype(np.int64), MICRO_TOPK, MICRO_PAIR_SAMPLES, seed_micro, *csr_full
        )
        micro_cut = float(micro_cut)

        cand_part = micro_part
        cand_cut = micro_cut
        if pol_cut_full > cand_cut:
            cand_part = pol_part
            cand_cut = pol_cut_full

        if cand_cut > best_cut + 1e-12:
            best_cut = cand_cut
            best_part = cand_part.copy()
            print(f"Cycle {cycle:02d} IMPROVE MPES={best_cut/TOTAL_W:.6f}")
        elif cycle % 4 == 0:
            print(f"Cycle {cycle:02d} best MPES={best_cut/TOTAL_W:.6f}")

        population.append((cand_part.copy(), cand_cut))
        population.sort(key=lambda x: x[1], reverse=True)
        population = population[:POP_KEEP]

        convergence_rows.append((cycle + 1, best_cut, best_cut / TOTAL_W))

    runtime = time.time() - t0
    ratio = best_cut / TOTAL_W

    print(f"\nFinal MPES={ratio:.6f} Cut={best_cut:.6f} Time={runtime:.2f}s")

    # ----------------------------
    # Save core outputs
    # ----------------------------
    save_partition_csv(out_part, best_part)
    save_convergence_csv(out_conv, convergence_rows)

    # ----------------------------
    # Comparisons + figures
    # ----------------------------
    loopei_row = {
        "name": "LoopEi (Tabu+Schedule+Relink+MicroLNS)",
        "cut": float(best_cut),
        "ratio": float(ratio),
        "runtime_sec": float(runtime),
    }
    comp_rows = base_rows + [loopei_row]

    pd.DataFrame(comp_rows).to_csv(out_comp_csv, index=False)
    with open(out_comp_json, "w", encoding="utf-8") as f:
        json.dump({"rows": comp_rows}, f, indent=2)

    make_convergence_figure(fig_conv, convergence_rows)
    make_baseline_comparison_figure(fig_base, comp_rows)

    weights = df[wcol].to_numpy(dtype=np.float64)
    make_weight_hist(fig_w, weights)

    deg = np.zeros(N, dtype=np.int64)
    uarr = df[a].to_numpy(dtype=np.int64)
    varr = df[b].to_numpy(dtype=np.int64)
    for u, v in zip(uarr, varr):
        deg[u] += 1
        deg[v] += 1
    make_degree_hist(fig_deg, deg)

    make_cut_edge_weight_hist(fig_cutw, df, best_part)

    # ----------------------------
    # JSON bundle
    # ----------------------------
    try:
        import numba
        numba_ver = getattr(numba, "__version__", "unknown")
    except Exception:
        numba_ver = "unknown"

    info = {
        "cut": float(best_cut),
        "cut_ratio": float(ratio),
        "total_weight": float(TOTAL_W),
        "runtime_sec": float(runtime),
        "partition": [int(x) for x in best_part.tolist()],
        "meta": {
            "dataset": {"path": str(args.data), "columns": {"u": a, "v": b, "w": wcol}},
            "config": {
                "seed": int(args.seed),
                "starts": int(starts),
                "cycles": int(cycles),
                "pop_keep": int(POP_KEEP),
                "elites": int(ELITES),
                "relink_steps": int(RELINK_STEPS),
                "micro_topk": int(MICRO_TOPK),
                "micro_pair_samples": int(MICRO_PAIR_SAMPLES),
                "iters": {"smooth": IT_SMOOTH, "mid": IT_MID, "full": IT_FULL},
                "stall": {"smooth": STALL_SMOOTH, "mid": STALL_MID, "full": STALL_FULL},
                "tenure_base": {
                    "smooth": int(8 + 0.15 * N),
                    "mid": int(9 + 0.16 * N),
                    "full": int(10 + 0.18 * N),
                },
                "baselines": {
                    "random_trials": int(args.baseline_random_trials),
                    "greedy_starts": int(args.baseline_greedy_starts),
                    "greedy_passes": int(args.baseline_greedy_passes),
                },
            },
            "method_flags": {
                "dynamic_smoothing": True,
                "elite_consensus": True,
                "path_relinking": True,
                "micro_lns_2flip_sampled": True,
                "adaptive_stop": True,
                "baselines_included": True,
                "figures_included": True,
            },
            "env": {
                "python": sys.version.replace("\n", " "),
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "numba": numba_ver,
            },
            "outputs": {
                "json": os.path.basename(out_json),
                "partition_csv": os.path.basename(out_part),
                "convergence_csv": os.path.basename(out_conv),
                "comparisons_csv": os.path.basename(out_comp_csv),
                "comparisons_json": os.path.basename(out_comp_json),
                "figures": [
                    os.path.basename(fig_conv),
                    os.path.basename(fig_base),
                    os.path.basename(fig_w),
                    os.path.basename(fig_deg),
                    os.path.basename(fig_cutw),
                ],
            },
        },
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    rep_info = dict(info)
    rep_info["N"] = int(N)
    rep_info["M"] = int(M)
    rep_info["seed"] = int(args.seed)
    write_summary_report(out_rep, rep_info, comp_rows)

    print("Saved outputs:")
    for p in [
        out_json, out_part, out_conv, out_rep,
        out_comp_csv, out_comp_json,
        fig_conv, fig_base, fig_w, fig_deg, fig_cutw
    ]:
        print("-", p)

if __name__ == "__main__":
    mp.freeze_support()
    main()