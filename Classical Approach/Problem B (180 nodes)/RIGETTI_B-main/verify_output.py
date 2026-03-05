import os, json, argparse
import numpy as np
import pandas as pd
from numba import njit

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
    # fallback
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_csv(path)

def to_csr(df: pd.DataFrame, n: int):
    a, b, wcol = infer_cols(df)
    adj = [[] for _ in range(n)]
    for r in df.itertuples(index=False):
        u = int(getattr(r, a))
        v = int(getattr(r, b))
        w = float(getattr(r, wcol))
        adj[u].append((v, w))
        adj[v].append((u, w))

    idxs = []
    wts = []
    ptrs = [0]
    for u in range(n):
        for v, w in adj[u]:
            idxs.append(v)
            wts.append(w)
        ptrs.append(len(idxs))

    return (
        np.asarray(idxs, dtype=np.int64),
        np.asarray(ptrs, dtype=np.int64),
        np.asarray(wts, dtype=np.float64),
        float(df[wcol].sum()),
    )

@njit
def cut_value(part, adj_indices, adj_ptrs, adj_weights):
    total = 0.0
    n = len(part)
    for u in range(n):
        pu = part[u]
        for k in range(adj_ptrs[u], adj_ptrs[u+1]):
            v = adj_indices[k]
            if u < v and pu != part[v]:
                total += adj_weights[k]
    return total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="problemb.parquet")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    out_json = os.path.join(args.outdir, "loopei_result_B.json")
    out_part = os.path.join(args.outdir, "partition.csv")

    info = json.load(open(out_json, "r", encoding="utf-8"))
    dfp = pd.read_csv(out_part)
    part = dfp["side"].to_numpy(dtype=np.int64)

    df = load_df(args.data)
    a, b, wcol = infer_cols(df)
    n = int(max(df[a].max(), df[b].max()) + 1)

    if len(part) != n:
        raise SystemExit(f"Partition length {len(part)} != N {n}")

    csr = to_csr(df, n)
    c = float(cut_value(part, csr[0], csr[1], csr[2]))
    total_w = float(csr[3])
    ratio = c / total_w

    print("=== VERIFY ===")
    print(f"cut(json)   = {info['cut']:.6f}")
    print(f"cut(recalc) = {c:.6f}")
    print(f"ratio(json) = {info['cut_ratio']:.9f}")
    print(f"ratio(calc) = {ratio:.9f}")

    dc = abs(c - float(info["cut"]))
    dr = abs(ratio - float(info["cut_ratio"]))
    print(f"abs diff cut   = {dc:.6e}")
    print(f"abs diff ratio = {dr:.6e}")

    if dc > 1e-6 or dr > 1e-9:
        raise SystemExit("VERIFY FAILED (mismatch beyond tolerance).")
    print("VERIFY PASSED.")

if __name__ == "__main__":
    main()