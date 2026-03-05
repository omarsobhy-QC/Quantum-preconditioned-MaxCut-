import os
import sys
import json
import random
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ==============================
# DATA LOADING
# ==============================

def load_graph(path):
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        raise ValueError("Unsupported file format")

    G = nx.Graph()
    for _, row in df.iterrows():
        u, v, w = int(row[0]), int(row[1]), float(row[2])
        G.add_edge(u, v, weight=w)

    total_weight = sum(d["weight"] for _, _, d in G.edges(data=True))
    return G, total_weight

# ==============================
# BASELINE
# ==============================

def random_cut(G):
    partition = {n: random.choice([0, 1]) for n in G.nodes}
    cut = sum(
        d["weight"]
        for u, v, d in G.edges(data=True)
        if partition[u] != partition[v]
    )
    return cut, partition

# ==============================
# LOCAL REFINEMENT
# ==============================

def improve_cut(G, partition):
    improved = True
    while improved:
        improved = False
        for node in G.nodes:
            current = partition[node]
            partition[node] = 1 - current
            new_cut = cut_value(G, partition)
            partition[node] = current
            old_cut = cut_value(G, partition)

            if new_cut > old_cut:
                partition[node] = 1 - current
                improved = True
    return partition

def cut_value(G, partition):
    return sum(
        d["weight"]
        for u, v, d in G.edges(data=True)
        if partition[u] != partition[v]
    )

# ==============================
# BASIN ESCAPE SEARCH
# ==============================

def basin_escape(G, restarts=6, trials=4000):
    best_cut = 0
    best_part = None

    for r in range(restarts):
        part = {n: random.choice([0, 1]) for n in G.nodes}

        for t in range(trials):
            node = random.choice(list(G.nodes))
            part[node] ^= 1

            if t % 500 == 0:
                part = improve_cut(G, part)

            current_cut = cut_value(G, part)

            if current_cut > best_cut:
                best_cut = current_cut
                best_part = part.copy()

        print(f"restart {r} | best_cut {best_cut:.2f}")

    return best_cut, best_part

# ==============================
# VISUALIZATION
# ==============================

def save_plot(G, partition, path):
    pos = nx.spring_layout(G, seed=42)
    colors = ["red" if partition[n] == 0 else "blue" for n in G.nodes]

    plt.figure(figsize=(6, 4))
    nx.draw(G, pos, node_color=colors, with_labels=True)
    plt.savefig(path)
    plt.close()

# ==============================
# MAIN
# ==============================

def main(dataset_path):
    os.makedirs("results", exist_ok=True)

    print(f"\nUsing dataset: {dataset_path}\n")

    G, total_weight = load_graph(dataset_path)
    print(f"Nodes: {len(G.nodes)} | Edges: {len(G.edges)}")
    print(f"Total weight: {total_weight:.2f}\n")

    print("Running baseline...")
    baseline, _ = random_cut(G)
    print(f"Baseline: {baseline:.2f}\n")

    print("Running refinement...")
    best_cut, best_partition = basin_escape(G)

    efficiency = best_cut / total_weight

    print("\n=== FINAL ===")
    print(f"Best Cut: {best_cut:.2f}")
    print(f"Efficiency: {efficiency:.2%}")

    # Save results
    results = {
        "best_cut": best_cut,
        "efficiency": efficiency,
        "total_weight": total_weight,
    }

    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=2)

    save_plot(G, best_partition, "results/partition.png")

    print("Results saved to /results\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    else:
        dataset = "data/problema.csv"

    main(dataset)