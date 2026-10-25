"""Create report figures from results/*.csv  ->  results/*.png"""
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config

R = config.RESULTS_DIR


def save(fig, name):
    fig.tight_layout()
    fig.savefig(R / name, dpi=200)
    plt.close(fig)
    print("saved", R / name)


def retrieval_plots():
    path = R / "retrieval_results.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)

    # 1. Chunk size effect (hit rate @5, no rerank), per mode/embedder
    d = df[(df.k == 5) & (~df.rerank)].copy()
    d["system"] = d["mode"] + " (" + d["embedder"] + ")"
    p = d.pivot_table(index="chunk_size", columns="system", values="hit_rate")
    ax = p.plot(marker="o", figsize=(7, 4))
    ax.set(title="Effect of chunk size on Hit Rate@5", xlabel="Chunk size (words)", ylabel="Hit Rate@5")
    save(ax.figure, "fig_chunk_size.png")

    # 2. Retrieval strategy comparison at best chunk size
    best_cs = d.groupby("chunk_size")["hit_rate"].mean().idxmax()
    d2 = df[(df.k == 5) & (df.chunk_size == best_cs)].copy()
    d2["system"] = d2["mode"] + "/" + d2["embedder"] + d2["rerank"].map({True: " +rerank", False: ""})
    d2 = d2.sort_values("mrr")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(d2["system"], d2["mrr"])
    ax.set(title=f"MRR@5 by retrieval strategy (chunk size {best_cs})", xlabel="MRR@5")
    save(fig, "fig_strategies_mrr.png")

    # 3. Hit rate vs k
    d3 = df[(df.chunk_size == best_cs) & (~df.rerank)].copy()
    d3["system"] = d3["mode"] + " (" + d3["embedder"] + ")"
    ax = d3.pivot_table(index="k", columns="system", values="hit_rate").plot(marker="o", figsize=(7, 4))
    ax.set(title="Hit Rate vs top-k", ylabel="Hit Rate")
    save(ax.figure, "fig_hit_vs_k.png")

    # 4. Latency vs accuracy (reranking cost)
    d4 = df[(df.k == 5) & (df.chunk_size == best_cs)]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for rr, grp in d4.groupby("rerank"):
        ax.scatter(grp["latency_ms"], grp["hit_rate"], label="with reranker" if rr else "no reranker")
    ax.set(title="Retrieval latency vs Hit Rate@5", xlabel="Avg latency (ms)", ylabel="Hit Rate@5")
    ax.legend()
    save(fig, "fig_latency_tradeoff.png")


def generation_plots():
    files = glob.glob(str(R / "generation_summary_*.csv"))
    if not files:
        return
    frames = []
    for f in files:
        s = pd.read_csv(f, index_col=0)
        s["llm"] = f.split("generation_summary_")[1][:-4]
        frames.append(s.reset_index())
    df = pd.concat(frames)
    for metric in ["correctness", "faithfulness", "rouge_l"]:
        if metric not in df or df[metric].isna().all():
            continue
        p = df.pivot_table(index="variant", columns="llm", values=metric)
        ax = p.plot.bar(figsize=(7, 4), rot=20)
        ax.set(title=f"{metric} by variant and LLM", ylabel=metric)
        save(ax.figure, f"fig_gen_{metric}.png")


if __name__ == "__main__":
    retrieval_plots()
    generation_plots()
