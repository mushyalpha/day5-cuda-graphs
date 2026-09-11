from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

COLORS = {
    "bg":       "#0d1117",
    "surface":  "#161b22",
    "text":     "#e6edf3",
    "muted":    "#7d8590",
    "grid":     "#21262d",

    "eager":    "#7d8590",
    "graph":    "#58a6ff",
    "compile":  "#3fb950",

    "ceiling":  "#d29922",

    "attention": "#f85149",
    "linear":    "#58a6ff",
    "norm_rope": "#3fb950",
    "other":     "#7d8590",

    "model_7b":  "#58a6ff",
    "model_05b": "#bc8cff",
}

CTX_COLORS = {128: "#58a6ff", 512: "#3fb950", 2048: "#8b949e", 1920: "#8b949e"}

# Day 4 HF eager, Qwen2.5-7B / H100. Day 5 did not re-measure HF.
# ctx=1920 cells use the Day-4 ctx=2048 number (closest cell).
HF_DAY4_MS = {
    (1, 128): 14.43,
    (1, 1920): 14.33,
    (4, 128): 14.49,
    (4, 1920): 14.27,
    (16, 128): 14.54,
    (16, 1920): 14.47,
    (64, 128): 14.31,
    (64, 1920): 22.57,
}

def apply_dark(fig, axes):
    fig.patch.set_facecolor(COLORS["bg"])
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(COLORS["surface"])
        ax.tick_params(colors=COLORS["text"], labelsize=11)
        ax.xaxis.label.set_color(COLORS["text"])
        ax.yaxis.label.set_color(COLORS["text"])
        ax.title.set_color(COLORS["text"])
        for spine in ax.spines.values():
            spine.set_color(COLORS["grid"])
        ax.grid(True, alpha=0.15, color=COLORS["muted"], linewidth=0.5)

WEIGHT_BYTES_7B = 14.0e9
KV_PER_TOKEN    = 57_344
HBM_BW          = 3.35e12

def roofline_tok_s(batch: int, ctx: int,
                   weight_bytes: float = WEIGHT_BYTES_7B,
                   kv_bytes_per_token: int = KV_PER_TOKEN,
                   bw: float = HBM_BW) -> float:
    total = weight_bytes + batch * ctx * kv_bytes_per_token
    return batch / (total / bw)

def hf_ms(row: dict) -> float | None:
    if row.get("hf_eager_ms"):
        return float(row["hf_eager_ms"])
    return HF_DAY4_MS.get((row["batch_size"], row["context_length"]))


def plot_roofline(ax, results: list[dict], title_suffix: str = ""):
    ax.set_title(f"Static eager vs CUDA graph vs HuggingFace{title_suffix}",
                 fontsize=13, fontweight="bold", pad=10)

    by_ctx: dict[int, dict] = {}
    for r in results:
        ctx = r["context_length"]
        if ctx not in by_ctx:
            by_ctx[ctx] = {
                "bs": [], "eager_tok_s": [], "graph_tok_s": [],
                "hf_tok_s": [], "floor_tok_s": [],
            }
        bs = r["batch_size"]
        floor_ms = r["floor_ms"]
        by_ctx[ctx]["bs"].append(bs)
        by_ctx[ctx]["floor_tok_s"].append(bs / (floor_ms / 1e3))
        if r.get("static_eager_ms"):
            by_ctx[ctx]["eager_tok_s"].append(bs / (r["static_eager_ms"] / 1e3))
        if r.get("graph_ms"):
            by_ctx[ctx]["graph_tok_s"].append(bs / (r["graph_ms"] / 1e3))
        h = hf_ms(r)
        if h:
            by_ctx[ctx]["hf_tok_s"].append(bs / (h / 1e3))

    for ctx in sorted(by_ctx.keys()):
        d = by_ctx[ctx]
        color = CTX_COLORS.get(ctx, COLORS["muted"])
        label = f"ctx={ctx:,}"

        ax.plot(d["bs"], d["floor_tok_s"], "--", color=color, alpha=0.35, linewidth=1.5)

        if d["eager_tok_s"]:
            ax.plot(d["bs"], d["eager_tok_s"], "o--", color=color, alpha=0.7,
                    linewidth=1.5, markersize=7, markerfacecolor="none",
                    markeredgewidth=1.8, label=f"{label} static eager")

        if d["graph_tok_s"]:
            ax.plot(d["bs"], d["graph_tok_s"], "-o", color=color, linewidth=2.2,
                    markersize=7, markeredgewidth=0, label=f"{label} graph", zorder=5)

        if d["hf_tok_s"]:
            ax.plot(d["bs"], d["hf_tok_s"], "-^", color=COLORS["ceiling"],
                    linewidth=2.0, markersize=8, markeredgewidth=0, alpha=0.95,
                    label=f"{label} HF eager", zorder=6)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    batch_sizes = sorted({r["batch_size"] for r in results})
    ax.set_xticks(batch_sizes)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Batch size", fontsize=12)
    ax.set_ylabel("tok/s", fontsize=12)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.3,
              edgecolor=COLORS["grid"], facecolor=COLORS["surface"],
              labelcolor=COLORS["text"], ncol=2)
    ax.text(0.97, 0.97, "── roofline ceiling", transform=ax.transAxes,
            fontsize=8, color=COLORS["muted"], ha="right", va="top", fontstyle="italic")
    ax.text(0.97, 0.03, "HF = Day 4, same GPU/model. ctx=1920 uses HF ctx=2048.",
            transform=ax.transAxes, fontsize=7.5, color=COLORS["muted"],
            ha="right", va="bottom")


def plot_other_collapse(ax, profile: dict):
    ax.set_title("Idle collapses. Kernel work does not.\nB=1, ctx=128, Qwen2.5-7B",
                 fontsize=13, fontweight="bold", pad=10)

    def row(mode: str):
        p = profile.get(mode, {})
        wall = p.get("cuda_event_wall_ms")
        gemm = p.get("gemm_ms_per_step")
        other = p.get("other_kernel_ms_per_step")
        idle = p.get("idle_ms_per_step")
        if None in (wall, gemm, other, idle):
            raise SystemExit(
                f"profile[{mode}] missing absolute-ms fields. Re-run parse_nsys_day5.py."
            )
        return wall, gemm, other, idle

    eager_wall, eager_gemm, eager_other, eager_idle = row("eager")
    graph_wall, graph_gemm, graph_other, graph_idle = row("graph")

    labels = ["GEMM", "other kernels", "idle gaps"]
    colors = [COLORS["linear"], "#8b949e", "rgba(125,133,144,0.45)"]
    # matplotlib wants hex
    colors = [COLORS["linear"], "#6e7681", "#30363d"]

    eager_vals = [eager_gemm, eager_other, eager_idle]
    graph_vals = [graph_gemm, graph_other, graph_idle]
    x = np.arange(2)
    width = 0.52
    bottoms = np.zeros(2)

    for i, (lab, col) in enumerate(zip(labels, colors)):
        vals = [eager_vals[i], graph_vals[i]]
        ax.bar(x, vals, width, bottom=bottoms, color=col, alpha=0.92,
               label=lab, edgecolor="none")
        for j, v in enumerate(vals):
            if v >= 1.4:
                ax.text(j, bottoms[j] + v / 2, f"{v:.1f}",
                        ha="center", va="center", fontsize=11,
                        fontweight="bold", color=COLORS["text"])
        bottoms += np.array(vals, dtype=float)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"static eager\n{eager_wall:.1f} ms", f"CUDA graph\n{graph_wall:.1f} ms"],
        fontsize=11,
    )
    ax.set_ylabel("milliseconds / decode step", fontsize=12)
    ax.set_ylim(0, max(eager_wall, graph_wall) * 1.18)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.3,
              edgecolor=COLORS["grid"], facecolor=COLORS["surface"],
              labelcolor=COLORS["text"])

    eager = profile["eager"]
    graph = profile["graph"]
    pct = graph.get("gemm_pct_peak")
    tb = graph.get("gemm_tb_s")
    ax.text(
        0.03, 0.97,
        f"CPU launches/step: {eager['kernel_launches_per_step']:,} → {graph['kernel_launches_per_step']}\n"
        f"Σ kernel: {eager['kernel_ms_per_step']:.2f} → {graph['kernel_ms_per_step']:.2f} ms\n"
        f"idle: {eager_idle:.1f} → {graph_idle:.1f} ms  (−{100*(1-graph_idle/eager_idle):.0f}%)\n"
        f"GEMM {graph['gemm_ms_per_step']:.1f} ms = {tb} TB/s ({pct:.0f}% of HBM peak)\n"
        f"Addressable leftover: ~{graph['other_kernel_ms_per_step']:.1f} ms unfused kernels",
        transform=ax.transAxes, fontsize=8.5, color=COLORS["muted"],
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=COLORS["surface"],
                  edgecolor=COLORS["grid"], alpha=0.88),
    )


def plot_speedup_vs_batch(ax, results_7b: list[dict],
                          results_05b: list[dict] | None = None):
    ax.set_title("Graph speedup vs batch  ·  ctx=128",
                 fontsize=13, fontweight="bold", pad=10)

    def extract_speedup(results):
        rows = [r for r in results if r.get("context_length") == 128]
        batches, speedups = [], []
        for r in sorted(rows, key=lambda x: x["batch_size"]):
            sv = r.get("graph_vs_static")
            if sv is not None:
                batches.append(r["batch_size"])
                speedups.append(sv)
        return batches, speedups

    batches_7, speedups_7 = extract_speedup(results_7b)
    ax.plot(batches_7, speedups_7, "-o", color=COLORS["model_7b"], linewidth=2.5,
            markersize=8, markeredgewidth=0, label="Qwen2.5-7B", zorder=5)
    for b, s in zip(batches_7, speedups_7):
        ax.text(b, s + 0.12, f"{s:.2f}×", ha="center", va="bottom",
                fontsize=9, color=COLORS["model_7b"], fontweight="bold")

    batches_05, speedups_05 = [], []
    if results_05b:
        batches_05, speedups_05 = extract_speedup(results_05b)
        ax.plot(batches_05, speedups_05, "-s", color=COLORS["model_05b"], linewidth=2.5,
                markersize=8, markeredgewidth=0, label="Qwen2.5-0.5B", zorder=5)
        if speedups_05:
            ax.text(batches_05[0], speedups_05[0] + 0.18, f"{speedups_05[0]:.2f}×",
                    ha="center", va="bottom", fontsize=9,
                    color=COLORS["model_05b"], fontweight="bold")

    all_bs = batches_7 or batches_05
    ax.axhline(1.0, color=COLORS["muted"], linewidth=1, linestyle=":", alpha=0.6)
    ax.text(all_bs[-1], 1.04, "no gain", fontsize=8,
            color=COLORS["muted"], ha="right")

    ax.set_xscale("log", base=2)
    ax.set_xticks(all_bs)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Batch size", fontsize=12)
    ax.set_ylabel("Speedup (static eager / graph)", fontsize=12)
    ymax = max(speedups_7 + speedups_05 + [2.5]) * 1.18
    ax.set_ylim(0.85, ymax)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.3,
              edgecolor=COLORS["grid"], facecolor=COLORS["surface"],
              labelcolor=COLORS["text"])
    ax.text(0.03, 0.07,
            "ctx=1920 series omitted (see caption / table).\n"
            "Graphs delete the gaps between kernels.",
            transform=ax.transAxes, fontsize=8.5, color=COLORS["muted"],
            fontstyle="italic", va="bottom", ha="left")

def make_charts(
    results_7b: list[dict],
    profile: dict,
    output_dir: Path,
    results_05b: list[dict] | None = None,
    model_name: str = "Qwen2.5-7B",
):
    output_dir.mkdir(parents=True, exist_ok=True)

    for fname, plot_fn, kwargs in [
        ("chart_1_roofline.png",  plot_roofline,          {"results": results_7b}),
        ("chart_2_other.png",     plot_other_collapse,    {"profile": profile}),
        ("chart_3_speedup.png",   plot_speedup_vs_batch,  {"results_7b": results_7b,
                                                            "results_05b": results_05b}),
    ]:
        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
        apply_dark(fig, [ax])
        plot_fn(ax, **kwargs)
        fig.tight_layout()
        fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        print(f"  ✓ {fname}")

    fig, axes = plt.subplots(1, 3, figsize=(27, 8), dpi=150)
    apply_dark(fig, axes)

    plot_roofline(axes[0], results_7b)
    plot_other_collapse(axes[1], profile)
    plot_speedup_vs_batch(axes[2], results_7b, results_05b)

    fig.suptitle(
        f"Day 5/45 · CUDA Graphs · {model_name} BF16 · H100 SXM · 3.35 TB/s",
        fontsize=18, fontweight="bold", color=COLORS["text"], y=1.01,
    )
    fig.tight_layout()
    hero_path = output_dir / "hero_chart.png"
    fig.savefig(hero_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"  ✓ hero_chart.png")
    return hero_path

def parse_args():
    p = argparse.ArgumentParser(description="Day 5 charts")
    p.add_argument("--data", required=True, help="results_day5.json (7B)")
    p.add_argument("--data-small", default=None, help="results_0.5B.json (optional)")
    p.add_argument("--profile", default=None,
                   help="profile_day5.json from nsys post-processing")
    p.add_argument("--output-dir", default="charts")
    return p.parse_args()

def main():
    args = parse_args()

    with open(args.data) as f:
        data_7b = json.load(f)
    results_7b = data_7b["results"]
    model_name = data_7b.get("model", "Qwen2.5-7B").split("/")[-1]

    results_05b = None
    if args.data_small and Path(args.data_small).exists():
        with open(args.data_small) as f:
            results_05b = json.load(f)["results"]

    profile = {}
    if args.profile and Path(args.profile).exists():
        with open(args.profile) as f:
            profile = json.load(f)
    else:
        print("⚠ No --profile file. Chart 2 needs parse_nsys_day5.py output.")

    output_dir = Path(args.output_dir)
    print(f"\nGenerating charts → {output_dir}/")
    hero = make_charts(results_7b, profile, output_dir, results_05b, model_name)
    print(f"\n  Hero chart: {hero}")
    print(f"  Individual: {output_dir}/chart_*.png")

if __name__ == "__main__":
    main()
