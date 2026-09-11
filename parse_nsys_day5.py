from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

NVTX_STEP_NAME = "decode_step"

# All of these are the linear/math backend, not just the two nvjet shapes
# that showed up in the old top-6 table.
GEMM_KEYS = ("nvjet", "gemm", "cutlass", "cublas", "xmma", "gemv")
ATTN_KEYS = (
    "attn", "attention", "sdpa", "scaled_dot_product", "fmha", "flash",
    "softmax", "gather",
)
ELEM_KEYS = ("elementwise",)


def query(db_path: str, sql: str) -> list[tuple]:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def tables(db_path: str) -> set[str]:
    rows = query(db_path, "SELECT name FROM sqlite_master WHERE type='table'")
    return {r[0] for r in rows}


def nvtx_predicate(alias: str = "n") -> str:
    return (
        f"({alias}.text = '{NVTX_STEP_NAME}' "
        f"OR {alias}.text LIKE '%{NVTX_STEP_NAME}%')"
    )


def in_decode_step(event_alias: str = "k") -> str:
    return f"""
        EXISTS (
            SELECT 1 FROM NVTX_EVENTS n
            WHERE {nvtx_predicate("n")}
              AND {event_alias}.start >= n.start
              AND {event_alias}.end   <= n.end
        )
    """


def get_step_wall_times_us(db_path: str) -> list[float]:
    rows = query(db_path, f"""
        SELECT (end - start) / 1000.0
        FROM NVTX_EVENTS
        WHERE {nvtx_predicate("NVTX_EVENTS")}
        ORDER BY start
    """)
    return [float(r[0]) for r in rows]


def resolve_name(name) -> str:
    if name is None:
        return ""
    if isinstance(name, bytes):
        return name.decode("utf-8", errors="replace")
    return str(name)


def categorise(name: str) -> str:
    low = resolve_name(name).lower()
    if any(k in low for k in GEMM_KEYS):
        return "linear"
    if any(k in low for k in ATTN_KEYS):
        return "attention"
    if any(k in low for k in ELEM_KEYS):
        return "elementwise"
    return "other"


def kernel_rows(db_path: str) -> list[tuple]:
    present = tables(db_path)
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in present:
        return []

    join = ""
    name_expr = "CAST(k.shortName AS TEXT)"
    if "StringIds" in present:
        join = "LEFT JOIN StringIds s ON s.id = k.shortName"
        name_expr = "COALESCE(s.value, CAST(k.shortName AS TEXT))"

    where = ""
    if "NVTX_EVENTS" in present:
        where = f"WHERE {in_decode_step('k')}"

    return query(db_path, f"""
        SELECT {name_expr} AS name,
               SUM(k.end - k.start) / 1000.0 AS total_us,
               COUNT(*) AS n
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        {join}
        {where}
        GROUP BY 1
        ORDER BY total_us DESC
    """)


def cpu_launches(db_path: str) -> int:
    present = tables(db_path)
    if "CUPTI_ACTIVITY_KIND_RUNTIME" not in present or "StringIds" not in present:
        return -1
    where = in_decode_step("r") if "NVTX_EVENTS" in present else "1"
    rows = query(db_path, f"""
        SELECT COUNT(*)
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON s.id = r.nameId
        WHERE {where}
          AND (
                s.value LIKE '%LaunchKernel%'
             OR s.value LIKE '%GraphLaunch%'
             OR s.value LIKE '%cuLaunchKernel%'
          )
    """)
    return int(rows[0][0]) if rows else -1


def get_kernel_stats(db_path: str, nsys_wall_times_us: list[float]) -> dict:
    rows = kernel_rows(db_path)
    n_steps = len(nsys_wall_times_us) or 1

    cat_us: dict[str, float] = {}
    gemm_us = 0.0
    gemm_n = 0
    gemm_variants: list[dict] = []
    total_kernels = 0
    grand_us = 0.0
    top = []

    for name, total_us, n in rows:
        us = float(total_us or 0.0)
        count = int(n or 0)
        grand_us += us
        total_kernels += count
        cat = categorise(name)
        cat_us[cat] = cat_us.get(cat, 0.0) + us
        top.append({
            "name": resolve_name(name),
            "ms_per_step": round(us / n_steps / 1000.0, 3),
            "n_per_step": round(count / n_steps, 1),
        })
        if cat == "linear":
            gemm_us += us
            gemm_n += count
            gemm_variants.append({
                "name": resolve_name(name),
                "ms_per_step": round(us / n_steps / 1000.0, 3),
                "n_per_step": round(count / n_steps, 1),
            })

    launches = cpu_launches(db_path)
    kernel_ms = grand_us / n_steps / 1000.0
    gemm_ms = gemm_us / n_steps / 1000.0
    cat_ms = {k: round(v / n_steps / 1000.0, 3) for k, v in cat_us.items()}

    return {
        "kernels_per_step": total_kernels // n_steps,
        "cpu_launches_per_step": launches // n_steps if launches >= 0 else -1,
        "n_steps": n_steps,
        "kernel_ms_per_step": round(kernel_ms, 3),
        "gemm_ms_per_step": round(gemm_ms, 3),
        "other_kernel_ms_per_step": round(kernel_ms - gemm_ms, 3),
        "cat_ms_per_step": cat_ms,
        "gemm_variants": gemm_variants,
        "gemm_kernel_count_per_step": round(gemm_n / n_steps, 1),
        "top_kernels": top[:12],
        "nsys_wall_ms_median": round(
            sorted(nsys_wall_times_us)[len(nsys_wall_times_us) // 2] / 1000.0, 3
        ) if nsys_wall_times_us else None,
    }


def attach_cuda_event_idle(stats: dict, wall_ms: float, weight_gb: float, peak_tb_s: float) -> dict:
    kernel_ms = stats["kernel_ms_per_step"]
    idle_ms = max(0.0, wall_ms - kernel_ms)
    gemm_ms = stats["gemm_ms_per_step"]
    gemm_tb_s = (weight_gb / (gemm_ms / 1e3)) / 1e3 if gemm_ms > 0 else None
    return {
        **stats,
        "cuda_event_wall_ms": round(wall_ms, 3),
        "idle_ms_per_step": round(idle_ms, 3),
        "gpu_busy_fraction": round(100.0 * kernel_ms / wall_ms, 1) if wall_ms else None,
        "gemm_tb_s": round(gemm_tb_s, 2) if gemm_tb_s else None,
        "gemm_pct_peak": round(100.0 * gemm_tb_s / peak_tb_s, 1) if gemm_tb_s else None,
        "note": (
            "idle = CUDA-event wall − nsys Σ kernel time. "
            "nsys wall is inflated by tracing overhead; kernel durations are not."
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eager", required=True, help="nsys_eager.sqlite")
    p.add_argument("--graph", required=True, help="nsys_graph.sqlite")
    p.add_argument("--output", default="profile_day5.json")
    p.add_argument("--eager-wall-ms", type=float, default=21.235,
                   help="CUDA-event median ms/step for static eager (B=1 ctx=128)")
    p.add_argument("--graph-wall-ms", type=float, default=9.545,
                   help="CUDA-event median ms/step for graph (B=1 ctx=128)")
    p.add_argument("--weight-gb", type=float, default=15.23,
                   help="Weight traffic per decode step, GB (7B BF16)")
    p.add_argument("--peak-tb-s", type=float, default=3.35)
    args = p.parse_args()

    walls = {"eager": args.eager_wall_ms, "graph": args.graph_wall_ms}
    result = {}
    for mode, db_path in [("eager", args.eager), ("graph", args.graph)]:
        if not Path(db_path).exists():
            print(f"WARNING: {db_path} not found, skipping {mode}")
            continue

        print(f"Processing {mode}: {db_path}")
        nsys_wall = get_step_wall_times_us(db_path)
        stats = attach_cuda_event_idle(
            get_kernel_stats(db_path, nsys_wall),
            walls[mode],
            args.weight_gb,
            args.peak_tb_s,
        )
        result[mode] = {
            "gpu_busy_fraction": stats["gpu_busy_fraction"],
            "kernel_launches_per_step": stats["cpu_launches_per_step"],
            "kernels_per_step": stats["kernels_per_step"],
            "cuda_event_wall_ms": stats["cuda_event_wall_ms"],
            "kernel_ms_per_step": stats["kernel_ms_per_step"],
            "idle_ms_per_step": stats["idle_ms_per_step"],
            "gemm_ms_per_step": stats["gemm_ms_per_step"],
            "other_kernel_ms_per_step": stats["other_kernel_ms_per_step"],
            "cat_ms_per_step": stats["cat_ms_per_step"],
            "gemm_tb_s": stats["gemm_tb_s"],
            "gemm_pct_peak": stats["gemm_pct_peak"],
            "gemm_variants": stats["gemm_variants"],
            "gemm_kernel_count_per_step": stats["gemm_kernel_count_per_step"],
            "top_kernels": stats["top_kernels"],
            "n_steps": stats["n_steps"],
            "nsys_wall_ms_median": stats["nsys_wall_ms_median"],
            "note": stats["note"],
        }
        s = result[mode]
        print(f"  CUDA-event wall: {s['cuda_event_wall_ms']} ms")
        print(f"  Σ kernel:        {s['kernel_ms_per_step']} ms")
        print(f"  idle:            {s['idle_ms_per_step']} ms")
        print(f"  busy:            {s['gpu_busy_fraction']}%  (kernel / CUDA-event wall)")
        print(f"  GEMM:            {s['gemm_ms_per_step']} ms  "
              f"({s['gemm_kernel_count_per_step']} kernels, "
              f"{len(s['gemm_variants'])} variants, "
              f"{s['gemm_tb_s']} TB/s = {s['gemm_pct_peak']}% of {args.peak_tb_s})")
        print(f"  other kernels:   {s['other_kernel_ms_per_step']} ms")
        print(f"  CPU launches/step: {s['kernel_launches_per_step']}")
        print(f"  Kernels/step:      {s['kernels_per_step']}")
        print("  GEMM variants:")
        for v in s["gemm_variants"]:
            print(f"    {v['ms_per_step']:6.3f} ms  n={v['n_per_step']:6.1f}  {v['name']}")

    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
