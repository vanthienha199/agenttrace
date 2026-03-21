"""
AgentTrace Profiler — Extract and analyze agentic workload metrics from Project Ava.

Pulls data from Supabase (runs + iterations + test_results + failures),
computes CPU vs GPU time decomposition, token amplification, idle time,
and convergence patterns.

Output: agenttrace_report.json (consumed by the dashboard)
"""

import json
import urllib.request
from collections import defaultdict

SUPABASE_URL = "https://yvpmoyzggbcfaldhsbkl.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inl2cG1veXpnZ2JjZmFsZGhzYmtsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzM5Njg5OTMsImV4cCI6MjA4OTU0NDk5M30.qgpE7ayn57SMzeJVgp7mBu3VJ825gTTe8G6OmfTc1b0"

# Token cost per million (Claude Sonnet pricing)
COST_PER_M_INPUT = 3.0
COST_PER_M_OUTPUT = 15.0

# Estimated tokens per iteration for claude_cli (no token data reported)
# Based on anthropic_api runs: avg ~4000 input, ~3000 output per iteration
EST_TOKENS_IN_PER_ITER = 4000
EST_TOKENS_OUT_PER_ITER = 3000


def fetch(table, params=""):
    url = f"{SUPABASE_URL}/rest/v1/{table}?select=*{params}"
    req = urllib.request.Request(url, headers={
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    })
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def compute_time_decomposition(runs, iterations_by_run):
    """Compute CPU vs GPU time for each run."""
    results = []
    for run in runs:
        run_id = run["id"]
        iters = iterations_by_run.get(run_id, [])

        total_llm_ms = sum(it.get("llm_latency_ms", 0) or 0 for it in iters)
        total_sim_ms = sum(it.get("sim_latency_ms", 0) or 0 for it in iters)
        total_wall_ms = run.get("total_latency_ms", 0) or 0

        # Orchestration = wall clock - LLM - simulation (network, parsing, file I/O)
        orchestration_ms = max(0, total_wall_ms - total_llm_ms - total_sim_ms)

        # GPU idle = time NOT doing LLM inference (sim + orchestration)
        gpu_idle_ms = total_sim_ms + orchestration_ms
        gpu_idle_pct = (gpu_idle_ms / total_wall_ms * 100) if total_wall_ms > 0 else 0

        # CPU active = simulation + orchestration
        cpu_active_ms = total_sim_ms + orchestration_ms
        cpu_active_pct = (cpu_active_ms / total_wall_ms * 100) if total_wall_ms > 0 else 0

        results.append({
            "design_name": run["design_name"],
            "total_wall_ms": round(total_wall_ms, 1),
            "llm_ms": round(total_llm_ms, 1),
            "sim_ms": round(total_sim_ms, 1),
            "orchestration_ms": round(orchestration_ms, 1),
            "llm_pct": round(total_llm_ms / total_wall_ms * 100, 1) if total_wall_ms > 0 else 0,
            "sim_pct": round(total_sim_ms / total_wall_ms * 100, 1) if total_wall_ms > 0 else 0,
            "orchestration_pct": round(orchestration_ms / total_wall_ms * 100, 1) if total_wall_ms > 0 else 0,
            "gpu_idle_pct": round(gpu_idle_pct, 1),
            "cpu_active_pct": round(cpu_active_pct, 1),
            "iterations": run.get("iterations", 1),
            "corrections": run.get("corrections", 0),
        })

    return results


def compute_token_amplification(runs, iterations_by_run):
    """Compute token usage and amplification factor per run."""
    results = []
    for run in runs:
        run_id = run["id"]
        iters = iterations_by_run.get(run_id, [])
        n_iters = max(len(iters), run.get("iterations", 1))

        # Use real token data if available, otherwise estimate
        tokens_in = run.get("tokens_in", 0) or 0
        tokens_out = run.get("tokens_out", 0) or 0

        if tokens_in == 0:
            # Estimate from iteration count
            tokens_in = n_iters * EST_TOKENS_IN_PER_ITER
            tokens_out = n_iters * EST_TOKENS_OUT_PER_ITER
            estimated = True
        else:
            estimated = False

        total_tokens = tokens_in + tokens_out

        # Amplification: how many more tokens vs a single-shot (1 iteration)
        amplification = n_iters  # Linear approximation

        # Cost estimate
        cost_input = tokens_in / 1_000_000 * COST_PER_M_INPUT
        cost_output = tokens_out / 1_000_000 * COST_PER_M_OUTPUT
        cost_total = cost_input + cost_output

        results.append({
            "design_name": run["design_name"],
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "total_tokens": total_tokens,
            "iterations": n_iters,
            "amplification_factor": round(amplification, 1),
            "cost_usd": round(cost_total, 4),
            "estimated": estimated,
        })

    return results


def compute_convergence(runs, iterations_by_run):
    """Compute convergence curves — pass count over iterations."""
    results = []
    for run in runs:
        run_id = run["id"]
        iters = iterations_by_run.get(run_id, [])
        iters_sorted = sorted(iters, key=lambda x: x.get("iteration_number", 0))

        curve = []
        for it in iters_sorted:
            curve.append({
                "iteration": it.get("iteration_number", 0),
                "type": it.get("iteration_type", "unknown"),
                "passed": it.get("passed", False),
                "pass_count": it.get("pass_count", 0),
                "fail_count": it.get("fail_count", 0),
                "llm_ms": round(it.get("llm_latency_ms", 0) or 0, 1),
                "sim_ms": round(it.get("sim_latency_ms", 0) or 0, 1),
                "ic": it.get("ic", 0),
                "ir": it.get("ir", 0),
            })

        results.append({
            "design_name": run["design_name"],
            "total_iterations": len(curve),
            "first_pass": curve[0].get("passed", False) if curve else False,
            "curve": curve,
        })

    return results


def compute_failure_distribution(failures):
    """Compute failure category distribution."""
    cats = defaultdict(int)
    for f in failures:
        cats[f.get("category", "unknown")] += 1
    return dict(cats)


def compute_pipeline_timeline(runs, iterations_by_run):
    """Build a timeline of GPU/CPU activity for each run (for visualization)."""
    results = []
    for run in runs:
        run_id = run["id"]
        iters = iterations_by_run.get(run_id, [])
        iters_sorted = sorted(iters, key=lambda x: x.get("iteration_number", 0))

        timeline = []
        t = 0  # cumulative time offset

        for it in iters_sorted:
            llm_ms = it.get("llm_latency_ms", 0) or 0
            sim_ms = it.get("sim_latency_ms", 0) or 0

            # LLM call (GPU active, CPU idle)
            timeline.append({
                "start": round(t, 1),
                "duration": round(llm_ms, 1),
                "type": "llm_inference",
                "resource": "GPU (cloud)",
                "label": f"{'Generate' if it.get('iteration_type') == 'generate' else 'Correct'} #{it.get('iteration_number', 0)}",
            })
            t += llm_ms

            # Simulation (CPU active, GPU idle)
            timeline.append({
                "start": round(t, 1),
                "duration": round(sim_ms, 1),
                "type": "simulation",
                "resource": "CPU (local)",
                "label": f"iverilog sim #{it.get('iteration_number', 0)}",
            })
            t += sim_ms

        results.append({
            "design_name": run["design_name"],
            "timeline": timeline,
            "total_duration_ms": round(t, 1),
        })

    return results


def compute_summary_stats(time_decomp, token_data, convergence):
    """Compute aggregate summary statistics."""
    total_llm = sum(t["llm_ms"] for t in time_decomp)
    total_sim = sum(t["sim_ms"] for t in time_decomp)
    total_orch = sum(t["orchestration_ms"] for t in time_decomp)
    total_wall = sum(t["total_wall_ms"] for t in time_decomp)
    total_tokens = sum(t["total_tokens"] for t in token_data)
    total_cost = sum(t["cost_usd"] for t in token_data)
    total_iters = sum(t["iterations"] for t in token_data)
    first_pass_count = sum(1 for c in convergence if c["first_pass"])

    return {
        "total_designs": len(time_decomp),
        "total_wall_time_s": round(total_wall / 1000, 1),
        "total_llm_time_s": round(total_llm / 1000, 1),
        "total_sim_time_s": round(total_sim / 1000, 1),
        "total_orchestration_s": round(total_orch / 1000, 1),
        "avg_llm_pct": round(total_llm / total_wall * 100, 1) if total_wall > 0 else 0,
        "avg_sim_pct": round(total_sim / total_wall * 100, 1) if total_wall > 0 else 0,
        "avg_gpu_idle_pct": round((total_sim + total_orch) / total_wall * 100, 1) if total_wall > 0 else 0,
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 2),
        "total_iterations": total_iters,
        "avg_iterations_per_design": round(total_iters / len(time_decomp), 1),
        "first_pass_rate": round(first_pass_count / len(convergence) * 100, 1),
        "designs_needing_correction": len(convergence) - first_pass_count,
        "avg_amplification": round(total_iters / len(token_data), 1),
        # Kim et al. comparison points
        "kim_gpu_idle_finding": "30-55%",
        "ava_gpu_idle_actual": round((total_sim + total_orch) / total_wall * 100, 1) if total_wall > 0 else 0,
        "kim_llm_calls_multiplier": "9.2x",
        "ava_avg_iterations": round(total_iters / len(time_decomp), 1),
    }


def main():
    print("AgentTrace Profiler v1.0")
    print("Extracting data from Project Ava (Supabase)...\n")

    # Fetch all data
    runs = fetch("runs", "&order=design_name.asc")
    iterations = fetch("iterations")
    test_results = fetch("test_results")
    failures = fetch("failures")
    designs = fetch("designs", "&order=name.asc")

    print(f"  Runs: {len(runs)}")
    print(f"  Iterations: {len(iterations)}")
    print(f"  Test results: {len(test_results)}")
    print(f"  Failures: {len(failures)}")
    print(f"  Designs: {len(designs)}")

    # Group iterations by run_id
    iters_by_run = defaultdict(list)
    for it in iterations:
        iters_by_run[it["run_id"]].append(it)

    # Compute all metrics
    print("\nComputing metrics...")

    time_decomp = compute_time_decomposition(runs, iters_by_run)
    print("  Time decomposition: done")

    token_data = compute_token_amplification(runs, iters_by_run)
    print("  Token amplification: done")

    convergence = compute_convergence(runs, iters_by_run)
    print("  Convergence curves: done")

    failure_dist = compute_failure_distribution(failures)
    print("  Failure distribution: done")

    timelines = compute_pipeline_timeline(runs, iters_by_run)
    print("  Pipeline timelines: done")

    summary = compute_summary_stats(time_decomp, token_data, convergence)
    print("  Summary statistics: done")

    # Build report
    report = {
        "version": "1.0",
        "agent": "Project Ava",
        "description": "Agentic workload infrastructure profile",
        "summary": summary,
        "time_decomposition": time_decomp,
        "token_amplification": token_data,
        "convergence": convergence,
        "failure_distribution": failure_dist,
        "timelines": timelines,
        "designs": [{"name": d["name"], "category": d.get("category", "unknown")} for d in designs],
        "methodology": {
            "llm_inference": "Cloud GPU (Anthropic API — Claude Sonnet). Time measured as round-trip API call latency.",
            "simulation": "Local CPU (Icarus Verilog + cocotb 2.0). Time measured as subprocess wall clock.",
            "orchestration": "Local CPU (Python — file I/O, parsing, prompt assembly). Computed as wall_clock - llm - sim.",
            "token_estimation": "claude_cli backend does not report tokens. Estimated at ~4000 input + ~3000 output per iteration based on anthropic_api calibration runs.",
            "cost_model": f"Claude Sonnet: ${COST_PER_M_INPUT}/M input, ${COST_PER_M_OUTPUT}/M output tokens.",
            "gpu_idle": "GPU is 100% idle during simulation and orchestration phases. No local GPU used.",
            "reference_papers": [
                "Kim et al., 'The Cost of Dynamic Reasoning' (arXiv:2506.04301, HPCA 2026) — GPU idle 30-55% during tool execution",
                "Zhu et al., 'NanoFlow: Towards Optimal LLM Serving Throughput' (OSDI 2025) — Intra-device parallelism for heterogeneous operations",
            ],
        },
    }

    # Save
    out_path = "agenttrace_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  AGENTTRACE SUMMARY")
    print(f"{'='*60}")
    print(f"  Designs profiled:        {summary['total_designs']}")
    print(f"  Total wall time:         {summary['total_wall_time_s']}s")
    print(f"  LLM inference time:      {summary['total_llm_time_s']}s ({summary['avg_llm_pct']}%)")
    print(f"  Simulation time:         {summary['total_sim_time_s']}s ({summary['avg_sim_pct']}%)")
    print(f"  GPU idle:                {summary['avg_gpu_idle_pct']}%")
    print(f"  Total tokens:            {summary['total_tokens']:,}")
    print(f"  Estimated cost:          ${summary['total_cost_usd']}")
    print(f"  Avg iterations/design:   {summary['avg_iterations_per_design']}")
    print(f"  First-pass success:      {summary['first_pass_rate']}%")
    print(f"  Designs needing fixes:   {summary['designs_needing_correction']}")
    print(f"\n  Kim et al. comparison:")
    print(f"    GPU idle (their finding): {summary['kim_gpu_idle_finding']}")
    print(f"    GPU idle (Ava measured):  {summary['ava_gpu_idle_actual']}%")
    print(f"    LLM call multiplier:      {summary['kim_llm_calls_multiplier']} (theirs) vs {summary['ava_avg_iterations']}x (Ava)")


if __name__ == "__main__":
    main()
