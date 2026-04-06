"""
Chopper Data Loader for AgentTrace
Converts Chopper's gpu.pkl output into JSON for the AgentTrace dashboard.

After running Lit Silicon on the AMD cluster, download the pkl files and run:
    python chopper_loader.py /path/to/gpu.pkl -o chopper_report.json

The dashboard loads this alongside agenttrace_report.json for combined
agent-level + hardware-level analysis.

Usage:
    # Single gpu.pkl file:
    python chopper_loader.py gpu.pkl

    # Multiple experiments (generates comparison):
    python chopper_loader.py llama_red_telemetry/gpu.pkl llama_realloc_telemetry/gpu.pkl

    # Serve as replay through telemetry server:
    python chopper_loader.py gpu.pkl --replay
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("Error: pandas required. Install with: pip install pandas")
    sys.exit(1)


def load_gpu_pkl(filepath):
    """Load Chopper gpu.pkl and return DataFrame."""
    df = pd.read_pickle(filepath)
    return df


def compute_straggler_analysis(df):
    """Compute per-GPU straggler s-values from frequency data.

    Mirrors Chopper's get_straggler_df() methodology:
    s-value = normalized lag from fastest GPU's frequency.
    """
    if "current_gfxclk" not in df.columns:
        return {}

    # Group by GPU, compute mean frequency
    gpu_avg = df.groupby("gpu")["current_gfxclk"].mean()
    max_freq = gpu_avg.max()
    if max_freq == 0:
        return {}

    s_values = {}
    for gpu_id, avg_freq in gpu_avg.items():
        s_values[int(gpu_id)] = round((max_freq - avg_freq) / max_freq, 4)

    lead_gpu = int(gpu_avg.idxmax())
    straggler_gpu = int(gpu_avg.idxmin())

    return {
        "s_values": s_values,
        "lead_gpu": lead_gpu,
        "straggler_gpu": straggler_gpu,
        "max_s_value": round(max(s_values.values()), 4),
    }


def compute_power_analysis(df):
    """Compute power statistics per GPU."""
    if "current_socket_power" not in df.columns:
        return {}

    gpu_power = df.groupby("gpu")["current_socket_power"]
    result = {}
    for gpu_id, group in gpu_power:
        result[int(gpu_id)] = {
            "mean_w": round(group.mean(), 1),
            "max_w": round(group.max(), 1),
            "min_w": round(group.min(), 1),
            "std_w": round(group.std(), 2),
        }
    return result


def compute_thermal_analysis(df):
    """Compute thermal spread and per-GPU temperature stats."""
    temp_col = None
    for col in ["temperature_hotspot", "temperature_edge", "current_temp"]:
        if col in df.columns:
            temp_col = col
            break

    if temp_col is None:
        return {}

    gpu_temp = df.groupby("gpu")[temp_col]
    per_gpu = {}
    for gpu_id, group in gpu_temp:
        per_gpu[int(gpu_id)] = {
            "mean_c": round(group.mean(), 1),
            "max_c": round(group.max(), 1),
            "min_c": round(group.min(), 1),
        }

    # Thermal spread over time
    timestamps = sorted(df["ts"].unique())
    spread_over_time = []
    for ts in timestamps[::10]:  # sample every 10th point
        snapshot = df[df["ts"] == ts]
        temps = snapshot[temp_col].values
        if len(temps) > 1:
            spread_over_time.append({
                "ts_ns": int(ts),
                "spread_c": round(float(temps.max() - temps.min()), 1),
                "max_c": round(float(temps.max()), 1),
                "min_c": round(float(temps.min()), 1),
            })

    return {
        "per_gpu": per_gpu,
        "spread_over_time": spread_over_time,
    }


def compute_frequency_timeline(df):
    """Extract per-GPU frequency timeline for visualization."""
    if "current_gfxclk" not in df.columns:
        return {}

    gpu_ids = sorted(df["gpu"].unique())
    min_ts = df["ts"].min()

    timelines = {}
    for gpu_id in gpu_ids:
        gpu_data = df[df["gpu"] == gpu_id].sort_values("ts")
        # Sample to max 300 points for dashboard performance
        step = max(1, len(gpu_data) // 300)
        sampled = gpu_data.iloc[::step]
        timelines[int(gpu_id)] = {
            "elapsed_s": [round((t - min_ts) / 1e9, 1)
                          for t in sampled["ts"].values],
            "gfxclk": [int(f) for f in sampled["current_gfxclk"].values],
            "memclk": [int(f) for f in sampled["current_uclk"].values]
            if "current_uclk" in sampled.columns else [],
        }

    return timelines


def compute_power_timeline(df):
    """Extract per-GPU power timeline."""
    if "current_socket_power" not in df.columns:
        return {}

    gpu_ids = sorted(df["gpu"].unique())
    min_ts = df["ts"].min()

    timelines = {}
    for gpu_id in gpu_ids:
        gpu_data = df[df["gpu"] == gpu_id].sort_values("ts")
        step = max(1, len(gpu_data) // 300)
        sampled = gpu_data.iloc[::step]
        timelines[int(gpu_id)] = {
            "elapsed_s": [round((t - min_ts) / 1e9, 1)
                          for t in sampled["ts"].values],
            "power_w": [int(p) for p in sampled["current_socket_power"].values],
        }

    return timelines


def pkl_to_report(filepath):
    """Convert a single gpu.pkl to a full analysis report."""
    df = load_gpu_pkl(filepath)

    gpu_ids = sorted(df["gpu"].unique())
    min_ts = df["ts"].min()
    max_ts = df["ts"].max()
    duration_s = (max_ts - min_ts) / 1e9

    report = {
        "source_file": str(filepath),
        "gpu_count": len(gpu_ids),
        "gpu_ids": [int(g) for g in gpu_ids],
        "duration_s": round(duration_s, 1),
        "sample_count": len(df),
        "columns": list(df.columns),
        "straggler": compute_straggler_analysis(df),
        "power": compute_power_analysis(df),
        "thermal": compute_thermal_analysis(df),
        "freq_timeline": compute_frequency_timeline(df),
        "power_timeline": compute_power_timeline(df),
    }

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Convert Chopper gpu.pkl to AgentTrace JSON report"
    )
    parser.add_argument(
        "pkl_files", nargs="+",
        help="Path(s) to gpu.pkl file(s)"
    )
    parser.add_argument(
        "-o", "--output", default="chopper_report.json",
        help="Output JSON filename (default: chopper_report.json)"
    )
    parser.add_argument(
        "--replay", action="store_true",
        help="Start telemetry server replaying the pkl data"
    )
    args = parser.parse_args()

    if args.replay:
        print("Replay mode not yet implemented — use live telemetry_server.py")
        sys.exit(1)

    reports = []
    for pkl_path in args.pkl_files:
        path = Path(pkl_path)
        if not path.exists():
            print(f"Error: {pkl_path} not found")
            continue
        print(f"Loading {pkl_path}...")
        report = pkl_to_report(path)
        report["experiment"] = path.parent.name
        reports.append(report)
        print(f"  {report['gpu_count']} GPUs, "
              f"{report['duration_s']}s duration, "
              f"{report['sample_count']} samples")
        if report["straggler"]:
            print(f"  Straggler: GPU {report['straggler']['straggler_gpu']} "
                  f"(s-value: {report['straggler']['max_s_value']})")
            print(f"  Lead:     GPU {report['straggler']['lead_gpu']}")

    output = {
        "version": "1.0",
        "tool": "chopper_loader",
        "experiments": reports,
    }

    import numpy as np

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    print(f"\nWrote {args.output} ({len(reports)} experiment(s))")
    print("Load in AgentTrace dashboard or analyze directly.")


if __name__ == "__main__":
    main()
