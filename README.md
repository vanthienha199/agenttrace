# AgentTrace

Infrastructure profiler for agentic AI workloads. Measures where time, tokens, and compute go inside an AI agent system.

Currently profiles [Project Ava](https://projectava.dev), an autonomous hardware verification agent. The profiler extracts data from Supabase (runs, iterations, test results, failures) and produces an interactive dashboard showing CPU vs GPU time decomposition, token amplification, pipeline timelines, and GPU idle analysis.

**Live:** [agenttrace.netlify.app](https://agenttrace.netlify.app)

---

## What It Measures

- **Time decomposition** — LLM inference (cloud GPU) vs simulation (local CPU) vs orchestration overhead per design
- **GPU idle time** — percentage of wall clock where GPU is waiting for CPU-bound tool execution
- **Token amplification** — how self-correction multiplies token usage (1 iteration vs 18 iterations)
- **Pipeline timeline** — visual activity trace showing GPU-burst → idle → CPU-sim → idle patterns
- **Failure taxonomy** — distribution of error categories across agent iterations
- **Cost estimation** — estimated API cost per design based on token consumption

## Key Findings (Project Ava)

| Metric | Measured | Literature |
|--------|----------|------------|
| GPU idle during tool execution | 43.3% | 30-55% (Kim et al., HPCA 2026) |
| LLM call amplification | 3.6x avg (up to 18x) | 9.2x (Kim et al.) |
| Simulation time (CPU) | 0.6% of wall clock | — |
| LLM inference (GPU) | 56.7% of wall clock | — |

The measured GPU idle time (43.3%) falls within the range reported by Kim et al. in "The Cost of Dynamic Reasoning" (arXiv:2506.04301), validating their findings on a real hardware verification agent.

## Quick Start

```bash
# Generate the profiling report from Supabase data
python3 profiler.py

# Preview the dashboard
python3 -m http.server 8081
# Open http://localhost:8081
```

## Files

```
agenttrace/
├── profiler.py              # Data extraction + metric computation
├── agenttrace_report.json   # Generated profiling report
├── index.html               # Mission Control dashboard
└── README.md
```

## References

- Kim et al., "The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective" (arXiv:2506.04301, HPCA 2026)
- Zhu et al., "NanoFlow: Towards Optimal Large Language Model Serving Throughput" (OSDI 2025)

## Author

**Ha Le** — University of Central Florida
