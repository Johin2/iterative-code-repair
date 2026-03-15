# TaskRouter: Cost-Aware Sub-Task Model Routing for Agentic Software Engineering Workflows

This repository contains the experiment code and results for the paper:

> **TaskRouter: Cost-Aware Sub-Task Model Routing for Agentic Software Engineering Workflows**
> Johin Johny Arimbur

## Abstract

LLM-based coding agents uniformly employ expensive frontier models for every sub-task, regardless of complexity. TaskRouter decomposes agentic SE workflows into typed sub-tasks (exploration, comprehension, localization, patch generation, test generation, verification), estimates per-step difficulty using lightweight code-structural features, and dynamically routes each sub-task to the cheapest capable model. On a benchmark of 20 coding tasks using four model tiers (Gemini 2.5 Flash, GPT-4o-mini, Claude 3.5 Haiku, GPT-4o), TaskRouter achieves **89.1% cost reduction** while retaining **94.7%** of the frontier baseline's resolve rate.

## Repository Structure

```
.
├── paper/
│   ├── main.tex              # LaTeX source
│   ├── references.bib        # Bibliography
│   └── figures/               # Paper figures
├── experiments/
│   ├── config.py             # Model tiers, thresholds, settings
│   ├── tasks.py              # 20 benchmark task definitions
│   ├── taskrouter.py         # Core routing logic
│   ├── subtask_classifier.py # Sub-task classification rules
│   ├── llm_client.py         # Unified API client (OpenAI, Anthropic, Google)
│   ├── run_experiment.py     # Experiment runner (all strategies)
│   ├── analyze_results.py    # Results analysis and LaTeX table generation
│   └── requirements.txt      # Python dependencies
├── results/
│   └── all_results.json      # Raw experiment results
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- API keys for OpenAI, Anthropic, and Google

### Installation

```bash
pip install -r experiments/requirements.txt
```

### Environment Variables

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

## Running Experiments

### Run all strategies

```bash
cd experiments
python run_experiment.py --num-tasks 20
```

### Run a single strategy

```bash
python run_experiment.py --strategy taskrouter --num-tasks 20
```

### Available strategies

| Strategy | Description |
|----------|-------------|
| `single_frontier` | All sub-tasks on GPT-4o (T4) |
| `single_small` | All sub-tasks on Gemini 2.5 Flash (T1) |
| `single_medium` | All sub-tasks on GPT-4o-mini (T2) |
| `single_large` | All sub-tasks on Claude 3.5 Haiku (T3) |
| `random_routing` | Random tier per sub-task |
| `type_only_routing` | Fixed tier by sub-task type (no difficulty estimation) |
| `taskrouter` | Full TaskRouter with difficulty estimation + cascading |

### Analyze results

```bash
python analyze_results.py
```

This generates LaTeX tables and figure scripts in `paper_outputs/`.

## Key Results

| Method | Resolve % | Cost ($) | Reduction |
|--------|-----------|----------|-----------|
| Single-Frontier (T4) | 95.0% | $0.3793 | --- |
| Single-Small (T1) | 65.0% | $0.0383 | 89.9% |
| Single-Medium (T2) | 90.0% | $0.0260 | 93.1% |
| Single-Large (T3) | 100.0% | $0.2160 | 43.1% |
| Random Routing | 85.0% | $0.1492 | 60.7% |
| Type-Only Routing | 95.0% | $0.1126 | 70.3% |
| **TaskRouter** | **90.0%** | **$0.0413** | **89.1%** |

## Model Pool

| Tier | Model | Input $/1M | Output $/1M |
|------|-------|------------|-------------|
| T1 (Small) | Gemini 2.5 Flash | $0.15 | $0.60 |
| T2 (Medium) | GPT-4o-mini | $0.15 | $0.60 |
| T3 (Large) | Claude 3.5 Haiku | $0.80 | $4.00 |
| T4 (Frontier) | GPT-4o | $2.50 | $10.00 |

## Citation

```bibtex
@article{arimbur2026taskrouter,
  title={TaskRouter: Cost-Aware Sub-Task Model Routing for Agentic Software Engineering Workflows},
  author={Arimbur, Johin Johny},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT
