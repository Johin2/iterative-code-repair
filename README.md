# How Many Tries Does It Take? Iterative Self-Repair in LLM Code Generation

This repository contains the experiment code, results, and paper for:

> **How Many Tries Does It Take? Iterative Self-Repair in LLM Code Generation Across Model Scales and Benchmarks**
> Johin Johny Arimbur

## Abstract

We investigate iterative self-repair across five open-weight models spanning 8B to 70B parameters and two architectures (dense and mixture-of-experts): Llama 3.1 8B, Llama 3.3 70B, Llama 4 Scout 17B (MoE, 16 experts), Llama 4 Maverick 17B (MoE, 128 experts), and Qwen3 32B. Evaluating on HumanEval (164 problems) and MBPP Sanitized (257 problems) with up to five attempts per problem, we find that self-repair universally improves pass rates. On HumanEval, gains range from +4.9 to +14.0 percentage points; on MBPP, gains range from +16.0 to +23.0 percentage points. Most gains are concentrated in the first two repair rounds, and error type strongly predicts repair success.

## Repository Structure

```
.
├── paper/
│   ├── main.tex              # LaTeX source
│   ├── main.pdf              # Compiled paper
│   ├── references.bib        # Bibliography
│   └── figures/              # Paper figures (PNG + PDF)
├── experiments/
│   ├── __init__.py           # Package marker
│   ├── config.py             # Models, paths, hyperparameters
│   ├── api_client.py         # Groq API client with retry logic
│   ├── self_repair.py        # Prompt building and code extraction
│   ├── code_executor.py      # Sandboxed Python execution
│   ├── data_loader.py        # HumanEval/MBPP data loading
│   ├── run_experiment.py     # Main experiment runner
│   ├── run_ablation.py       # Prompt ablation study
│   ├── analyze_results.py    # Result analysis and visualization
│   └── analyze_ablation.py   # Ablation result analysis
├── results/
│   ├── *.json                # Per-model result files (10 main + 2 summaries)
│   └── ablation/             # Ablation study results (6 files)
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT license
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/) (free tier is sufficient)

### Installation

```bash
git clone https://github.com/Johin2/Self-repair.git
cd Self-repair
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your-key-here
```

## Reproducing the Paper Results

All scripts are run from the **project root** using `-m` to ensure correct imports.

### 1. Main experiments (Table 1 and Table 2)

Run each model on both benchmarks (5 attempts per problem, greedy decoding):

```bash
# HumanEval (164 problems) — all 5 models
python -m experiments.run_experiment --benchmark humaneval

# MBPP Sanitized (257 problems) — all 5 models
python -m experiments.run_experiment --benchmark mbpp
```

To run a single model:

```bash
python -m experiments.run_experiment --benchmark humaneval --models Llama-3.3-70B
```

Results are saved incrementally to `results/`, so interrupted runs can be resumed.

### 2. Prompt ablation (Table 6)

Compare minimal, explain-then-fix, and chain-of-thought repair strategies:

```bash
python -m experiments.run_ablation --models Llama-3.3-70B Llama-4-Scout-17B
```

### 3. Generate figures and tables

```bash
# Main figures (cumulative pass rates, error distributions, etc.)
python -m experiments.analyze_results

# Ablation comparison figure
python -m experiments.analyze_ablation
```

Figures are saved to `paper/figures/` in both PNG (300 dpi) and PDF formats.

### Available models

| Name | Groq Model ID | Architecture |
|------|---------------|-------------|
| `Llama-3.1-8B` | `llama-3.1-8b-instant` | 8B dense |
| `Llama-3.3-70B` | `llama-3.3-70b-versatile` | 70B dense |
| `Llama-4-Scout-17B` | `meta-llama/llama-4-scout-17b-16e-instruct` | 17B active, 16-expert MoE |
| `Llama-4-Maverick-17B` | `meta-llama/llama-4-maverick-17b-128e-instruct` | 17B active, 128-expert MoE |
| `Qwen3-32B` | `qwen/qwen3-32b` | 32B dense |

## Key Results

### HumanEval (164 problems)

| Model | Initial (R0) | Final (R4) | Gain |
|-------|-------------|------------|------|
| Llama 3.1 8B | 67.1% | 76.8% | +9.8pp |
| Llama 3.3 70B | 82.9% | 93.3% | +10.4pp |
| Llama 4 Scout 17B (16E) | 75.6% | 89.6% | +14.0pp |
| Llama 4 Maverick 17B (128E) | 87.2% | 93.9% | +6.7pp |
| Qwen3 32B | 87.8% | 92.7% | +4.9pp |

### MBPP Sanitized (257 problems)

| Model | Initial (R0) | Final | Gain |
|-------|-------------|-------|------|
| Llama 3.1 8B | 55.6% | 71.6% | +16.0pp |
| Llama 3.3 70B | 67.7% | 90.7% | +23.0pp |
| Llama 4 Scout 17B (16E) | 65.4% | 83.3% | +17.9pp |
| Llama 4 Maverick 17B (128E) | 72.0% | 92.6% | +20.6pp |
| Qwen3 32B | 70.8% | 88.3% | +17.5pp |

## Citation

```bibtex
@article{arimbur2026selfrepair,
  title={How Many Tries Does It Take? Iterative Self-Repair in LLM Code Generation Across Model Scales and Benchmarks},
  author={Arimbur, Johin Johny},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT
