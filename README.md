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
│   ├── run_experiment.py     # Main experiment runner
│   ├── run_ablation.py       # Prompt ablation study
│   ├── analyze_results.py    # Result analysis and visualization
│   ├── analyze_ablation.py   # Ablation result analysis
│   ├── self_repair.py        # Prompt building and code extraction
│   ├── code_executor.py      # Sandboxed Python execution
│   ├── data_loader.py        # HumanEval/MBPP data loading
│   ├── config.py             # Configuration
│   └── requirements.txt      # Python dependencies
├── results/
│   ├── *.json                # Per-model result files
│   └── ablation/             # Ablation study results
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- Groq API key (free tier)

### Installation

```bash
pip install -r experiments/requirements.txt
```

### Environment Variables

```bash
export GROQ_API_KEY="your-key"
```

### Data

HumanEval and MBPP datasets should be placed in `experiments/data/`:
- `experiments/data/HumanEval.jsonl`
- `experiments/data/MBPP_sanitized.jsonl`

## Running Experiments

### Run main experiments

```bash
cd experiments
python run_experiment.py
```

### Run ablation study

```bash
python run_ablation.py
```

### Analyze results

```bash
python analyze_results.py
python analyze_ablation.py
```

## Key Results

### HumanEval

| Model | Initial (R0) | Final (R4) | Gain |
|-------|-------------|------------|------|
| Llama 3.1 8B | 67.1% | 76.8% | +9.8pp |
| Llama 3.3 70B | 82.9% | 93.3% | +10.4pp |
| Llama 4 Scout 17B (16E) | 75.6% | 89.6% | +14.0pp |
| Llama 4 Maverick 17B (128E) | 87.2% | 93.9% | +6.7pp |
| Qwen3 32B | 87.8% | 92.7% | +4.9pp |

### MBPP Sanitized

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
