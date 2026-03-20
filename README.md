# How Many Tries Does It Take?
## Iterative Self-Repair in LLM Code Generation Across Model Scales and Benchmarks

<p align="center">
  <a href="paper/main.pdf"><strong>Paper</strong></a> &nbsp;|&nbsp;
  <a href="#key-results"><strong>Results</strong></a> &nbsp;|&nbsp;
  <a href="#reproducing-experiments"><strong>Reproduce</strong></a> &nbsp;|&nbsp;
  <a href="#citation"><strong>Citation</strong></a>
</p>

This repository contains the code, data, and paper for **"How Many Tries Does It Take? Iterative Self-Repair in LLM Code Generation Across Model Scales and Benchmarks"** by Johin Johny Arimbur.

## Overview

We study **iterative self-repair** for LLM code generation: when a model's code fails, we feed the error back and ask it to try again, for up to five attempts. We evaluate five open-weight models (8B to 70B parameters) across dense and mixture-of-experts (MoE) architectures on HumanEval (164 problems) and MBPP Sanitized (257 problems).

**Main findings:**

- Self-repair **universally improves** pass rates for all five models on both benchmarks, with gains of +4.9 to +14.0 pp on HumanEval and +16.0 to +23.0 pp on MBPP.
- **Most gains come early.** The first two repair rounds capture 77 to 88% of the total achievable improvement.
- **Error type predicts repairability.** Syntax errors are fixed at >95% rates, type/name errors at 60 to 80%, and assertion errors (logic mistakes) at only ~41%.
- **Chain-of-thought repair prompting** yields up to +5.5 pp additional gain over minimal prompting for capable models.
- **Self-repair is more token-efficient** than independent resampling for capable models (32B+), using up to 54% fewer tokens.

## Models Evaluated

| Model | Parameters | Architecture | Type |
|-------|-----------|--------------|------|
| Llama 3.1 8B | 8B | Dense | Instruction-tuned |
| Llama 3.3 70B | 70B | Dense | Instruction-tuned |
| Llama 4 Scout 17B | 17B active | MoE (16 experts) | Instruction-tuned |
| Llama 4 Maverick 17B | 17B active | MoE (128 experts) | Instruction-tuned |
| Qwen3 32B | 32B | Dense | Instruction-tuned |

All models are accessed through the [Groq](https://console.groq.com/) free-tier API with greedy decoding (temperature = 0) for full reproducibility.

## Key Results

### HumanEval (164 problems)

| Model | Initial (R0) | Final (R4) | Gain |
|-------|-------------|------------|------|
| Llama 3.1 8B | 67.1% | 76.8% | +9.8 pp |
| Llama 3.3 70B | 82.9% | 93.3% | +10.4 pp |
| Llama 4 Scout 17B (16E) | 75.6% | 89.6% | +14.0 pp |
| Llama 4 Maverick 17B (128E) | 87.2% | 93.9% | +6.7 pp |
| Qwen3 32B | 87.8% | 92.7% | +4.9 pp |

### MBPP Sanitized (257 problems)

| Model | Initial (R0) | Final | Gain |
|-------|-------------|-------|------|
| Llama 3.1 8B | 55.6% | 71.6% | +16.0 pp |
| Llama 3.3 70B | 67.7% | 90.7% | +23.0 pp |
| Llama 4 Scout 17B (16E) | 65.4% | 83.3% | +17.9 pp |
| Llama 4 Maverick 17B (128E) | 72.0% | 92.6% | +20.6 pp |
| Qwen3 32B | 70.8% | 88.3% | +17.5 pp |

## Repository Structure

```
.
├── paper/
│   ├── main.tex              # LaTeX source
│   ├── main.pdf              # Compiled paper
│   ├── references.bib        # Bibliography
│   └── figures/              # All figures (PDF + PNG)
├── experiments/
│   ├── config.py             # Model definitions, paths, hyperparameters
│   ├── api_client.py         # Groq API client with retry logic
│   ├── self_repair.py        # Prompt construction and code extraction
│   ├── code_executor.py      # Sandboxed Python execution
│   ├── data_loader.py        # HumanEval / MBPP data loading
│   ├── run_experiment.py     # Main experiment runner
│   ├── run_ablation.py       # Prompt ablation study
│   ├── analyze_results.py    # Result analysis and visualization
│   └── analyze_ablation.py   # Ablation result analysis
├── results/
│   ├── *.json                # Per-model result files
│   ├── ablation/             # Prompt ablation results
│   ├── resampling/           # Independent resampling results
│   ├── livecodebench/        # LiveCodeBench results
│   └── qwen_thinking/        # Qwen3 thinking-mode results
├── requirements.txt
├── LICENSE
└── README.md
```

## Reproducing Experiments

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/) (free tier is sufficient)

### Installation

```bash
git clone https://github.com/Johin2/Self-repair.git
cd Self-repair
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your-key-here
```

### Running Experiments

All scripts are run from the project root using `-m` for correct imports.

**Main experiments (Tables I and II in the paper):**

```bash
# HumanEval -- all 5 models
python -m experiments.run_experiment --benchmark humaneval

# MBPP Sanitized -- all 5 models
python -m experiments.run_experiment --benchmark mbpp

# Single model
python -m experiments.run_experiment --benchmark humaneval --models Llama-3.3-70B
```

**Prompt ablation (Table VI):**

```bash
python -m experiments.run_ablation --models Llama-3.3-70B Llama-4-Scout-17B
```

**Generate figures and analysis:**

```bash
python -m experiments.analyze_results
python -m experiments.analyze_ablation
```

Figures are saved to `paper/figures/` in both PNG (300 dpi) and PDF formats. Results are saved incrementally to `results/`, so interrupted runs can be resumed.

### Available Model IDs

| CLI Name | Groq Model ID |
|----------|---------------|
| `Llama-3.1-8B` | `llama-3.1-8b-instant` |
| `Llama-3.3-70B` | `llama-3.3-70b-versatile` |
| `Llama-4-Scout-17B` | `meta-llama/llama-4-scout-17b-16e-instruct` |
| `Llama-4-Maverick-17B` | `meta-llama/llama-4-maverick-17b-128e-instruct` |
| `Qwen3-32B` | `qwen/qwen3-32b` |

## Citation

If you find this work useful, please cite:

```bibtex
@article{arimbur2026selfrepair,
  title={How Many Tries Does It Take? Iterative Self-Repair in LLM Code Generation Across Model Scales and Benchmarks},
  author={Arimbur, Johin Johny},
  journal={arXiv preprint},
  year={2026}
}
```

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgments

We thank [Groq](https://groq.com/) for providing free-tier API access that enabled all experiments in this work at zero monetary cost.
