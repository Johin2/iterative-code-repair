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

We study **iterative self-repair** for LLM code generation: when a model's code fails, we feed the error back and ask it to try again, for up to five attempts. We evaluate seven models from three families (Meta, Alibaba, Google), spanning open-weight and proprietary models, dense and mixture-of-experts (MoE) architectures, on HumanEval (164 problems) and MBPP Sanitized (257 problems).

**Main findings:**

- Self-repair **universally improves** pass rates for all seven models on both benchmarks, with gains of +4.9 to +17.1 pp on HumanEval and +16.0 to +30.0 pp on MBPP.
- **Gemini 2.5 Flash achieves the highest final pass rates**: 96.3% on HumanEval and 93.8% on MBPP.
- **Most gains come early.** The first two repair rounds capture 76 to 95% of the total achievable improvement.
- **Error type predicts repairability.** Name errors are repaired at ~77%, syntax errors at ~66%, and assertion errors (logic mistakes) at only ~45%.
- **Chain-of-thought repair prompting** yields up to +5.5 pp additional gain over minimal prompting for capable models.
- **Self-repair is more token-efficient** than independent resampling for capable models (32B+), using up to 54% fewer tokens.

## Models Evaluated

| Model | Parameters | Architecture | Provider |
|-------|-----------|--------------|----------|
| Llama 3.1 8B | 8B | Dense | Groq |
| Llama 3.3 70B | 70B | Dense | Groq |
| Llama 4 Scout 17B | 17B active | MoE (16 experts) | Groq |
| Llama 4 Maverick 17B | 17B active | MoE (128 experts) | Groq |
| Qwen3 32B | 32B | Dense | Groq |
| Gemini 2.5 Flash | Undisclosed | Undisclosed | Vertex AI |
| Gemini 2.5 Pro | Undisclosed | Undisclosed | Vertex AI |

Open-weight models are accessed through the [Groq](https://console.groq.com/) free-tier API. Gemini models are accessed via [Google Cloud Vertex AI](https://cloud.google.com/vertex-ai). All models use greedy decoding (temperature = 0) for reproducibility.

## Key Results

### HumanEval (164 problems)

| Model | Initial (R0) | Final (R4) | Gain |
|-------|-------------|------------|------|
| Llama 3.1 8B | 67.1% | 76.8% | +9.8 pp |
| Llama 3.3 70B | 82.9% | 93.3% | +10.4 pp |
| Llama 4 Scout 17B (16E) | 75.6% | 89.6% | +14.0 pp |
| Llama 4 Maverick 17B (128E) | 87.2% | 93.9% | +6.7 pp |
| Qwen3 32B | 87.8% | 92.7% | +4.9 pp |
| Gemini 2.5 Flash | 86.6% | 96.3% | +9.8 pp |
| Gemini 2.5 Pro | 73.2% | 90.2% | +17.1 pp |

### MBPP Sanitized (257 problems)

| Model | Initial (R0) | Final (R4) | Gain |
|-------|-------------|------------|------|
| Llama 3.1 8B | 55.6% | 71.6% | +16.0 pp |
| Llama 3.3 70B | 67.7% | 90.7% | +23.0 pp |
| Llama 4 Scout 17B (16E) | 65.4% | 83.3% | +17.9 pp |
| Llama 4 Maverick 17B (128E) | 72.0% | 92.6% | +20.6 pp |
| Qwen3 32B | 70.8% | 88.3% | +17.5 pp |
| Gemini 2.5 Flash | 63.8% | 93.8% | +30.0 pp |
| Gemini 2.5 Pro | 66.5% | 92.2% | +25.7 pp |

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
│   ├── vertex_client.py      # Vertex AI client for Gemini models
│   ├── self_repair.py        # Prompt construction and code extraction
│   ├── code_executor.py      # Sandboxed Python execution
│   ├── data_loader.py        # HumanEval / MBPP data loading
│   ├── run_experiment.py     # Main experiment runner (Groq)
│   ├── run_vertex.py         # Main experiment runner (Vertex AI)
│   ├── run_ablation.py       # Prompt ablation study
│   ├── analyze_results.py    # Result analysis and visualization
│   └── analyze_ablation.py   # Ablation result analysis
├── results/
│   ├── *.json                # Per-model result files
│   ├── vertex/               # Gemini model results
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
- A [Groq API key](https://console.groq.com/) (free tier is sufficient) for open-weight models
- A [Google Cloud project](https://cloud.google.com/) with Vertex AI enabled for Gemini models

### Installation

```bash
git clone https://github.com/Johin2/iterative-code-repair.git
cd iterative-code-repair
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your-key-here
VERTEX_PROJECT=your-gcp-project-id
VERTEX_LOCATION=us-central1
```

### Running Experiments

All scripts are run from the project root using `-m` for correct imports.

**Main experiments (Tables I and II in the paper):**

```bash
# HumanEval -- all open-weight models (Groq)
python -m experiments.run_experiment --benchmark humaneval

# MBPP Sanitized -- all open-weight models (Groq)
python -m experiments.run_experiment --benchmark mbpp

# Single model
python -m experiments.run_experiment --benchmark humaneval --models Llama-3.3-70B

# Gemini models (Vertex AI)
python -m experiments.run_vertex --benchmark humaneval
python -m experiments.run_vertex --benchmark mbpp
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

Gemini models (`gemini-2.5-flash`, `gemini-2.5-pro`) are accessed via Vertex AI and configured in `experiments/vertex_client.py`.

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

We thank [Groq](https://groq.com/) for providing free-tier API access that enabled the open-weight model experiments, and Google Cloud for Vertex AI credits used for the Gemini experiments.
