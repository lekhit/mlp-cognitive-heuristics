# MLP Cognitive Heuristics: Quantized MLP-LoRA Training for Resource-Constrained Diagnostics

This repository contains the full source code, datasets, and research paper for an end-to-end experiment demonstrating that **targeting Multilayer Perceptron (MLP) feed-forward layers with QLoRA can train stable cognitive search heuristics** in language models, bypassing context window hardware bottlenecks.

By teaching a small model (\texttt{Qwen2.5-3B-Instruct}) *how* to plan and navigate directories step-by-step, it achieves equivalent accuracy to a model more than double its size (7B), while executing **72x faster** than host CPU brute-forcing and completely avoiding GPU memory crashes.

---

## 🔬 Core Scientific Rationale
Ingesting massive logs brute-force inside large context windows suffers from three major issues: quadratic activation memory scale $O(N^2)$, distraction (losing the needle), and heavy execution latency.

Interpretability research established that while attention heads primarily dictate *where* a model shifts its focus, the **MLP feed-forward networks** act as key-value databases storing associations and planning structures. By targeting MLP layers (\texttt{gate\_proj}, \texttt{up\_proj}, \texttt{down\_proj}) with LoRA, we bake structured cognitive search Monologues (\texttt{Thought Process:} $\rightarrow$ \texttt{Next Action:}) directly into the model's weights. The resulting agent dynamically loads files from disk only when requested, keeping the active context window under 2.7k tokens.

---

## 📊 A/B/C/D Performance Comparison

We evaluated three computational architectures on a complex **13,546-token multi-file diagnostics task** split across 5 log files containing a SQL Injection exploit vector and subsequent root privilege escalation:

| Metric | Scenario A: Baseline 3B (Brute Force GPU Ingest) | Scenario B: Heuristic 3B (Dynamic MLP-LoRA Agent) | Scenario C: Base 7B Model (CPU Brute Ingestion) | Scenario D: Base 7B Model (Estimated GPU Brute)* |
| :--- | :--- | :--- | :--- | :--- |
| **Search Accuracy** | **FAIL** (0% - Crashed) | **PASS** (100% success) | **PASS** (100% success) | **PASS** (100% success) |
| **Peak Memory Usage**| >11.0 GB VRAM (OOM) | **<1.5 GB VRAM** (Peak 2,741 tokens) | **18.2 GB System CPU RAM** | **~14.5 GB VRAM** (4-bit + KV Cache) |
| **Time to First Token (TTFT)**| N/A | **0.84s** | ~2750.00s (Prefill bottleneck) | **~0.35s** (FlashAttention prefill) |
| **Total Computation Time** | N/A (Crashed immediately) | **39.10 seconds** 🚀 | **2,813.12 seconds (46.89 min)** 🐢 | **~4.50 seconds** ⚡ |
| **Compute Device** | GPU (RTX 2080 Ti) | GPU (RTX 2080 Ti) | Host CPU (24 Processors) | High-VRAM GPU (e.g. RTX 3090/A100)* |
| **Task Outcome** | **CRITICAL OOM CRASH** | **SUCCESS (Perfect Trace)** | **SUCCESS (Slow/Brute Force)** | **SUCCESS (Requires >16GB VRAM)** |

*\*Note: Scenario D represents estimated performance on a high-VRAM GPU (e.g., 24GB RTX 3090/4090 or 80GB A100) running the 7B model in 4-bit with GPU FlashAttention. While extremely fast (~4.5s), it is completely unsupported on our 11GB RTX 2080 Ti card, which immediately OOMs under brute-force context ingestion.*

---

## 📂 Repository Structure

```text
mlp_heuristic_project/
│
├── requirements.txt            # Package stack (TRL, PEFT, bitsandbytes, datasets)
├── config.py                   # Central hyperparameters and model directories
├── model_manager.py            # Quantized model loading (bfloat16) and GPU cache management
├── heuristic_builder.py        # Logs active interactions and compiles ChatML datasets
├── micro_trainer.py            # Quantized SFT sleep-cycle training using TRL SFTConfig
├── agent_pipeline.py           # Phase 1 simulation loop & micro-training trigger
├── eval_haystack.py            # Phase 1 A/B haystack benchmark evaluation
│
├── agent_extreme_context.py    # Phase 2 13.5k-token logs generator and Scenario A/B agent loop
├── eval_large_model.py         # Phase 2 Scenario C base 7B model CPU evaluation
│
├── mlp_heuristic_paper.tex     # LaTeX source for the ACM Transaction styled research paper
├── mlp_heuristic_paper.pdf     # Pre-compiled ACM Transaction publication PDF
│
├── phase1_results.md           # Log traces and metrics for Phase 1 A/B benchmark
└── phase2_results.md           # Log traces and metrics for Phase 2 A/B/C benchmark
```

---

## 🛠️ Getting Started

### 1. Set Up Environment & Install Dependencies
Ensure you have Python 3.10+ and a CUDA-capable GPU (Turing architecture or newer).
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Execute Phase 1 (QLoRA Training & Benchmarking)
This generates baseline traces, runs active diagnostics steps, triggers the micro-training sleep cycle on GPU, and runs the A/B haystack evaluation:
```bash
# Run simulation loop and training
python agent_pipeline.py

# Run Phase 1 A/B haystack benchmark
python eval_haystack.py
```

### 3. Execute Phase 2 (Extreme Context & Large Model Benchmarking)
This creates the massive 13,546-token logs database, triggers Scenario A (VRAM OOM crash), executes Scenario B (3B Heuristic Agent), and Scenario C (7B CPU brute-forcing):
```bash
# Run Scenario A and Scenario B
python agent_extreme_context.py

# Run Scenario C (requires ~14GB available CPU RAM)
python eval_large_model.py
```

---

## 📄 Academic Research Paper
We have prepared a complete peer-reviewed style research paper titled:  
**"Training Cognitive Search Heuristics via Multilayer Perceptron (MLP) LoRA Adapters for Resource-Constrained Agentic Workflows"**

The publication-grade PDF is available in this repository: [mlp_heuristic_paper.pdf](./mlp_heuristic_paper.pdf).  
You can view or compile the LaTeX source code using: [mlp_heuristic_paper.tex](./mlp_heuristic_paper.tex).
