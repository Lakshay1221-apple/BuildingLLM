# Building Large Language Models From Scratch

A pedagogical, first-principles implementation of a **GPT-style decoder-only Transformer** built from scratch in pure **PyTorch**.

This repository documents the step-by-step engineering journey of constructing an autoregressive Large Language Model (LLM)—from text preprocessing, Byte Pair Encoding (BPE), and causal self-attention mechanisms to full Transformer block assembly and autoregressive text generation.

---

## 📑 Table of Contents

- [Architectural Overview](#-architectural-overview)
- [Repository Structure](#-repository-structure)
- [Deep Dive into Modules](#-deep-dive-into-modules)
  - [1. Tokenization & Data Pipelines (`tokenizer/`)](#1-tokenization--data-pipelines-tokenizer)
  - [2. Attention Mechanisms (`Attention/`)](#2-attention-mechanisms-attention)
  - [3. Normalization Concepts (`Normalization/`)](#3-normalization-concepts-normalization)
  - [4. Core GPT Architecture (`LLM/`)](#4-core-gpt-architecture-llm)
- [GPT-2 Architecture Specifications](#-gpt-2-architecture-specifications)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation & Environment Setup](#installation--environment-setup)
  - [Running the Notebooks](#running-the-notebooks)
- [Verification & Testing](#-verification--testing)
- [Roadmap & Next Steps](#-roadmap--next-steps)

---

## 🏛 Architectural Overview

```
Prompt Text: "Every effort moves you"
       │
       ▼
[ Tokenizer (BPE / tiktoken) ] ────────► Token IDs [B, T]
       │
       ├────────────────────────────────────────┐
       ▼                                        ▼
[ Token Embedding Layer ]              [ Positional Embedding Layer ]
  (Vocab: 50,257 -> 768)                 (Context: 1,024 -> 768)
       │                                        │
       └───────────────────┬────────────────────┘
                           ▼
                 [ Element-wise Sum ] ────────► Input Tensor [B, T, 768]
                           │
                           ▼
                    [ Dropout (0.1) ]
                           │
 ┌─────────────────────────┴─────────────────────────┐
 │ Transformer Block (Repeated L = 12 times)         │
 │                                                   │
 │    ┌───► [ LayerNorm 1 ]                          │
 │    │          │                                   │
 │    │          ▼                                   │
 │    │     [ Causal Multi-Head Self-Attention ]     │
 │    │          │                                   │
 │    │          ▼                                   │
 │    │     [ Dropout ]                              │
 │    │          │                                   │
 │    └───► ( + ) Residual / Shortcut Connection     │
 │               │                                   │
 │    ┌───► [ LayerNorm 2 ]                          │
 │    │          │                                   │
 │    │          ▼                                   │
 │    │     [ Feed-Forward Network (GELU) ]          │
 │    │       (768 -> 3,072 -> 768)                  │
 │    │          │                                   │
 │    │          ▼                                   │
 │    │     [ Dropout ]                              │
 │    │          │                                   │
 │    └───► ( + ) Residual / Shortcut Connection     │
 └─────────────────────────┬─────────────────────────┘
                           │
                           ▼
                 [ Final LayerNorm ]
                           │
                           ▼
            [ Linear LM Head (768 -> 50,257) ]
                           │
                           ▼
               Next-Token Logits [B, T, 50257]
                           │
                           ▼
         [ Greedy Decoding / Softmax + Argmax ]
                           │
                           ▼
               Generated Token IDs [B, T+1]
```

---

## 📂 Repository Structure

```text
BuildingLargeLanguageModel/
├── pyproject.toml              # Build system and project metadata
├── requirements.txt            # Python dependencies (PyTorch, tiktoken, etc.)
├── README.md                   # Project documentation
└── src/
    └── buildinglargelanguagemodel/
        ├── data/
        │   └── the-verdict.txt # Short story corpus (Edith Wharton) for tokenization
        ├── tokenizer/
        │   ├── tokenizer.ipynb # Naive regex splitting, vocab construction, V1 & V2 tokenizers
        │   └── Bpt.ipynb       # Byte Pair Encoding (BPE), GPTDatasetV1 & sliding window sampling
        ├── Attention/
        │   ├── self_attention_mech.ipynb       # Scaled dot-product self-attention
        │   ├── causal_self_attention.ipynb     # Causal masking (lower triangular) & dropout
        │   └── multihead_self_attention.ipynb  # Multi-head attention (MHA) parallel projections
        ├── Normalization/
        │   └── norm.ipynb      # Statistical normalization intuition & LayerNorm vs. BatchNorm
        └── LLM/
            └── llm_architecture.ipynb # Unified 124M GPTModel, Transformer blocks, and text generation
```

---

## 🔍 Deep Dive into Modules

### 1. Tokenization & Data Pipelines (`tokenizer/`)
- **[tokenizer.ipynb](src/buildinglargelanguagemodel/tokenizer/tokenizer.ipynb)**:
  - Demonstrates why simple whitespace splits fail (punctuation adheres to words).
  - Implements regular expressions (`r'([,.:;?_!"()\']|--|\s)'`) to separate words and punctuation symbols.
  - Builds `SimpleTokenizerV1` (`encode` and `decode` lookup tables).
  - Builds `SimpleTokenizerV2` supporting special tokens:
    - `<|unk|>`: Handles out-of-vocabulary (OOV) tokens safely without `KeyError`.
    - `<|endoftext|>`: Acts as document delimiter across concatenated texts.
- **[Bpt.ipynb](src/buildinglargelanguagemodel/tokenizer/Bpt.ipynb)**:
  - Explores OpenAI's `tiktoken` BPE tokenizer with GPT-2 vocabulary (50,257 tokens).
  - Implements autoregressive data sampling: shifts inputs $x$ and targets $y$ by 1 token.
  - Implements `GPTDatasetV1` and `create_dataloader_v1` using PyTorch `Dataset` and `DataLoader` for windowed sequence batching with configurable `max_length` and `stride`.
  - Implements token and learned positional embeddings ($\mathbf{X}_{\text{input}} = \mathbf{X}_{\text{token}} + \mathbf{X}_{\text{pos}}$).

### 2. Attention Mechanisms (`Attention/`)
- **[self_attention_mech.ipynb](src/buildinglargelanguagemodel/Attention/self_attention_mech.ipynb)**:
  - Scaled dot-product formulation:
    $$\mathbf{S} = \mathbf{Q}\mathbf{K}^\top, \quad \mathbf{A} = \text{softmax}\left(\frac{\mathbf{S}}{\sqrt{d_k}}\right), \quad \mathbf{Z} = \mathbf{A}\mathbf{V}$$
  - Dimensional analysis with linear projections $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$.
- **[causal_self_attention.ipynb](src/buildinglargelanguagemodel/Attention/causal_self_attention.ipynb)**:
  - Explains the autoregressive property: predicting token $t_i$ cannot condition on future tokens $t_{>i}$.
  - Implements lower-triangular masks (`torch.tril`) and sets future positions to $-\infty$ via `.masked_fill()`.
  - Applies attention dropout to prevent co-adaptation of attention heads.
- **[multihead_self_attention.ipynb](src/buildinglargelanguagemodel/Attention/multihead_self_attention.ipynb)**:
  - Efficient tensor reshaping: $[B, T, d_{\text{model}}] \xrightarrow{\text{view}} [B, T, h, d_{\text{head}}] \xrightarrow{\text{transpose}} [B, h, T, d_{\text{head}}]$.
  - Runs all heads in parallel via batched matrix multiplication (`@`).
  - Registers the causal mask buffer (`torch.triu(..., diagonal=1)`) to avoid redundant tensor allocations during execution.

### 3. Normalization Concepts (`Normalization/`)
- **[norm.ipynb](src/buildinglargelanguagemodel/Normalization/norm.ipynb)**:
  - Mathematical breakdown of sample mean, sample variance, and standardization:
    $$\mu = \frac{1}{d} \sum x_i, \quad \sigma^2 = \frac{1}{d} \sum (x_i - \mu)^2, \quad \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
  - Explains why Batch Normalization is unsuitable for autoregressive sequence modeling with variable sequence lengths, and why Layer Normalization is required.

### 4. Core GPT Architecture (`LLM/`)
- **[llm_architecture.ipynb](src/buildinglargelanguagemodel/LLM/llm_architecture.ipynb)**:
  - **Activation**: GELU (Gaussian Error Linear Unit) implemented from scratch and benchmarked against ReLU.
  - **FeedForward**: Expansion to $4 \times d_{\text{model}}$ ($768 \rightarrow 3072 \rightarrow 768$).
  - **Residuals**: Empirical demonstration proving shortcuts preserve gradient flow across deep networks.
  - **LayerNorm**: Custom layer normalization module operating over the feature dimension (`dim=-1`) with learnable scale $\gamma$ and shift $\beta$.
  - **TransformerBlock**: Pre-LayerNorm block integrating attention, feed-forward, dropout, and residuals.
  - **GPTModel**: Full architecture connecting token/position embeddings, 12 transformer blocks, final LayerNorm, and output projection head.
  - **Text Generation**: Implements `generate_text_simple` for autoregressive greedy decoding.

---

## ⚙️ GPT-2 Architecture Specifications

The model follows the architecture of **GPT-2 Small (124M)**:

| Hyperparameter | Configuration Value | Description |
| :--- | :--- | :--- |
| `vocab_size` | **50,257** | Subword vocabulary entries via `tiktoken` (GPT-2 BPE) |
| `context_length` | **1,024** | Maximum context sequence length ($T_{\max}$) |
| `emb_dim` | **768** | Hidden embedding dimension ($d_{\text{model}}$) |
| `n_heads` | **12** | Attention heads ($d_{\text{head}} = 768 / 12 = 64$) |
| `n_layers` | **12** | Stacked Transformer blocks ($L$) |
| `drop_rate` | **0.1** | Dropout probability across attention and residuals |
| `qkv_bias` | **False** | Bias disabled in Q, K, V projections (GPT-2 style) |
| **Total Parameters** | **~163M** | Without weight tying (~124M with weight tying) |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- PyTorch 2.0+

### Installation & Environment Setup

Using `uv` (recommended):
```bash
# Clone the repository
git clone https://github.com/Lakshay1221-apple/BuildingLargeLanguageModel.git
cd BuildingLargeLanguageModel

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Or using standard `pip`:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running the Notebooks

Launch JupyterLab to interact with any chapter:
```bash
jupyter lab
```

---

## 🧪 Verification & Testing

Every notebook in this repository is designed to be **sequentially executable** from the first cell to the last in a clean kernel.

You can verify all notebooks programmatically from the CLI:
```bash
# Run and verify the complete LLM architecture notebook
python3 -c "
import nbformat
from nbclient import NotebookClient

with open('src/buildinglargelanguagemodel/LLM/llm_architecture.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name='python3')
client.execute()
print('All cells in llm_architecture.ipynb executed successfully!')
"
```

Expected verification output:
- **Input Batch Shape**: `[2, 9]`
- **Logits Tensor**: `[2, 9, 50257]`
- **Generated Text Sequence**: Successfully appends greedy tokens and decodes back to text.

---

## 🗺 Roadmap & Next Steps

- [x] Chapter 2: Text Preprocessing & Byte Pair Encoding
- [x] Chapter 3: Scaled Dot-Product, Causal & Multi-Head Self-Attention
- [x] Chapter 4: Implementing the GPT-2 (124M) Model Architecture & Text Generation
- [ ] Chapter 5: Pretraining on Unlabeled Text Data (Cross-Entropy Loss & AdamW Optimization)
- [ ] Chapter 6: Loading Pretrained Weights from OpenAI / Hugging Face
- [ ] Chapter 7: Instruction Fine-Tuning & Task Alignment (SFT)
- [ ] Chapter 8: Alignment with Direct Preference Optimization (DPO) / RLHF

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
