<div align="center">

# AgriIR

### A Scalable Framework for Domain-Specific Knowledge Retrieval

[![Paper](https://img.shields.io/badge/ECIR%202026-IR%20for%20Good-blue?style=for-the-badge&logo=springer)](https://ecir2026.eu/calls/call-for-ir-for-good-papers)
[![Demo](https://img.shields.io/badge/Demo-Watch%20Video-red?style=for-the-badge&logo=youtube)](https://bit.ly/AgriIR)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8--3.12-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

[![Ollama](https://img.shields.io/badge/Ollama-Supported-black?style=flat-square&logo=ollama)](https://ollama.ai)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-orange?style=flat-square&logo=meta)](https://github.com/facebookresearch/faiss)
[![Flask](https://img.shields.io/badge/Flask-Web%20UI-lightgrey?style=flat-square&logo=flask)](https://flask.palletsprojects.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-yellow?style=flat-square&logo=huggingface)](https://huggingface.co)

<br/>

**AgriIR** is a modular RAG framework that achieves **ChatGPT-4o level performance** using **27B parameter models** while providing **deterministic citations** — designed for agricultural information access in resource-constrained environments.

[Getting Started](#-quick-start) •
[Documentation](#-documentation) •
[Benchmarks](#-benchmark-results) •
[Contributing](#-contributing)

</div>

---

## Highlights

| Feature | Description |
|:--------|:------------|
| **Statistical Parity with GPT-4o** | Composite score 0.820 vs 0.840 (p=0.493, not significant) |
| **73% Perfect Citations** | Deterministic citation system vs 0% for baseline LLMs |
| **6-Stage Pipeline** | Modular architecture with temperature stratification |
| **Indian Language Support** | Hindi, Tamil, Telugu, Kannada, Malayalam voice I/O |
| **Low Resource Friendly** | Runs on 8GB RAM with Gemma3:1B |

---

## Quick Start

```bash
git clone https://github.com/Shuvam-Banerji-Seal/AgriIR-A-Scalable-Framework-for-Domain-Specific-Knowledge-Retrieval.git
cd AgriIR-A-Scalable-Framework-for-Domain-Specific-Knowledge-Retrieval
chmod +x install_agriir.sh && ./install_agriir.sh
./start_agriir.sh
```
Open **http://localhost:5000**

---

## Architecture

```
USER QUERY
    ▼
┌────────────────────────────────────────────────────────────────────┐
│ Stage 1: Query Refinement (T=0.1)        → Gemma3:1B               │
│ Stage 2: Sub-Query Decomposition (T=0.5) → 3-5 sub-queries         │
│ Stage 3: Parallel Retrieval              → FAISS + DuckDuckGo      │
│ Stage 4: Domain-Agent Enhancement        → Specialist selection    │
│ Stage 5: Answer Synthesis (T=0.2)        → Gemma3:27B              │
│ Stage 6: Citation Insertion              → Similarity > 0.75       │
└────────────────────────────────────────────────────────────────────┘
    ▼
CITED RESPONSE
```

---

## Benchmark Results

Evaluated on **191 queries** by **30 annotators**. Composite = 0.7×(Answer/4) + 0.3×(Citation/2)

| Model | Answer | Good% | Citations | Composite |
|:------|:------:|:-----:|:---------:|:---------:|
| ChatGPT-4o | 3.36 | 88.5% | — | **0.840** |
| **AgriIR (Gemma3:27B)** | 3.24 | 86.9% | 73.0% | **0.820** |
| Gemini 2.5 Flash | 3.12 | 78.7% | — | 0.779 |
| GPT-OSS-120B | 2.82 | 70.5% | — | 0.705 |

**Statistical parity with ChatGPT-4o** (p=0.493) while providing **73% perfect citations** vs 0% for baselines.

---

## Installation

| Requirement | Minimum | Recommended |
|:--|:--|:--|
| Python | 3.8 | 3.11+ |
| RAM | 8 GB | 32 GB |
| GPU | Optional | 8GB+ VRAM |

```bash
./install_agriir.sh                      # Automated setup
./install_agriir.sh --download-embeddings # With pre-built embeddings (~40GB)
./install_agriir.sh --no-rag              # Web-search only mode
```

<details>
<summary><b>Manual Installation</b></summary>

```bash
python3 -m venv agriir_env && source agriir_env/bin/activate
pip install -r requirements.txt
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull gemma3:1b llama3.2:3b
```
</details>

---

## Usage

```bash
./start_agriir.sh                                    # Auto-select mode
python3 agri_bot_searcher/src/enhanced_web_ui.py     # RAG mode
python3 agri_bot_searcher/src/enhanced_voice_web_ui.py # Voice mode
```

**API:**
```python
requests.post('http://localhost:5000/api/query', json={'query': 'Best wheat practices in Punjab?'})
```

---

## Project Structure

```
AgriIR/
├── agri_bot_searcher/           # RAG pipeline, Web UI, API, Voice
├── embedding_generator/         # Document embedding system
├── organized_database_creation/ # Agentic data curation (32 agents)
├── install_agriir.sh / start_agriir.sh
└── requirements.txt
```

---

## Models

| Model | VRAM | Use |
|:------|:----:|:----|
| `gemma3:1b` | 1.6GB | Query processing |
| `llama3.2:3b` | 2GB | Default synthesis |
| `gemma3:27b` | 16GB | High-quality synthesis |
| `Qwen3-Embedding-8B` | 5GB | Embeddings |

**Voice:** English, Hindi, Tamil, Telugu, Kannada, Malayalam

---

## Citation

```bibtex
@inproceedings{seal2026agriir,
  title     = {AgriIR: A Scalable Framework for Domain-Specific Knowledge Retrieval},
  author    = {Seal, Shuvam Banerji and Poddar, Aheli and Mishra, Alok and Roy, Dwaipayan},
  booktitle = {ECIR 2026, IR for Good Track},
  year      = {2026},
  publisher = {Springer},
  series    = {LNCS}
}
```

---

## Authors

**Shuvam Banerji Seal** (IISER Kolkata) • **Aheli Poddar** (IEM Kolkata) • **Alok Mishra** (IISER Kolkata) • **Dwaipayan Roy** (IISER Kolkata)

---

## License

MIT License — see [LICENSE](LICENSE)

**Acknowledgments:** [Ollama](https://ollama.ai) • [FAISS](https://github.com/facebookresearch/faiss) • [Qwen](https://github.com/QwenLM/Qwen) • [AI4Bharat](https://ai4bharat.iitm.ac.in) • [ICAR](https://icar.org.in) • [FAO](https://www.fao.org)

---

<div align="center">

**[Demo](https://bit.ly/AgriIR)** • **[Issues](https://github.com/Shuvam-Banerji-Seal/AgriIR-A-Scalable-Framework-for-Domain-Specific-Knowledge-Retrieval/issues)**

</div>
