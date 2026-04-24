# Vietnamese Legal RAG System (Fully Fine-tuned)

An end-to-end **Vietnamese Legal RAG** project that builds and evaluates a production-style RAG stack:

- Lexical retriever (BM25),
- Dense retriever (fine-tuned bi-encoder),
- RRF fusion + cross-encoder reranking,
- Instruction-tuned generator (Gemma 4),
- Multi-layer evaluation (retrieval, citation quality, semantic quality, RAGAS).

The repository is notebook-first and captures the full experimentation lifecycle from data preprocessing to final pipeline evaluation.

## 1. Project Goal

The core objective is to improve answer quality for Vietnamese legal RAG by:

1. Increasing retrieval relevance (especially top-ranked passages),
2. Grounding generation in retrieved legal text,
3. Enforcing citation-friendly answer behavior,
4. Measuring quality with both retrieval and generation-centric metrics.

Furthermore, this project serves as a case study in building **domain-specific RAG systems** using edge models like **Gemma 4 E2B from Google DeepMind**, which was just released in **April 2026**.

## 2. What This Project Contains

### 2.1 Pipeline Phases

The project is organized into seven notebooks:

| Phase | Notebook                                    | Purpose                                                | Main Outputs                                                      |
| ----- | ------------------------------------------- | ------------------------------------------------------ | ----------------------------------------------------------------- |
| 1     | `01_Preprocess_and_EDA.ipynb`               | Clean and split legal QA data                          | `data/cleaned/*.parquet`                                          |
| 2     | `02_Finetune_Biencoder.ipynb`               | Fine-tune sentence embedding retriever                 | `models/biencoder/`                                               |
| 3     | `03_Generate_Embeddings_and_Indexing.ipynb` | Build FAISS and BM25 artifacts                         | `data/processed/*`                                                |
| 4     | `04_Finetune_Reranker.ipynb`                | Train cross-encoder reranker with mined hard negatives | `models/reranker/`, `data/finetune/ft_reranker_data.jsonl`        |
| 5     | `05_Retrieval_Evaluation.ipynb`             | Retrieval ablation and RRF tuning                      | `results/eval_results.csv`                                        |
| 6     | `06_Supervised_Finetune_Gemma4.ipynb`       | Create synthetic SFT data + tune generator             | `data/finetune/sft_data.jsonl`, `models/gemma4/` (runtime output) |
| 7     | `07_Full_RAG_Pipeline_Evaluation.ipynb`     | Full QA pipeline + generation/RAG evaluation           | `results/phase7_*` (runtime output)                               |

### 2.2 Current Repository Layout

```text
VN-LegalDoc-QA/
├─ data/
│  ├─ raw/
│  ├─ cleaned/
│  ├─ processed/
│  └─ finetune/
├─ models/
│  ├─ biencoder/
│  ├─ reranker/
│  └─ gemma4/
├─ notebooks/
│  ├─ 01_Preprocess_and_EDA.ipynb
│  ├─ 02_Finetune_Biencoder.ipynb
│  ├─ 03_Generate_Embeddings_and_Indexing.ipynb
│  ├─ 04_Finetune_Reranker.ipynb
│  ├─ 05_Retrieval_Evaluation.ipynb
│  ├─ 06_Supervised_Finetune_Gemma4.ipynb
│  └─ 07_Full_RAG_Pipeline_Evaluation.ipynb
└─ results/
   └─ eval_results.csv
```

## 3. Data Overview

### 3.1 Raw vs Processed Scale

| Dataset                            |    Rows | Notes                               |
| ---------------------------------- | ------: | ----------------------------------- |
| `data/raw/corpus.csv`              | 261,597 | Raw legal corpus passages           |
| `data/raw/train.csv`               | 119,456 | Raw question-context relevance data |
| `data/cleaned/corpus.parquet`      | 236,199 | Cleaned corpus                      |
| `data/cleaned/train_split.parquet` |  96,453 | Training split                      |
| `data/cleaned/val_split.parquet`   |  10,717 | Validation split                    |

### 3.2 Label Structure

Relevant-document count per query (cleaned data):

- Train: `{1: 87,813, 2: 8,050, 3: 590}`
- Val: `{1: 9,757, 2: 895, 3: 65}`

Most queries have one relevant passage; multi-document relevance is present and explicitly modeled.

## 4. Modeling Stack

### 4.1 Retriever (Dense)

- Base model: `AITeamVN/Vietnamese_Embedding`
- Fine-tuned model: `YuITC/vietnamese-embedding-vn-legal`
- Training objective: `CachedMultipleNegativesRankingLoss`
- Embedding dimension: **1024**
- Index: **FAISS HNSW** (`IndexHNSWFlat`, `M=32`, `efConstruction=200`, `efSearch=64`)

### 4.2 Retriever (Sparse)

- BM25 via `bm25s`
- Vietnamese tokenization via `underthesea`
- Indexed token parquet artifacts for corpus/train/val

### 4.3 Hybrid Retrieval and Reranking

- Candidate pool: top-30 from BM25 and dense retriever
- Fusion: Reciprocal Rank Fusion (RRF)
- Tuned RRF config (Phase 5): `k=10`, weights `[BM25=0.2, Dense=1.8]`
- Cross-encoder reranker: base `BAAI/bge-reranker-v2-m3`, tuned and published as `YuITC/bge-reranker-v2-m3-vn-legal`

### 4.4 Generator

- Base model: `unsloth/gemma-4-E4B-it`
- Target model ID: `YuITC/gemma4-e4b-it-vn-legal-16bit`
- SFT pipeline:
  - synthetic answer generation from legal contexts,
  - chat-format supervision data (`messages`),
  - LoRA fine-tuning with Unsloth + TRL.

## 5. Quantitative Results (Retrieval Ablation)

| Metric    |   BM25 | Dense (base) | Dense (tuned) | BM25 + Dense tuned + RRF | BM25 + Dense tuned + RRF + Rerank |
| --------- | -----: | -----------: | ------------: | -----------------------: | --------------------------------: |
| recall@1  | 0.2729 |       0.3404 |        0.5113 |               **0.5507** |                            0.5139 |
| ndcg@1    | 0.2846 |       0.3522 |        0.5342 |               **0.5698** |                            0.5371 |
| mrr@1     | 0.2846 |       0.3522 |        0.5342 |               **0.5698** |                            0.5371 |
| recall@3  | 0.4493 |       0.5244 |        0.7342 |                   0.7037 |                        **0.7375** |
| ndcg@3    | 0.3798 |       0.4526 |        0.6497 |                   0.6470 |                        **0.6529** |
| mrr@3     | 0.3646 |       0.4363 |        0.6341 |               **0.6387** |                            0.6373 |
| recall@5  | 0.5242 |       0.6016 |        0.8035 |                   0.7608 |                        **0.8059** |
| ndcg@5    | 0.4112 |       0.4850 |        0.6791 |                   0.6711 |                        **0.6820** |
| mrr@5     | 0.3822 |       0.4543 |        0.6495 |                   0.6519 |                        **0.6526** |
| recall@10 | 0.6149 |       0.6952 |        0.8710 |                   0.8378 |                        **0.8749** |
| ndcg@10   | 0.4413 |       0.5159 |        0.7017 |                   0.6970 |                        **0.7051** |
| mrr@10    | 0.3947 |       0.4668 |        0.6583 |               **0.6622** |                            0.6616 |

## 6. End-to-End Evaluation Design (Phase 7)

Phase 7 evaluates full RAG behavior beyond retrieval-only metrics:

- **BERTScore** (Vietnamese semantic overlap proxy),
- **Citation Accuracy** (precision/recall/F1 over cited retrieved passages),
- **RAGAS**:
  - Faithfulness
  - Answer Relevancy
  - Context Precision
  - Context Recall

Generated artifacts (when Phase 7 is executed):

- `results/phase7_summary.csv`
- `results/phase7_ragas_detailed.csv`
- `results/phase7_rag_results_final.parquet`
- `results/phase7_dashboard.png`
- `results/phase7_score_distributions.png`

> Note: Phase 7 requires setting `OPENAI_API_KEY` in the notebook for RAGAS components using OpenAI-backed evaluators.

## 7. How to Run

### 7.1 Environment

- Python: **3.13** (`.python-version`, `pyproject.toml`)
- Dependency manager: **uv** (project uses `pyproject.toml` and `uv.lock`)

```bash
uv sync
uv run jupyter lab
```

### 7.2 Recommended Execution Order

Run notebooks in strict order:

1. `01_Preprocess_and_EDA.ipynb`
2. `02_Finetune_Biencoder.ipynb`
3. `03_Generate_Embeddings_and_Indexing.ipynb`
4. `04_Finetune_Reranker.ipynb`
5. `05_Retrieval_Evaluation.ipynb`
6. `06_Supervised_Finetune_Gemma4.ipynb`
7. `07_Full_RAG_Pipeline_Evaluation.ipynb`

### 7.3 Practical Notes

- Notebooks assume a `workspace/...` path convention (Kaggle/Colab style).  
  If running locally, keep paths consistent or adapt path roots once.
- GPU resources are strongly recommended for Phases 2, 4, 6, and 7.
- Large artifact generation (embeddings/models) requires substantial disk space.

## 8. Engineering Decisions

The system is deliberately designed as a **hybrid retrieval + constrained generation** QA stack:

- Dense embeddings capture semantic intent in Vietnamese legal language.
- BM25 preserves lexical precision on statutory terms and formal legal phrasing.
- RRF stabilizes retrieval across query types.
- Cross-encoder reranking improves top-ranked evidence quality.
- Citation-oriented prompting enforces auditable, source-grounded answers.
- Multi-perspective evaluation avoids overfitting to retrieval-only metrics.

## 9. Current Limitations

1. Notebook-first implementation (not yet packaged into reusable Python modules/services).
2. End-to-end runtime can be expensive in GPU memory/time.
3. Phase 7 RAGAS path depends on external API credentials.

## 10. Next Engineering Steps

1. Refactor notebook logic into a modular package (`src/`) with CLI entry points.
2. Add deterministic experiment configs (YAML) and run tracking.
3. Introduce automated regression checks for retrieval/generation metrics.
4. Add inference API + serving profile (latency, memory, throughput benchmarks).
5. Expand legal-domain robustness tests (multi-hop, contradictory evidence, citation strictness).

---

If you are evaluating this work for practical deployment, start with Phase 5 metrics to verify retrieval behavior, then run Phase 7 with your target compliance and citation requirements.
