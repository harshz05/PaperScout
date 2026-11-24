# PaperScout
Hybrid Retrieval-Augmented Generation (RAG) &amp; Methodology Matrix Engine over arXiv AI/ML Papers.

---

## 📌 Person 1: Data Processing and Retrieval Module

This module handles dataset ingestion, cleaning, category filtering, vector embedding generation, ChromaDB indexing, and semantic retrieval over recent AI/ML research literature.

### 📦 Quick Start & Integration Guide (For Person 2 / Streamlit App)

Person 2 can directly import and call the retrieval pipeline without re-implementing any vector search or ChromaDB logic:

```python
from scripts import retrieve_papers

# 1. Unrestricted semantic retrieval across all 5,496 papers
results = retrieve_papers(
    question="How do diffusion models improve parallel decoding in LLMs?",
    category=None,
    top_k=4
)

# 2. Category-filtered retrieval (e.g., Computer Vision)
cv_results = retrieve_papers(
    question="Object detection using vision transformers",
    category="cs.CV",
    top_k=4
)
```

### 📋 Function Specification

#### `retrieve_papers(question: str, category: Optional[str] = None, top_k: int = 4) -> List[Dict[str, str]]`

- **Parameters:**
  - `question` (*str*, required): The user's research query. Must be a non-empty string.
  - `category` (*Optional[str]*, default=`None`): Category filter. Supported values:
    - `"cs.CL"`: Computation and Language (NLP)
    - `"cs.LG"`: Machine Learning
    - `"cs.CV"`: Computer Vision
    - `None`: Searches across all target categories without filtering.
  - `top_k` (*int*, default=`4`): Number of relevant papers to retrieve. Must be a positive integer.

- **Returns:**
  A list of dictionaries with the exact structure:
  ```python
  [
      {
          "title": str,       # Clean title of the research paper
          "abstract": str,    # Full abstract text
          "arxiv_id": str,    # Unique arXiv identifier (e.g. '2604.26951v1')
          "category": str     # Primary arXiv category (e.g. 'cs.CL')
      },
      ...
  ]
  ```

- **Error Handling:**
  - Raises `ValueError` if `question` is empty or not a string.
  - Raises `ValueError` if `category` is not in `["cs.CL", "cs.LG", "cs.CV", None]`.
  - Raises `ValueError` if `top_k` is not a positive integer (rejects `bool`, non-int, `<= 0`).

---

### 🗄️ System Architecture & Data Pipeline

1. **Raw Ingestion**: `data/raw/arxiv_ai_ml_papers.csv` (7,701 arXiv papers from 2025–2026).
2. **Cleaning** (`scripts/data_cleaning.py`): Whitespace normalized, placeholder sentinel values replaced, empty titles/abstracts validated, exported to `data/processed/arxiv_papers_cleaned.csv`.
3. **Category Filtering** (`scripts/data_filtering.py`): Filtered to papers belonging to `cs.CL`, `cs.LG`, or `cs.CV` (including cross-listed research), yielding **5,496 curated papers** in `data/processed/arxiv_papers_filtered.csv`.
4. **Embeddings** (`scripts/generate_embeddings.py`): 384-dimensional unit-normalized embeddings generated via `all-MiniLM-L6-v2` and cached in `data/processed/paper_embeddings.npy`.
5. **Vector Store** (`scripts/store_embeddings.py`): Indexed into persistent ChromaDB at `chroma_db/` under collection `paperscout_papers` with cosine distance metric and Boolean category flags (`is_cs_cl`, `is_cs_lg`, `is_cs_cv`).
6. **Retrieval Engine** (`scripts/retrieve_papers.py`): Query encoding, HNSW pre-filtered vector similarity search, and standard response formatting.
