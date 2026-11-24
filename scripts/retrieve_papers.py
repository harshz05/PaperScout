"""
scripts/retrieve_papers.py
Stage 6 & 7: Query Embedding, Top-K Paper Retrieval, and Category Filtering for PaperScout.

Responsibilities:
1. Connect to the persistent ChromaDB database at chroma_db/.
2. Access the collection 'paperscout_papers'.
3. Load the SentenceTransformer model 'all-MiniLM-L6-v2' once and reuse it across queries.
4. Implement retrieve_papers(question, category=None, top_k=4) returning a list of dicts:
   [
       {
           "title": "Paper title",
           "abstract": "Paper abstract",
           "arxiv_id": "2501.12345",
           "category": "cs.LG"
       }
   ]
5. Support category-based pre-filtering on:
   - 'cs.CL' -> where={'is_cs_cl': True}
   - 'cs.LG' -> where={'is_cs_lg': True}
   - 'cs.CV' -> where={'is_cs_cv': True}
   - None    -> unrestricted search across all categories
6. Cross-listed papers are properly matched via Boolean metadata flags.
7. Filtering occurs before final top-k selection inside ChromaDB's HNSW query.
8. Validate question, category, and top_k with clear ValueErrors.
9. Resolve paths relative to the project root directory.
"""

import os
from typing import Any, Dict, List, Optional
import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "paperscout_papers"

# Supported target categories and their metadata Boolean flag mappings
CATEGORY_FLAG_MAP = {
    "cs.CL": "is_cs_cl",
    "cs.LG": "is_cs_lg",
    "cs.CV": "is_cs_cv",
}

# Resolve project paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CHROMA_DIR = os.path.join(PROJECT_ROOT, "chroma_db")

# Module-level singletons to avoid reloading on every function call
_MODEL: SentenceTransformer = None
_COLLECTION = None


def get_embedding_model() -> SentenceTransformer:
    """Load and cache the SentenceTransformer model once."""
    global _MODEL
    if _MODEL is None:
        print(f"Loading embedding model '{MODEL_NAME}'...")
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def get_chroma_collection(db_path: str = CHROMA_DIR) -> Any:
    """Connect to persistent ChromaDB and retrieve the papers collection once."""
    global _COLLECTION
    if _COLLECTION is None:
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"ChromaDB directory not found at: {db_path}. Please run Stage 5 storage first."
            )
        client = chromadb.PersistentClient(path=db_path)
        existing_collections = [c.name for c in client.list_collections()]
        if COLLECTION_NAME not in existing_collections:
            raise ValueError(
                f"Collection '{COLLECTION_NAME}' does not exist in ChromaDB at {db_path}."
            )
        _COLLECTION = client.get_collection(name=COLLECTION_NAME)
    return _COLLECTION


def retrieve_papers(
    question: str,
    category: Optional[str] = None,
    top_k: int = 4,
) -> List[Dict[str, str]]:
    """
    Retrieve the top_k most semantically relevant papers for a natural-language question,
    with optional category pre-filtering.

    Args:
        question: User query string.
        category: Optional category filter. Allowed values: 'cs.CL', 'cs.LG', 'cs.CV', or None.
        top_k: Number of relevant papers to retrieve (default: 4). Must be a positive integer.

    Returns:
        List of dictionaries with keys: 'title', 'abstract', 'arxiv_id', 'category'.
    """
    # 1. Validate query question
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Query 'question' must be a non-empty string.")

    cleaned_question = question.strip()

    # 2. Validate top_k: reject non-integers, booleans, zero, and negative values
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(
            f"Parameter 'top_k' must be a positive integer greater than 0, got {top_k!r} ({type(top_k).__name__})."
        )

    # 3. Validate category parameter and build ChromaDB 'where' filter
    where_filter: Optional[Dict[str, Any]] = None
    if category is not None:
        if not isinstance(category, str) or category.strip() not in CATEGORY_FLAG_MAP:
            allowed = list(CATEGORY_FLAG_MAP.keys())
            raise ValueError(
                f"Invalid category '{category}'. Allowed categories are {allowed} or None for unrestricted retrieval."
            )
        cleaned_category = category.strip()
        flag_field = CATEGORY_FLAG_MAP[cleaned_category]
        where_filter = {flag_field: True}

    # 4. Access collection and handle collection size bounds
    collection = get_chroma_collection()
    total_count = collection.count()

    if total_count == 0:
        return []

    # Bound top_k safely if collection has fewer items than requested
    k = min(top_k, total_count)

    # 5. Generate query embedding with L2 normalization (consistent with Stage 4)
    model = get_embedding_model()
    query_vector = model.encode(
        cleaned_question,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    # 6. Perform vector similarity search in ChromaDB with category pre-filtering
    # The 'where' filter is evaluated inside ChromaDB before the top-k results are selected
    query_kwargs: Dict[str, Any] = {
        "query_embeddings": [query_vector],
        "n_results": k,
        "include": ["metadatas", "distances"],
    }
    if where_filter is not None:
        query_kwargs["where"] = where_filter

    results = collection.query(**query_kwargs)

    # 7. Check if results were returned
    if not results or not results.get("ids") or len(results["ids"][0]) == 0:
        return []

    retrieved_papers: List[Dict[str, str]] = []
    ids = results["ids"][0]
    metadatas = results["metadatas"][0]

    # 8. Map to required output schema
    for paper_id, meta in zip(ids, metadatas):
        retrieved_papers.append(
            {
                "title": meta.get("title", ""),
                "abstract": meta.get("abstract", ""),
                "arxiv_id": str(paper_id),
                "category": meta.get("primary_category", ""),
            }
        )

    return retrieved_papers


if __name__ == "__main__":
    print("=" * 60)
    print("STAGE 6 & 7 TEST: RETRIEVAL & CATEGORY FILTERING")
    print("=" * 60)

    collection = get_chroma_collection()

    test_cases = [
        {
            "name": "Test 1: Unrestricted Retrieval (category=None)",
            "query": "How do diffusion models improve parallel decoding in large language models?",
            "category": None,
        },
        {
            "name": "Test 2: Category Filter cs.CL (Computation & Language)",
            "query": "What techniques optimize key-value caching in large language models?",
            "category": "cs.CL",
        },
        {
            "name": "Test 3: Category Filter cs.LG (Machine Learning)",
            "query": "How does reinforcement learning with human feedback stabilize policy gradients?",
            "category": "cs.LG",
        },
        {
            "name": "Test 4: Category Filter cs.CV (Computer Vision)",
            "query": "How do vision transformers and convolutional backbones perform object detection?",
            "category": "cs.CV",
        },
    ]

    try:
        # Run standard top-4 retrieval tests
        for tc in test_cases:
            print(f"\n--- {tc['name']} ---")
            print(f"Query: '{tc['query']}'")
            print(f"Filter: category={tc['category']}")

            papers = retrieve_papers(tc["query"], category=tc["category"], top_k=4)
            print(f"Retrieved {len(papers)} papers:")

            flag_key = CATEGORY_FLAG_MAP.get(tc["category"]) if tc["category"] else None
            for rank, paper in enumerate(papers, start=1):
                raw = collection.get(ids=[paper["arxiv_id"]], include=["metadatas"])
                meta = raw["metadatas"][0] if raw["metadatas"] else {}
                target_cats = meta.get("target_categories", "")
                primary_cat = meta.get("primary_category", "")

                # Verification check: did this paper belong to the requested category?
                if flag_key:
                    has_flag = meta.get(flag_key) is True
                    if not has_flag:
                        raise ValueError(
                            f"Integrity check failed: Paper {paper['arxiv_id']} retrieved for category "
                            f"{tc['category']} but {flag_key} is False!"
                        )

                is_cross_listed = (tc["category"] is not None and primary_cat != tc["category"])
                cross_str = f" [Cross-listed: primary is {primary_cat}]" if is_cross_listed else ""

                print(f"  [{rank}] arXiv ID: {paper['arxiv_id']} | Primary: {paper['category']} | Targets: {target_cats}{cross_str}")
                print(f"      Title   : {paper['title'][:65]}...")

        # Test 5: Comprehensive Cross-Listing & Full-Corpus Flag Agreement Verification
        print("\n--- Test 5: Full Corpus Pagination Verification Across All 5,496 Records ---")
        total_collection_count = collection.count()
        batch_size = 500

        total_scanned = 0
        count_cl_true = 0
        count_lg_true = 0
        count_cv_true = 0
        count_cross_listed = 0

        # Category-specific cross-listed tracking
        cross_by_cat = {
            "cs.CL": [],
            "cs.LG": [],
            "cs.CV": [],
        }

        offset = 0
        while True:
            batch = collection.get(
                include=["metadatas"],
                limit=batch_size,
                offset=offset,
            )
            batch_ids = batch.get("ids", [])
            batch_metas = batch.get("metadatas", [])
            if not batch_ids:
                break

            # 5. Verify batch result consistency before zip()
            if len(batch_ids) != len(batch_metas):
                raise ValueError(
                    f"Batch result count mismatch at offset {offset}: "
                    f"received {len(batch_ids)} IDs but {len(batch_metas)} metadata records."
                )

            for pid, m in zip(batch_ids, batch_metas):
                total_scanned += 1
                target_str = str(m.get("target_categories", ""))
                tokens = [c.strip() for c in target_str.replace(";", " ").split() if c.strip()]
                primary_cat = m.get("primary_category", "")

                flag_cl = m.get("is_cs_cl")
                flag_lg = m.get("is_cs_lg")
                flag_cv = m.get("is_cs_cv")

                # Verify all flags are strict instances of bool
                if not isinstance(flag_cl, bool) or not isinstance(flag_lg, bool) or not isinstance(flag_cv, bool):
                    raise ValueError(f"Record {pid} contains non-boolean category flag!")

                # Expected membership derived independently from target_categories
                expected_cl = "cs.CL" in tokens
                expected_lg = "cs.LG" in tokens
                expected_cv = "cs.CV" in tokens

                # Verify BOTH True and False values match exact token membership
                if flag_cl != expected_cl:
                    raise ValueError(f"Flag mismatch for cs.CL on record {pid}: actual={flag_cl}, expected={expected_cl} (targets: '{target_str}')")
                if flag_lg != expected_lg:
                    raise ValueError(f"Flag mismatch for cs.LG on record {pid}: actual={flag_lg}, expected={expected_lg} (targets: '{target_str}')")
                if flag_cv != expected_cv:
                    raise ValueError(f"Flag mismatch for cs.CV on record {pid}: actual={flag_cv}, expected={expected_cv} (targets: '{target_str}')")

                if flag_cl:
                    count_cl_true += 1
                    if primary_cat != "cs.CL":
                        cross_by_cat["cs.CL"].append((pid, primary_cat, target_str))
                if flag_lg:
                    count_lg_true += 1
                    if primary_cat != "cs.LG":
                        cross_by_cat["cs.LG"].append((pid, primary_cat, target_str))
                if flag_cv:
                    count_cv_true += 1
                    if primary_cat != "cs.CV":
                        cross_by_cat["cs.CV"].append((pid, primary_cat, target_str))

                if (int(flag_cl) + int(flag_lg) + int(flag_cv)) > 1:
                    count_cross_listed += 1

            offset += len(batch_ids)
            if len(batch_ids) < batch_size:
                break

        # 4. Verify pagination completeness
        if total_scanned != total_collection_count:
            raise ValueError(
                f"Incomplete scan: scanned {total_scanned}, "
                f"expected {total_collection_count} records."
            )

        print(f"  Paginating completed: Scanned {total_scanned}/{total_collection_count} records in ChromaDB.")
        print(f"  Verified 100% of True and False flag values agree with target_categories token membership.")
        print(f"\n  Corpus Category Totals:")
        print(f"    - is_cs_cl == True : {count_cl_true}")
        print(f"    - is_cs_lg == True : {count_lg_true}")
        print(f"    - is_cs_cv == True : {count_cv_true}")
        print(f"    - Cross-listed papers (>1 True flag): {count_cross_listed}")

        print(f"\n  Category Cross-Listing Breakdown (primary != target category):")
        for cat, items in cross_by_cat.items():
            if len(items) == 0:
                raise ValueError(f"Integrity check failed: Zero cross-listed records found for {cat}!")
            sample = items[0]
            print(f"    - {cat}: {len(items)} cross-listed papers (e.g. ID {sample[0]}, primary: '{sample[1]}', targets: '{sample[2]}')")

    except Exception as e:
        raise RuntimeError(f"Stage 7 retrieval test execution failed: {e}") from e

    print("\n" + "=" * 60)
    print("STAGE 7 TEST COMPLETE: All category filters, full corpus flags, and cross-listing verified.")
    print("=" * 60)
