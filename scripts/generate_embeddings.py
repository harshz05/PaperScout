"""
scripts/generate_embeddings.py
Stage 4: Embedding Generation for PaperScout.

Responsibilities:
1. Load filtered papers from data/processed/arxiv_papers_filtered.csv.
2. Combine each paper's title and abstract into a single formatted text representation.
3. Load the pretrained SentenceTransformer model 'all-MiniLM-L6-v2'.
4. Generate 384-dimensional embeddings using batch processing for memory efficiency.
5. Verify the embedding count (matching filtered rows) and dimensions (384).
6. Save the embeddings as a NumPy array to data/processed/paper_embeddings.npy.
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EXPECTED_DIM = 384
BATCH_SIZE = 64


def prepare_combined_texts(df: pd.DataFrame) -> list[str]:
    """
    Combines 'title' and 'abstract' for each paper into an informative text representation:
    'Title: <title>\nAbstract: <abstract>'
    """
    titles = df["title"].fillna("").astype(str).str.strip()
    abstracts = df["abstract"].fillna("").astype(str).str.strip()
    combined = "Title: " + titles + "\nAbstract: " + abstracts
    return combined.tolist()


def generate_and_save_embeddings(
    input_csv_path: str,
    output_npy_path: str,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    print("=" * 60)
    print("STAGE 4: EMBEDDING GENERATION")
    print("=" * 60)

    # 1. Validation and data loading
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input file not found at: {input_csv_path}")

    print(f"Loading filtered dataset from: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    total_papers = len(df)
    print(f"Total papers to embed: {total_papers}")

    for req_col in ["title", "abstract"]:
        if req_col not in df.columns:
            raise ValueError(f"Required column '{req_col}' missing from filtered dataset.")

    # 2. Prepare combined text representations
    print("Combining 'title' and 'abstract' for each paper...")
    texts_to_embed = prepare_combined_texts(df)

    # 3. Load SentenceTransformer model
    print(f"\nLoading SentenceTransformer model: '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    # 4. Generate embeddings with batch processing
    print(f"Generating embeddings (batch size = {batch_size}, show_progress_bar = True)...")
    embeddings = model.encode(
        texts_to_embed,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # Normalizing enables fast cosine similarity via dot product
        convert_to_numpy=True,
    )

    # 5. Verification checks
    print("\n--- Verification Checks ---")
    print(f"  Shape of embeddings: {embeddings.shape}")
    print(f"  Data type: {embeddings.dtype}")

    if embeddings.shape[0] != total_papers:
        raise ValueError(
            f"Embedding count mismatch! Expected {total_papers}, got {embeddings.shape[0]}"
        )
    if embeddings.shape[1] != EXPECTED_DIM:
        raise ValueError(
            f"Embedding dimension mismatch! Expected {EXPECTED_DIM}, got {embeddings.shape[1]}"
        )

    print(f"  Total embeddings generated: {embeddings.shape[0]}")
    print(f"  Vector dimension: {embeddings.shape[1]} (matches expected: {EXPECTED_DIM})")

    # 6. Save embeddings as .npy
    os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)
    np.save(output_npy_path, embeddings)
    file_size_mb = os.path.getsize(output_npy_path) / (1024 * 1024)
    print(f"\nEmbeddings saved successfully to: {output_npy_path} ({file_size_mb:.2f} MB)")
    print("=" * 60)

    return embeddings


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(script_dir)
    filtered_csv = os.path.join(repo_dir, "data", "processed", "arxiv_papers_filtered.csv")
    output_npy = os.path.join(repo_dir, "data", "processed", "paper_embeddings.npy")

    generate_and_save_embeddings(filtered_csv, output_npy)
