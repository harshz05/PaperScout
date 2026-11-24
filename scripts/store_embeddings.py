"""
scripts/store_embeddings.py
Stage 5 & 7: ChromaDB Storage & In-Place Metadata Enrichment for PaperScout.

Responsibilities:
1. Connect to existing ChromaDB collection 'paperscout_papers' at chroma_db/.
2. Verify collection pre-conditions:
   - Distance metric: 'cosine'
   - Exact count: 5,496
   - All 5,496 CSV paper IDs present in the collection
3. In batches of 100, RETRIEVE the existing metadata directly from ChromaDB.
4. Verify returned_ids match requested batch_ids exactly and pair metadata accordingly.
5. Preserve all existing metadata fields intact, updating ONLY:
   - is_cs_cl
   - is_cs_lg
   - is_cs_cv
   (derived from the record's existing 'target_categories')
6. Call collection.update(ids=returned_ids, metadatas=batch_updated_metadatas)
   (strictly no embeddings or documents supplied).
7. Post-update comprehensive verification across ALL 5,496 records in batches:
   - Collection count remains exactly 5,496
   - Per-record verification comparing ChromaDB target_categories against CSV ground truth
   - Every single record's is_cs_cl, is_cs_lg, is_cs_cv flags match exact
     token membership in its 'target_categories'
   - Total True flag counts and cross-listed count are independently compared against
     the ground-truth CSV (cs.CL: 1685, cs.LG: 2815, cs.CV: 1728) and raise on mismatch
   - Existing embeddings and documents remain unchanged
"""

import os
import chromadb
import numpy as np
import pandas as pd

COLLECTION_NAME = "paperscout_papers"
EXPECTED_COUNT = 5496
EXPECTED_DIM = 384
DEFAULT_BATCH_SIZE = 100


def update_chromadb_metadata(
    csv_path: str,
    db_dir: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> chromadb.Collection:
    print("=" * 60)
    print("STAGE 7: IN-PLACE METADATA ENRICHMENT (EXISTING RECORD PRESERVATION)")
    print("=" * 60)

    # 1. Validate batch_size: reject non-integers, booleans, zero, and negative values
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(
            f"Parameter 'batch_size' must be a positive integer greater than 0, got {batch_size!r} ({type(batch_size).__name__})."
        )

    # 2. Validation and CSV loading for ID ground truth and baseline calculations
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Filtered CSV not found at: {csv_path}")
    if not os.path.exists(db_dir):
        raise FileNotFoundError(f"ChromaDB directory not found at: {db_dir}")

    print(f"Loading filtered CSV for ID validation and baseline metrics: {csv_path}")
    df = pd.read_csv(csv_path)
    total_records = len(df)
    print(f"Total papers in CSV: {total_records}")

    if total_records != EXPECTED_COUNT:
        raise ValueError(f"Expected {EXPECTED_COUNT} papers in CSV, found {total_records}")

    # 2a. Validate required CSV columns
    required_csv_cols = ["paper_id", "target_categories"]
    for col in required_csv_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the filtered dataset at {csv_path}.")

    missing_ids = df["paper_id"].isna().sum() + (df["paper_id"].astype(str).str.strip() == "").sum()
    if missing_ids > 0:
        raise ValueError(f"Found {missing_ids} missing or empty 'paper_id' values in the dataset.")

    duplicate_ids = df["paper_id"].duplicated().sum()
    if duplicate_ids > 0:
        raise ValueError(f"Found {duplicate_ids} duplicate 'paper_id' values in the dataset.")

    csv_ids = df["paper_id"].astype(str).tolist()

    # Build per-record CSV lookup dictionary for ground truth comparison
    csv_lookup = dict(zip(csv_ids, df["target_categories"].fillna("").astype(str).str.strip()))

    # Independently calculate ground-truth expected counts from CSV
    print("\n--- Calculating Ground-Truth Counts from Filtered CSV ---")
    csv_targets = df["target_categories"].fillna("").astype(str)
    csv_tokens = csv_targets.apply(lambda s: [c.strip() for c in s.replace(";", " ").split() if c.strip()])

    expected_csv_cl = csv_tokens.apply(lambda t: "cs.CL" in t).sum()
    expected_csv_lg = csv_tokens.apply(lambda t: "cs.LG" in t).sum()
    expected_csv_cv = csv_tokens.apply(lambda t: "cs.CV" in t).sum()
    expected_csv_cross = (
        (csv_tokens.apply(lambda t: "cs.CL" in t).astype(int) +
         csv_tokens.apply(lambda t: "cs.LG" in t).astype(int) +
         csv_tokens.apply(lambda t: "cs.CV" in t).astype(int)) > 1
    ).sum()

    print(f"  Expected cs.CL count from CSV : {expected_csv_cl}")
    print(f"  Expected cs.LG count from CSV : {expected_csv_lg}")
    print(f"  Expected cs.CV count from CSV : {expected_csv_cv}")
    print(f"  Expected cross-listed count   : {expected_csv_cross}")

    # 3. Connect to existing ChromaDB collection and verify pre-conditions
    print(f"\nConnecting to ChromaDB at: {db_dir}")
    client = chromadb.PersistentClient(path=db_dir)

    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME not in existing_collections:
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' does not exist in ChromaDB at {db_dir}. "
            f"Please populate the initial collection first."
        )

    collection = client.get_collection(name=COLLECTION_NAME)

    # 3a. Verify distance metric is 'cosine'
    coll_metadata = collection.metadata or {}
    current_metric = coll_metadata.get("hnsw:space")
    if current_metric != "cosine":
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' has distance metric '{current_metric}', expected 'cosine'."
        )
    print(f"  Verified collection '{COLLECTION_NAME}' distance metric: 'cosine'")

    # 3b. Verify initial collection count is exactly 5,496
    initial_count = collection.count()
    if initial_count != EXPECTED_COUNT:
        raise ValueError(
            f"Pre-update count mismatch: expected {EXPECTED_COUNT} items, found {initial_count}."
        )
    print(f"  Verified pre-update collection count: {initial_count}")

    # 3c. Verify all CSV paper IDs exist in the collection
    print("  Verifying all 5,496 CSV paper IDs exist in ChromaDB...")
    existing_records = collection.get(ids=csv_ids, include=[])
    retrieved_id_count = len(existing_records["ids"])
    if retrieved_id_count != EXPECTED_COUNT:
        raise ValueError(
            f"ID presence verification failed! Found {retrieved_id_count} of {EXPECTED_COUNT} IDs in collection."
        )
    print(f"  Verified all {retrieved_id_count} paper IDs are present in the collection.")

    # Capture a baseline sample record to verify embeddings/documents remain unchanged
    sample_id = csv_ids[0]
    baseline_sample = collection.get(ids=[sample_id], include=["embeddings", "documents", "metadatas"])
    baseline_embedding = np.array(baseline_sample["embeddings"][0])
    baseline_document = baseline_sample["documents"][0]
    baseline_metadata = dict(baseline_sample["metadatas"][0])

    # 4. Retrieve existing metadata in batches, preserve all fields, and update only the 3 flags
    num_batches = (total_records + batch_size - 1) // batch_size
    print(f"\nRetrieving existing metadata and updating flags in {num_batches} batches of {batch_size}...")

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, total_records)
        batch_ids = csv_ids[start_idx:end_idx]

        # Retrieve the EXISTING metadata directly from ChromaDB for this batch
        batch_records = collection.get(ids=batch_ids, include=["metadatas"])
        returned_ids = batch_records.get("ids", [])
        existing_metadatas = batch_records.get("metadatas", [])

        # Verify returned_ids contain exactly the requested batch IDs
        if len(returned_ids) != len(batch_ids) or set(returned_ids) != set(batch_ids):
            raise ValueError(
                f"Batch {i + 1} ID mismatch! Requested {len(batch_ids)} IDs, but ChromaDB returned {len(returned_ids)} IDs."
            )
        if len(returned_ids) != len(existing_metadatas):
            raise ValueError(
                f"Batch {i + 1} metadata length mismatch! Got {len(returned_ids)} IDs but {len(existing_metadatas)} metadata records."
            )

        batch_updated_metadatas = []
        # Pair retrieved metadata with returned_ids
        for pid, meta in zip(returned_ids, existing_metadatas):
            # Preserve all original fields intact
            updated_meta = dict(meta)

            # Derive the 3 Boolean flags from the existing target_categories string
            target_str = str(updated_meta.get("target_categories", ""))
            tokens = [c.strip() for c in target_str.replace(";", " ").split() if c.strip()]

            # Update ONLY these three fields
            updated_meta["is_cs_cl"] = ("cs.CL" in tokens)
            updated_meta["is_cs_lg"] = ("cs.LG" in tokens)
            updated_meta["is_cs_cv"] = ("cs.CV" in tokens)

            batch_updated_metadatas.append(updated_meta)

        # Call collection.update() with returned_ids and paired metadatas (zero embeddings or documents)
        collection.update(
            ids=returned_ids,
            metadatas=batch_updated_metadatas,
        )

        if (i + 1) % 10 == 0 or (i + 1) == num_batches:
            print(f"  Processed batch {i + 1}/{num_batches} ({end_idx}/{total_records} records)")

    # 5. Post-update verification checks across ALL 5,496 records in batches
    print("\n--- Post-Update Comprehensive Verification (All 5,496 Records in Batches) ---")

    # 5a. Verify collection count remains exactly 5,496
    final_count = collection.count()
    print(f"  Final collection count: {final_count}")
    if final_count != EXPECTED_COUNT:
        raise ValueError(
            f"Post-update count verification failed! Expected {EXPECTED_COUNT}, got {final_count}."
        )

    # 5b. Retrieve all 5,496 metadata records in batches, independently verify flags and per-record CSV agreement
    actual_cl = 0
    actual_lg = 0
    actual_cv = 0
    actual_cross = 0

    print(f"  Retrieving and verifying all {total_records} records in batches of {batch_size}...")
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, total_records)
        batch_ids = csv_ids[start_idx:end_idx]

        batch_data = collection.get(ids=batch_ids, include=["metadatas"])
        batch_ret_ids = batch_data.get("ids", [])
        batch_metas = batch_data.get("metadatas", [])

        if len(batch_ret_ids) != len(batch_metas):
            raise ValueError(f"Verification batch {i + 1} ID and metadata count mismatch!")

        for pid, m in zip(batch_ret_ids, batch_metas):
            # Verify required pre-existing fields were preserved
            for req_field in ["title", "abstract", "authors", "primary_category", "all_categories", "target_categories"]:
                if req_field not in m:
                    raise ValueError(f"Record {pid} is missing preserved field '{req_field}'!")

            # Per-record verification: Compare ChromaDB target_categories against CSV ground truth
            chroma_target_str = str(m.get("target_categories", "")).strip()
            csv_target_str = csv_lookup.get(pid)
            if csv_target_str is None:
                raise ValueError(f"Record {pid} present in ChromaDB but missing from CSV lookup!")
            if chroma_target_str != csv_target_str:
                raise ValueError(
                    f"Per-record mismatch for paper_id '{pid}'! ChromaDB target_categories: '{chroma_target_str}', "
                    f"CSV target_categories: '{csv_target_str}'"
                )

            # Verify all three flags are instances of bool
            flag_cl = m.get("is_cs_cl")
            flag_lg = m.get("is_cs_lg")
            flag_cv = m.get("is_cs_cv")

            if not isinstance(flag_cl, bool) or not isinstance(flag_lg, bool) or not isinstance(flag_cv, bool):
                raise ValueError(f"Record {pid} has non-boolean category flag!")

            # Independently derive expected membership from record's target_categories
            tokens = [c.strip() for c in chroma_target_str.replace(";", " ").split() if c.strip()]

            expected_cl = "cs.CL" in tokens
            expected_lg = "cs.LG" in tokens
            expected_cv = "cs.CV" in tokens

            # Verify exact flag agreement
            if flag_cl != expected_cl:
                raise ValueError(f"Record {pid}: is_cs_cl={flag_cl} does not match expected {expected_cl} (targets: '{chroma_target_str}')")
            if flag_lg != expected_lg:
                raise ValueError(f"Record {pid}: is_cs_lg={flag_lg} does not match expected {expected_lg} (targets: '{chroma_target_str}')")
            if flag_cv != expected_cv:
                raise ValueError(f"Record {pid}: is_cs_cv={flag_cv} does not match expected {expected_cv} (targets: '{chroma_target_str}')")

            if flag_cl:
                actual_cl += 1
            if flag_lg:
                actual_lg += 1
            if flag_cv:
                actual_cv += 1
            if (int(flag_cl) + int(flag_lg) + int(flag_cv)) > 1:
                actual_cross += 1

    print(f"  Verified across all {total_records} records in ChromaDB:")
    print(f"    - All individual records match CSV target_categories ground truth 100%.")
    print(f"    - Actual is_cs_cl == True : {actual_cl} (CSV Expected: {expected_csv_cl})")
    print(f"    - Actual is_cs_lg == True : {actual_lg} (CSV Expected: {expected_csv_lg})")
    print(f"    - Actual is_cs_cv == True : {actual_cv} (CSV Expected: {expected_csv_cv})")
    print(f"    - Actual cross-listed     : {actual_cross} (CSV Expected: {expected_csv_cross})")

    # 5c. Rigorously compare actual ChromaDB counts against CSV-derived expected counts
    if actual_cl != expected_csv_cl:
        raise ValueError(f"ChromaDB cs.CL flag count mismatch! Got {actual_cl}, expected {expected_csv_cl}")
    if actual_lg != expected_csv_lg:
        raise ValueError(f"ChromaDB cs.LG flag count mismatch! Got {actual_lg}, expected {expected_csv_lg}")
    if actual_cv != expected_csv_cv:
        raise ValueError(f"ChromaDB cs.CV flag count mismatch! Got {actual_cv}, expected {expected_csv_cv}")
    if actual_cross != expected_csv_cross:
        raise ValueError(f"ChromaDB cross-listed count mismatch! Got {actual_cross}, expected {expected_csv_cross}")

    print("  All ChromaDB category flag counts and cross-listed totals match CSV ground truth 100%.")

    # 5d. Verify sample embedding and document remain unchanged
    post_sample = collection.get(ids=[sample_id], include=["embeddings", "documents", "metadatas"])
    post_embedding = np.array(post_sample["embeddings"][0])
    post_document = post_sample["documents"][0]
    post_meta = post_sample["metadatas"][0]

    if not np.allclose(baseline_embedding, post_embedding):
        raise ValueError("Embedding mismatch detected after metadata update!")
    if baseline_document != post_document:
        raise ValueError("Document text mismatch detected after metadata update!")

    # Verify original keys were completely preserved without modification
    for k, v in baseline_metadata.items():
        if post_meta.get(k) != v:
            raise ValueError(f"Original metadata key '{k}' changed from {v!r} to {post_meta.get(k)!r}!")

    print(f"  Verified sample record '{sample_id}' embedding and document text are completely unchanged.")
    print(f"  Sample metadata flags: is_cs_cl={post_meta.get('is_cs_cl')}, "
          f"is_cs_lg={post_meta.get('is_cs_lg')}, is_cs_cv={post_meta.get('is_cs_cv')}")

    print("\n" + "=" * 60)
    print("STAGE 7 SUCCESS: In-place metadata update verified successfully across all records.")
    print("=" * 60)
    return collection


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(script_dir)
    filtered_csv = os.path.join(repo_dir, "data", "processed", "arxiv_papers_filtered.csv")
    chroma_path = os.path.join(repo_dir, "chroma_db")

    update_chromadb_metadata(filtered_csv, chroma_path)
