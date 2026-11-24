"""
scripts/data_filtering.py
Stage 3: Category Filtering for PaperScout.

Responsibilities:
1. Validate required columns exist before processing.
2. Load cleaned dataset from data/processed/arxiv_papers_cleaned.csv.
3. Filter papers belonging to target categories: cs.CL, cs.LG, cs.CV.
4. Use exact category matches in 'all_categories' (retaining cross-listed papers).
5. Add 'target_categories' column with list of matched categories (e.g. "cs.CL; cs.LG").
6. Save filtered dataset to data/processed/arxiv_papers_filtered.csv.
7. Provide verification counts for each target category and overall dataset using exact membership.
"""

import os
import pandas as pd

REQUIRED_COLUMNS = ["paper_id", "title", "abstract", "all_categories"]
TARGET_CATEGORIES = ["cs.CL", "cs.LG", "cs.CV"]


def extract_matched_targets(all_cats_str: str) -> list[str]:
    """
    Split category strings by semicolon/whitespace and match exact category tags.
    Handles delimiters like 'cs.CL; cs.AI; cs.LG' or 'cs.CL cs.AI'.
    """
    if not isinstance(all_cats_str, str):
        return []
    tokens = [cat.strip() for cat in all_cats_str.replace(";", " ").split() if cat.strip()]
    return [t for t in TARGET_CATEGORIES if t in tokens]


def filter_dataset(input_csv_path: str, output_csv_path: str) -> pd.DataFrame:
    print("=" * 60)
    print("STAGE 3: CATEGORY FILTERING")
    print("=" * 60)

    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input file not found at: {input_csv_path}")

    print(f"Loading cleaned dataset: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    total_cleaned = len(df)
    print(f"Total cleaned rows: {total_cleaned}")

    # Validate required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input dataset is missing required columns: {missing_cols}")

    # Extract target category matches per paper
    df["target_categories_list"] = df["all_categories"].apply(extract_matched_targets)
    
    # Filter rows that have at least one target category
    df_filtered = df[df["target_categories_list"].apply(lambda cats: len(cats) > 0)].copy()

    # Category breakdown using exact list membership (avoiding regex string matching)
    filtered_count = len(df_filtered)
    retention_pct = (filtered_count / total_cleaned) * 100
    print(f"\nFiltered paper count: {filtered_count} ({retention_pct:.2f}% of cleaned data)")

    print("\n--- Target Category Counts ---")
    for cat in TARGET_CATEGORIES:
        count = df_filtered["target_categories_list"].apply(lambda cats: cat in cats).sum()
        pct = (count / filtered_count) * 100
        print(f"  {cat}: {count} papers ({pct:.2f}%)")

    # Format target_categories as clean semicolon-separated string for CSV export
    df_filtered["target_categories"] = df_filtered["target_categories_list"].apply(lambda cats: "; ".join(cats))
    df_filtered = df_filtered.drop(columns=["target_categories_list"])

    # Verification checks
    empty_titles = (df_filtered["title"].astype(str).str.strip() == "").sum()
    empty_abstracts = (df_filtered["abstract"].astype(str).str.strip() == "").sum()
    dup_ids = df_filtered["paper_id"].duplicated().sum()

    print("\n--- Verification Checks ---")
    print(f"  Empty titles: {empty_titles}")
    print(f"  Empty abstracts: {empty_abstracts}")
    print(f"  Duplicate paper IDs: {dup_ids}")

    # Save filtered dataset
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_filtered.to_csv(output_csv_path, index=False)
    print(f"\nFiltered dataset saved to: {output_csv_path}")
    print("=" * 60)
    return df_filtered


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(script_dir)
    cleaned_path = os.path.join(repo_dir, "data", "processed", "arxiv_papers_cleaned.csv")
    filtered_path = os.path.join(repo_dir, "data", "processed", "arxiv_papers_filtered.csv")
    filter_dataset(cleaned_path, filtered_path)
