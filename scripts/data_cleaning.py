"""
scripts/data_cleaning.py
Stage 2: Data Cleaning for PaperScout.

Responsibilities:
1. Validate required columns exist before processing.
2. Load raw dataset from data/raw/arxiv_ai_ml_papers.csv.
3. Strip extraneous whitespace from relevant text columns.
4. Convert sentinel values like 'Not Available' in journal_ref and doi to NaN.
5. Remove records with missing or empty titles or abstracts.
6. Validate against duplicates by paper_id.
7. Save cleaned data to data/processed/arxiv_papers_cleaned.csv.
"""

import os
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["paper_id", "title", "authors", "abstract", "all_categories"]


def clean_dataset(input_csv_path: str, output_csv_path: str) -> pd.DataFrame:
    print("=" * 60)
    print("STAGE 2: DATA CLEANING")
    print("=" * 60)

    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input file not found at: {input_csv_path}")

    print(f"Loading raw dataset: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    initial_rows = len(df)
    print(f"Initial row count: {initial_rows}")

    # Validate required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input dataset is missing required columns: {missing_cols}")

    # 1. Text columns whitespace stripping
    text_cols = ["paper_id", "title", "authors", "abstract", "primary_category", "all_categories"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # 2. Replace sentinel values with NaN in journal_ref and doi
    for col in ["journal_ref", "doi"]:
        if col in df.columns:
            df[col] = df[col].replace(["Not Available", "not available", "None", "nan", ""], np.nan)

    # 3. Drop rows where title or abstract is empty or NaN
    df = df[df["title"].notna() & (df["title"] != "") & (df["title"].str.lower() != "nan")]
    df = df[df["abstract"].notna() & (df["abstract"] != "") & (df["abstract"].str.lower() != "nan")]

    # 4. Remove duplicate paper_id records if any exist
    duplicate_ids = df["paper_id"].duplicated().sum()
    if duplicate_ids > 0:
        print(f"Removing {duplicate_ids} duplicate paper_id records...")
        df = df.drop_duplicates(subset=["paper_id"], keep="first")
    else:
        print("Duplicate paper_ids: 0 (all IDs are unique)")

    cleaned_rows = len(df)
    dropped_rows = initial_rows - cleaned_rows
    print(f"Cleaned row count: {cleaned_rows} (Dropped: {dropped_rows})")

    # 5. Verification checks
    empty_titles = (df["title"].str.strip() == "").sum()
    empty_abstracts = (df["abstract"].str.strip() == "").sum()
    print(f"Verification - Empty titles remaining: {empty_titles}")
    print(f"Verification - Empty abstracts remaining: {empty_abstracts}")

    # 6. Save result
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Cleaned dataset successfully saved to: {output_csv_path}")
    print("=" * 60)
    return df


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(script_dir)
    raw_path = os.path.join(repo_dir, "data", "raw", "arxiv_ai_ml_papers.csv")
    cleaned_path = os.path.join(repo_dir, "data", "processed", "arxiv_papers_cleaned.csv")
    clean_dataset(raw_path, cleaned_path)
