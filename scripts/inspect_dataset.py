import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(script_dir)
csv_path = os.path.join(repo_dir, "data", "raw", "arxiv_ai_ml_papers.csv")

df = pd.read_csv(csv_path)

print("=== CHECKING FOR SENTINEL 'MISSING' VALUES ===")
sentinels = ["Not Available", "none", "nan", "null", "unknown", ""]
for col in df.select_dtypes(include="object").columns:
    counts = {}
    for s in sentinels:
        c = (df[col].astype(str).str.strip().str.lower() == s.lower()).sum()
        if c > 0:
            counts[s] = c
    if counts:
        print(f"Column '{col}' placeholder values: {counts}")

print("\n=== PAPER_ID DUPLICATES ===")
print("paper_id duplicates:", df["paper_id"].duplicated().sum())

print("\n=== DATE RANGE ===")
print("Min submission date:", df["submitted_date"].min())
print("Max submission date:", df["submitted_date"].max())
print("Years represented:", df["year"].value_counts().to_dict())
