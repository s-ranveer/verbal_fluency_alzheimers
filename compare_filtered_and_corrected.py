"""
Compare word_counts_year_1.csv and word_counts_year_1_corrected.csv.

Computes per-patient and group-level (MCI vs Normal Aging) differences
for the semantic and phonetic word counts.
"""

import pandas as pd
import numpy as np
import os

OUTPUT_DIR = "outputs"

# ---------------------------------------------------------------------------
# 1. Load the two word-count CSVs
# ---------------------------------------------------------------------------
original_path = os.path.join(OUTPUT_DIR, "word_counts_year_1.csv")
corrected_path = os.path.join(OUTPUT_DIR, "word_counts_year_1_corrected.csv")

df_original = pd.read_csv(original_path, dtype={"subject_id": str})
df_corrected = pd.read_csv(corrected_path, dtype={"subject_id": str})

# ---------------------------------------------------------------------------
# 2. Load participant classification from the Excel file
# ---------------------------------------------------------------------------
excel_path = os.path.join(OUTPUT_DIR, "Updated10152025_UTDallas_CogntiveTestList.xlsx")
df_participants = pd.read_excel(excel_path)

# Extract numeric ID string from RWRAD_XXX format (keep as zero-padded str)
participant_col = "Participants (Normal Aging or MCI at Baseline)"
classification_col = "Year 1/Baseline"

df_participants["subject_id"] = (
    df_participants[participant_col]
    .str.extract(r"(\d+)$")[0]
)
df_participants = df_participants[["subject_id", classification_col]].rename(
    columns={classification_col: "group"}
)

# ---------------------------------------------------------------------------
# 3. Columns of interest
# ---------------------------------------------------------------------------
COMPARE_COLS = ["semantic", "phonetic"]

# ---------------------------------------------------------------------------
# 4. Merge datasets on subject_id and compute differences
# ---------------------------------------------------------------------------
common_ids = set(df_original["subject_id"]) & set(df_corrected["subject_id"])
df_original = df_original[df_original["subject_id"].isin(common_ids)].copy()
df_corrected = df_corrected[df_corrected["subject_id"].isin(common_ids)].copy()

df_original = df_original.sort_values("subject_id").reset_index(drop=True)
df_corrected = df_corrected.sort_values("subject_id").reset_index(drop=True)

# Build comparison dataframe
comparison = pd.DataFrame({"subject_id": df_original["subject_id"]})

for col in COMPARE_COLS:
    comparison[f"{col}_original"] = df_original[col].values
    comparison[f"{col}_corrected"] = df_corrected[col].values
    comparison[f"{col}_diff"] = (
        df_corrected[col].values - df_original[col].values
    )
    comparison[f"{col}_pct_change"] = np.where(
        df_original[col].values != 0,
        (comparison[f"{col}_diff"] / df_original[col].abs().values) * 100,
        np.nan,
    )

# Attach group label (MCI / Normal Aging)
comparison = comparison.merge(df_participants, on="subject_id", how="left")

# ---------------------------------------------------------------------------
# 5. Print per-patient comparison
# ---------------------------------------------------------------------------
print("=" * 100)
print("PER-PATIENT COMPARISON: word_counts_year_1 vs word_counts_year_1_corrected")
print("=" * 100)

for col in COMPARE_COLS:
    print(f"\n--- {col} ---")
    display_cols = [
        "subject_id",
        "group",
        f"{col}_original",
        f"{col}_corrected",
        f"{col}_diff",
        f"{col}_pct_change",
    ]
    print(comparison[display_cols].to_string(index=False))

# ---------------------------------------------------------------------------
# 6. Group-level summary (MCI vs Normal Aging)
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("GROUP-LEVEL SUMMARY (MCI vs Normal Aging)")
print("=" * 100)

for col in COMPARE_COLS:
    print(f"\n--- {col} ---")

    for group_name in ["MCI", "Normal Aging"]:
        grp = comparison[comparison["group"] == group_name]

        if grp.empty:
            print(f"\n  [{group_name}] No patients found.")
            continue

        original_vals = grp[f"{col}_original"]
        corrected_vals = grp[f"{col}_corrected"]
        diff_vals = grp[f"{col}_diff"]
        pct_vals = grp[f"{col}_pct_change"]

        print(f"\n  [{group_name}]  (n = {len(grp)})")
        print(f"    Original   — mean: {original_vals.mean():.4f},  std: {original_vals.std():.4f},  "
              f"min: {original_vals.min()},  max: {original_vals.max()}")
        print(f"    Corrected  — mean: {corrected_vals.mean():.4f},  std: {corrected_vals.std():.4f},  "
              f"min: {corrected_vals.min()},  max: {corrected_vals.max()}")
        print(f"    Difference — mean: {diff_vals.mean():.4f},  std: {diff_vals.std():.4f},  "
              f"min: {diff_vals.min()},  max: {diff_vals.max()}")
        print(f"    % Change   — mean: {pct_vals.mean():.2f}%,  std: {pct_vals.std():.2f}%,  "
              f"min: {pct_vals.min():.2f}%,  max: {pct_vals.max():.2f}%")

# ---------------------------------------------------------------------------
# 7. Save comparison to CSV
# ---------------------------------------------------------------------------
out_path = os.path.join(OUTPUT_DIR, "comparison_word_counts_original_vs_corrected.csv")
comparison.to_csv(out_path, index=False)
print(f"\nComparison saved to {out_path}")
