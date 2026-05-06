"""
Read processed JSON files from outputs/transcriptions_wo_speakers/year_1/
and count extracted response words for R1, R2, R3, R4,
plus phonetic (R1+R2) and semantic (R3+R4) totals.
"""

import json
import os
import csv
import re

INPUT_DIR = "outputs/transcriptions_corrected/year_1"
OUTPUT_CSV = "outputs/word_counts_year_1_corrected.csv"


def extract_subject_id(filename):
    """Extract the numeric subject ID from filename like RWRAD_001CogTest_..."""
    match = re.search(r"RWRAD_(\d+)", filename)
    return match.group(1) if match else filename


def count_words(json_path):
    """Return word counts for R1-R4 from a processed JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    responses = data.get("responses", {})
    counts = {}
    for key in ["R1", "R2", "R3", "R4"]:
        extracted = responses.get(key, {}).get("extracted_answer", [])
        counts[key] = len(extracted)

    counts["phonetic"] = counts["R1"] + counts["R2"]
    counts["semantic"] = counts["R3"] + counts["R4"]
    return counts


def main():
    files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if f.endswith("_processed.json")
    )

    rows = []
    for filename in files:
        filepath = os.path.join(INPUT_DIR, filename)
        subject_id = extract_subject_id(filename)
        counts = count_words(filepath)
        rows.append({"subject_id": subject_id, **counts})

    # Print to console
    header = f"{'Subject':<10} {'R1':>4} {'R2':>4} {'R3':>4} {'R4':>4} {'Phonetic':>9} {'Semantic':>9}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['subject_id']:<10} {row['R1']:>4} {row['R2']:>4} "
            f"{row['R3']:>4} {row['R4']:>4} {row['phonetic']:>9} {row['semantic']:>9}"
        )

    # Write to CSV
    fieldnames = ["subject_id", "R1", "R2", "R3", "R4", "phonetic", "semantic"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
