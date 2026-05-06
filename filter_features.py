"""
Filter features_year_1.csv to only include patients present in
outputs/transcriptions_corrected/year_1/ and write to features_year_1_filtered.csv.
"""

import os
import re
import csv

FEATURES_CSV = "outputs/features_year_1.csv"
CORRECTED_DIR = "outputs/transcriptions_corrected/year_1"
OUTPUT_CSV = "outputs/features_year_1_filtered.csv"


def get_corrected_patient_ids():
    """Return set of patient IDs (as strings) from corrected transcriptions directory."""
    ids = set()
    for f in os.listdir(CORRECTED_DIR):
        if f.endswith("_processed.json"):
            match = re.search(r"RWRAD_(\d+)", f)
            if match:
                ids.add(match.group(1))
    return ids


def main():
    corrected_ids = get_corrected_patient_ids()

    with open(FEATURES_CSV, "r") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        all_rows = list(reader)

    filtered = [row for row in all_rows if row["patient_id"].strip() in corrected_ids]

    print(f"Total patients in features CSV: {len(all_rows)}")
    print(f"Patients in corrected transcriptions: {len(corrected_ids)}")
    print(f"Patients after filtering: {len(filtered)}")
    print(f"Filtered patient IDs: {sorted(row['patient_id'].strip() for row in filtered)}")

    with open(OUTPUT_CSV, "w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered)

    print(f"\nFiltered features written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
