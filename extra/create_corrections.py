# This is the file for creating the csv where we will create the log file used for marking down files corrected manually

import os
import csv

# Path to the transcriptions folder
transcriptions_path = os.path.join(os.path.dirname(__file__), 'transcriptions')

# Output CSV file
csv_file = os.path.join(os.path.dirname(__file__), 'corrections_log.csv')

# Open CSV file for writing
with open(csv_file, 'w', newline='') as csvfile:
    fieldnames = ['transcription_id', 'person_correcting', 'status']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Write header
    writer.writeheader()
    
    # List to collect rows
    rows = []
    
    # Walk through the transcriptions directory
    for root, dirs, files in os.walk(transcriptions_path):
        for file in files:
            if file.endswith('.txt'):
                # Get the year folder name
                year_folder = os.path.basename(root)
                # Extract transcription_id (prepend year and remove .txt extension)
                transcription_id = f"{year_folder}_{file[:-4]}"
                # Initialize person_correcting and status
                person_correcting = ""
                status = "pending"
                # Collect row
                row = {
                    'transcription_id': transcription_id,
                    'person_correcting': person_correcting,
                    'status': status
                }
                rows.append(row)
    
    # Sort rows by transcription_id
    rows.sort(key=lambda x: x['transcription_id'])
    
    # Write sorted rows
    for row in rows:
        writer.writerow(row)

print(f"CSV file '{csv_file}' created successfully.")