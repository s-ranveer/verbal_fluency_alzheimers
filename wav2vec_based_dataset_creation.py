import os
import numpy as np
import pandas as pd
import torch
import librosa

from transformers import Wav2Vec2Processor, Wav2Vec2Model

# --------------------------------------------------
# Configuration
# --------------------------------------------------

AUDIO_DIR = "audio_files"
OUTPUT_FILE = "recording_embeddings.parquet"

CHUNK_SECONDS = 30
SAMPLE_RATE = 16000

MODEL_NAME = "facebook/wav2vec2-base-960h"

# --------------------------------------------------
# Load model
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

# --------------------------------------------------
# Embedding function
# --------------------------------------------------

@torch.no_grad()
def embed_chunk(audio_chunk):
    inputs = processor(
        audio_chunk,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    outputs = model(**inputs)

    embedding = (
        outputs.last_hidden_state
        .mean(dim=1)
        .squeeze(0)
        .cpu()
        .numpy()
    )

    return embedding

# --------------------------------------------------
# Process one recording
# --------------------------------------------------

def process_recording(audio_path):

    print(f"Processing {audio_path}")

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE
    )

    chunk_size = CHUNK_SECONDS * SAMPLE_RATE

    chunk_embeddings = []

    for start in range(
        0,
        len(audio),
        chunk_size
    ):

        chunk = audio[start:start + chunk_size]

        # Skip tiny tail chunks
        if len(chunk) < SAMPLE_RATE:
            continue

        emb = embed_chunk(chunk)

        chunk_embeddings.append(emb)

    chunk_embeddings = np.array(chunk_embeddings)

    # Recording-level representation
    mean_emb = chunk_embeddings.mean(axis=0)
    std_emb = chunk_embeddings.std(axis=0)

    recording_embedding = np.concatenate(
        [mean_emb, std_emb]
    )

    return recording_embedding

# --------------------------------------------------
# Main loop
# --------------------------------------------------

rows = []

for filename in os.listdir(AUDIO_DIR):

    if not filename.lower().endswith(
        (".wav", ".mp3", ".flac", ".m4a")
    ):
        continue

    filepath = os.path.join(
        AUDIO_DIR,
        filename
    )

    embedding = process_recording(filepath)

    row = {
        "recording_id": filename
    }

    for i, value in enumerate(embedding):
        row[f"f_{i}"] = float(value)

    rows.append(row)

# --------------------------------------------------
# Save
# --------------------------------------------------

df = pd.DataFrame(rows)

df.to_parquet(
    OUTPUT_FILE,
    index=False
)

print(df.shape)
print(f"Saved to {OUTPUT_FILE}")