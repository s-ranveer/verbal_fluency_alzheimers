# This is the file for using WhisperX to transcribe audio files. We want the timestamps as well as the speaker labels.
import torch
import torch.serialization
from omegaconf import ListConfig
from omegaconf.base import ContainerMetadata
torch.serialization.add_safe_globals([ListConfig, ContainerMetadata])

# Patch torch.load to default weights_only=False for compatibility with PyTorch 2.6+
original_load = torch.load
def patched_load(f, map_location=None, pickle_module=None, *, weights_only=None, **kwargs):
    if weights_only is None:
        weights_only = False
    return original_load(f, map_location, pickle_module, weights_only=weights_only, **kwargs)
torch.load = patched_load

import os
import whisperx
import gc
import argparse
from whisperx.diarize import DiarizationPipeline
from tqdm import tqdm


def transcribe_with_speakers(audio_path, output_file, hf_token, language=None):
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32  # reduce if low on GPU mem
    compute_type = "float16" if device == "cuda" else "int8"

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("large-v3", device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size, language=language)

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # 3. Assign speaker labels and run diarization
    diarize_model = DiarizationPipeline(model_name="pyannote/speaker-diarization-3.1", use_auth_token=hf_token, device=device)
    diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=2)
    
    
    # Assign speakers to words
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # 4. Save the transcript with speakers
    with open(output_file, "w") as f:
        for segment in result["segments"]:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            speaker = segment.get("speaker", "UNKNOWN")
            f.write(f"[{start:.2f} - {end:.2f}] {speaker}: {text}\n")

    # Clean up
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio files with WhisperX and speaker diarization.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode: process only one sample file.")    
    parser.add_argument("--language", type=str, default="en", help="Language code for transcription (e.g., 'en' for English).")    
    args = parser.parse_args()

    print("Starting the WhisperX transcription script with speaker diarization...")
    if args.debug:
        print("Debug mode: Processing only one sample file.")
    
    # Load HF token once
    with open("./hugging_face_token.txt", "r") as token_file:
        hf_token = token_file.read().strip()
    
    os.makedirs("./transcriptions", exist_ok=True)
    file_dir = "./audio"

    year_mapping = {"year1": "year_1", "year2": "year_2", "year3": "year_3"}
    processed_count = 0
    
    for year_dir in os.listdir(file_dir):
        if os.path.isdir(os.path.join(file_dir, year_dir)):
            trans_dir = year_mapping.get(year_dir, year_dir)
            os.makedirs(f"./transcriptions/{trans_dir}", exist_ok=True)
            dir_path = os.path.join(file_dir, year_dir)
            total_files = len([f for f in os.listdir(dir_path) if f.endswith(('.wav', '.mp3', '.flac', '.m4a'))])
            for file in tqdm(os.listdir(dir_path), desc=f"Processing files in {trans_dir}", total=total_files):
                if file.endswith(('.wav', '.mp3', '.flac', '.m4a')):
                    output_file = os.path.join("./transcriptions", trans_dir, f"{os.path.splitext(file)[0]}_whisperx.txt")
                    audio_path = os.path.join(dir_path, file)
                    transcribe_with_speakers(audio_path, output_file, hf_token, args.language)
                    processed_count += 1
                    
                    if args.debug and processed_count >= 1:
                        print("Debug mode: Stopping after processing one file.")
                        break
            
            if args.debug and processed_count >= 1:
                break
