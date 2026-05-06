# This is the file for naive rules based extraction of responses from text
import os
import re
import json
import argparse
import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)

_lemmatizer = WordNetLemmatizer()

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="data", help="Path to the data directory")
parser.add_argument("--output_path", type=str, default="data/rules_based/extracted", help="Path to the output directory")
args = parser.parse_args()

word_list_letters = ["l", "f"]


def relevant_components_found(segment, animal_set, vegetable_set, word_list_letters,
                               word_list_threshold=0.80, min_category_matches=3,
                               min_span_density=0.10, min_unique_matches=3,
                               min_word_list_count=3):
    """
    Returns a list of (match_type, extracted_answer) pairs.
    A single segment can yield multiple matches (e.g. L-words task + animal task
    that share the same begin/stop window in the transcript).

    match_type: "animal" | "vegetable" | "word_list_f" | "word_list_l" (etc.)
    extracted_answer: specific matched words for that type
    """
    words = []
    for line in segment["cleaned"]:
        words.extend([word for word in re.sub(r'[^a-zA-Z\s]', '', line).split(" ") if word != ""])

    if len(words) == 0:
        return []

    def in_set(w, s):
        return w in s or _lemmatizer.lemmatize(w, pos='n') in s

    matches = []

    # ── Animal ──────────────────────────────────────────────────────────────────
    animal_indices = [i for i, w in enumerate(words) if in_set(w, animal_set)]
    if len(animal_indices) >= min_category_matches:
        span_length   = animal_indices[-1] - animal_indices[0] + 1
        density       = len(animal_indices) / span_length
        unique_count  = len({_lemmatizer.lemmatize(words[i], pos='n') for i in animal_indices})
        if density >= min_span_density and unique_count >= min_unique_matches:
            span_words      = words[animal_indices[0]:animal_indices[-1] + 1]
            extracted       = [w for w in span_words if in_set(w, animal_set)]
            matches.append(("animal", extracted))

    # ── Vegetable ────────────────────────────────────────────────────────────────
    veg_indices = [i for i, w in enumerate(words) if in_set(w, vegetable_set)]
    if len(veg_indices) >= min_category_matches:
        span_length   = veg_indices[-1] - veg_indices[0] + 1
        density       = len(veg_indices) / span_length
        unique_count  = len({_lemmatizer.lemmatize(words[i], pos='n') for i in veg_indices})
        if density >= min_span_density and unique_count >= min_unique_matches:
            span_words      = words[veg_indices[0]:veg_indices[-1] + 1]
            extracted       = [w for w in span_words if in_set(w, vegetable_set)]
            matches.append(("vegetable", extracted))

    # ── Word list (per letter, span-based density) ───────────────────────────────
    # Use span density (first→last letter word) so the ratio isn't diluted by
    # animal/vegetable words that follow in the same segment.
    for letter in word_list_letters:
        letter_indices = [i for i, w in enumerate(words) if w[0] == letter]
        if len(letter_indices) < min_word_list_count:
            continue
        span_length = letter_indices[-1] - letter_indices[0] + 1
        density     = len(letter_indices) / span_length
        if density >= word_list_threshold:
            span_words = words[letter_indices[0]:letter_indices[-1] + 1]
            extracted  = [w for w in span_words if w[0] == letter]
            matches.append((f"word_list_{letter}", extracted))

    return matches



def parse_timestamp(line):
    """Extract (start, end) in seconds from a '[start - end] text' line. Returns None if no timestamp."""
    m = re.match(r'\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\]', line)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def strip_timestamp(line):
    """Remove the [start - end] prefix, preserving original casing."""
    return re.sub(r'^\[.*?\]\s*', '', line).strip()


def extract_components(transcript, max_gap_seconds=120):
    lines = transcript.split("\n")

    def clean(line):
        return re.sub(r'\[.*?\]\s*', '', line).strip().lower()

    cleaned    = [clean(l) for l in lines]
    originals  = [strip_timestamp(l) for l in lines]   # original casing, no timestamp prefix
    timestamps = [parse_timestamp(l) for l in lines]   # (start, end) or None per line

    segments = []

    for i, line in enumerate(cleaned):
        if not re.search(r'\bbegin\b|\bstart\b', line):
            continue

        # find next begin/stop boundary
        next_begin = None
        for j in range(i + 1, len(cleaned)):
            if re.search(r'\bbegin\b|\bstart\b', cleaned[j]):
                next_begin = j
                break

        next_stop = None
        for j in range(i + 1, len(cleaned)):
            if re.search(r'\bstop\b|\bend\b', cleaned[j]):
                next_stop = j
                break

        if next_stop is not None and (next_begin is None or next_stop < next_begin):
            end = next_stop
        elif next_begin is not None:
            end = next_begin
        else:
            end = len(cleaned)

        # Split on timestamp gaps > max_gap_seconds
        raw_segment_lines = list(range(i + 1, end))
        current_chunk_cleaned    = []
        current_chunk_originals  = []
        current_chunk_timestamps = []
        prev_end_time = None

        for idx in raw_segment_lines:
            ts = timestamps[idx]
            if ts is not None:
                start_time, end_time = ts
                if prev_end_time is not None and (start_time - prev_end_time) > max_gap_seconds:
                    if current_chunk_cleaned:
                        segments.append({
                            "cleaned":    current_chunk_cleaned,
                            "original":   current_chunk_originals,
                            "timestamps": current_chunk_timestamps,
                        })
                    current_chunk_cleaned    = []
                    current_chunk_originals  = []
                    current_chunk_timestamps = []
                current_chunk_cleaned.append(cleaned[idx])
                current_chunk_originals.append(originals[idx])
                current_chunk_timestamps.append(ts)
                prev_end_time = end_time
            else:
                current_chunk_cleaned.append(cleaned[idx])
                current_chunk_originals.append(originals[idx])
                current_chunk_timestamps.append(None)

        if current_chunk_cleaned:
            segments.append({
                "cleaned":    current_chunk_cleaned,
                "original":   current_chunk_originals,
                "timestamps": current_chunk_timestamps,
            })

    return segments


if __name__ == "__main__":
    # Load the animals and vegetables
    with open(os.path.join(args.data_path, "animal_groups.txt"), "r") as f:
        animals = f.read().splitlines()
    with open(os.path.join(args.data_path, "vegetable_groups.txt"), "r") as f:
        vegetables = f.read().splitlines()

    animal_set = set()
    for line in animals:
        category, animals_list = line.split(":")
        animal_set.update([x.strip() for x in animals_list.split(",")])

    vegetable_set = set()
    for line in vegetables:
        category, vegetables_list = line.split(":")
        vegetable_set.update([x.strip() for x in vegetables_list.split(",")])

    # Walk over all years / files under data_path/transcriptions_wo_speakers/
    transcription_root = os.path.join(args.data_path, "transcriptions_wo_speakers")

    for year in sorted(os.listdir(transcription_root)):
        year_dir = os.path.join(transcription_root, year)
        if not os.path.isdir(year_dir):
            continue

        out_dir = os.path.join(args.output_path, year)
        os.makedirs(out_dir, exist_ok=True)

        txt_files = sorted([f for f in os.listdir(year_dir) if f.endswith(".txt")])
        print(f"\n[{year}] Processing {len(txt_files)} files...")

        for fname in txt_files:
            fpath = os.path.join(year_dir, fname)
            with open(fpath, "r") as f:
                transcription = f.read()

            segments = extract_components(transcription)

            # Collect best segment per match_type (one response per category)
            best_per_type = {}  # match_type -> entry dict

            for segment in segments:
                match_type, words, extracted_answer = relevant_components_found(
                    segment, animal_set, vegetable_set, word_list_letters
                )
                if match_type is None:
                    continue

                # full_response: original casing lines joined
                full_response = " ".join(segment["original"])

                # response_timestamps: span from first to last timestamped line
                valid_ts = [ts for ts in segment["timestamps"] if ts is not None]
                if valid_ts:
                    seg_start = str(valid_ts[0][0])
                    seg_end   = str(valid_ts[-1][1])
                else:
                    seg_start = seg_end = None

                entry = {
                    "match_type":          match_type,
                    "full_response":       full_response,
                    "response_timestamps": [{"start": seg_start, "end": seg_end}] if seg_start else [],
                    "extracted_answer":    extracted_answer,
                    "pauses":              [],
                }

                # Keep this segment only if it has more extracted answers than the current best
                if (match_type not in best_per_type or
                        len(extracted_answer) > len(best_per_type[match_type]["extracted_answer"])):
                    best_per_type[match_type] = entry

            # Assign sequential keys in a stable order
            type_order = ["word_list_f", "word_list_l", "animal", "vegetable"]
            responses = {}
            response_idx = 1
            for t in type_order:
                if t in best_per_type:
                    responses[f"R{response_idx}"] = best_per_type[t]
                    response_idx += 1


            # Save JSON — same stem as input file
            stem = os.path.splitext(fname)[0]
            out_path = os.path.join(out_dir, f"{stem}.json")
            with open(out_path, "w") as f:
                json.dump({"responses": responses}, f, indent=4)

            types_found = sorted({v["match_type"] for v in responses.values()})
            print(f"  {fname} -> {len(responses)} responses {types_found}")




