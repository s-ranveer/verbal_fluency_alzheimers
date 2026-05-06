# This is the file for naive rules based extraction of responses from text
import os
import re
import json
import argparse
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)

_lemmatizer = WordNetLemmatizer()

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="data", help="Path to the data directory")
parser.add_argument("--output_path", type=str, default="data/rules_based/extracted", help="Path to the output directory")
args = parser.parse_args()

word_list_letters = ["l", "f"]
RESPONSE_SLOT_ORDER = [
    ("R1", "word_list_f"),
    ("R2", "word_list_l"),
    ("R3", "animal"),
    ("R4", "vegetable"),
]
PROMPT_PATTERNS = [
    ("word_list_f", re.compile(r"\bletter f\b|\bbegin with the letter f\b")),
    ("word_list_l", re.compile(r"\bnext letter is l\b|\bletter l\b")),
    ("animal", re.compile(r"\bcategory is animals\b|\bcategory of animals\b|\bcategory animals\b|\bthe category is animals\b")),
    ("vegetable", re.compile(r"\bcategory vegetables\b|\bcategory of vegetables\b|\bthe category vegetables\b|\bthe category of vegetables\b")),
]


def detect_prompt_type(cleaned_lines, idx):
    line = cleaned_lines[idx]
    for expected_type, pattern in PROMPT_PATTERNS:
        if pattern.search(line):
            return expected_type

    if "begin with the letter" in line or re.search(r"\bnext letter is\b", line):
        lookahead = " ".join(cleaned_lines[idx:min(len(cleaned_lines), idx + 3)])
        if re.search(r"\bf\b", lookahead):
            return "word_list_f"
        if re.search(r"\bl\b", lookahead):
            return "word_list_l"

    return None


def empty_response(match_type):
    return {
        "match_type": match_type,
        "full_response": "",
        "response_timestamps": [],
        "extracted_answer": [],
        "pauses": [],
    }


def tokenize_line(text):
    # Drop possessive "'s" so words like "eagle's" do not become the false token "eagles".
    normalized = re.sub(r"[’']s\b", "", text.lower())
    normalized = re.sub(r'[^a-zA-Z\s]', ' ', normalized)
    return [word for word in normalized.split() if word]


def relevant_components_found(segment, animal_set, vegetable_set, word_list_letters,
                               word_list_threshold=0.50, min_category_matches=3,
                               min_span_density=0.10, min_unique_matches=3,
                               min_word_list_count=3, min_valid_word_list_matches=2,
                               max_word_list_gap=5,
                               expected_type=None):
    """
    Returns a list of (match_type, extracted_answer, start_word_idx, end_word_idx) tuples.
    A single segment can yield multiple matches when adjacent task responses are
    bundled into the same transcript window.
    """
    words = []
    word_line_indices = []  # Maps each word in 'words' back to its line index in segment["cleaned"]
    for i, line in enumerate(segment["cleaned"]):
        line_words = tokenize_line(line)
        words.extend(line_words)
        word_line_indices.extend([i] * len(line_words))

    if len(words) == 0:
        return [], []

    # Animal/vegetable: span from first to last category word.
    # Valid only if >= min_category_matches words found AND span density >= min_span_density
    # (density filter kills false positives where animal words are scattered across a long span).
    def in_set(w, s):
        return w in s or _lemmatizer.lemmatize(w, pos='n') in s

    animal_indices    = [i for i, w in enumerate(words) if in_set(w, animal_set)]
    vegetable_indices = [i for i, w in enumerate(words) if in_set(w, vegetable_set)]

    matches = []

    if expected_type in (None, "animal") and len(animal_indices) >= min_category_matches:
        span_length = animal_indices[-1] - animal_indices[0] + 1
        density = len(animal_indices) / span_length
        # Also require >= min_unique_matches distinct animals (catches repeated incidental mentions)
        unique_animals = len({_lemmatizer.lemmatize(words[i], pos='n') for i in animal_indices})
        if density >= min_span_density and unique_animals >= min_unique_matches:
            span_words = words[animal_indices[0]:animal_indices[-1] + 1]
            matches.append((
                "animal",
                [w for w in span_words if in_set(w, animal_set)],
                animal_indices[0],
                animal_indices[-1]
            ))

    if expected_type in (None, "vegetable") and len(vegetable_indices) >= min_category_matches:
        span_length = vegetable_indices[-1] - vegetable_indices[0] + 1
        density = len(vegetable_indices) / span_length
        unique_vegetables = len({_lemmatizer.lemmatize(words[i], pos='n') for i in vegetable_indices})
        if density >= min_span_density and unique_vegetables >= min_unique_matches:
            span_words = words[vegetable_indices[0]:vegetable_indices[-1] + 1]
            matches.append((
                "vegetable",
                [w for w in span_words if in_set(w, vegetable_set)],
                vegetable_indices[0],
                vegetable_indices[-1]
            ))

    # Per-letter word list: use the span from first to last matching word so the
    # ratio is not diluted by neighboring prompt text or a later task in the same segment.
    allowed_letters = word_list_letters
    if expected_type is not None and expected_type.startswith("word_list_"):
        allowed_letters = [expected_type.rsplit("_", 1)[-1]]

    for letter in allowed_letters:
        valid_letter_indices = [
            i for i, w in enumerate(words)
            if w[0] == letter and len(w) > 1 and w != "letter"
        ]
        if len(valid_letter_indices) < min_word_list_count:
            continue

        clusters = []
        cluster_start = valid_letter_indices[0]
        cluster_prev = valid_letter_indices[0]
        cluster_count = 1
        for idx in valid_letter_indices[1:]:
            if idx - cluster_prev <= max_word_list_gap:
                cluster_prev = idx
                cluster_count += 1
            else:
                clusters.append((cluster_start, cluster_prev, cluster_count))
                cluster_start = idx
                cluster_prev = idx
                cluster_count = 1
        clusters.append((cluster_start, cluster_prev, cluster_count))

        best_cluster = None
        best_cluster_score = None
        for span_start, span_end, cluster_count in clusters:
            if cluster_count < min_word_list_count:
                continue
            span_length = span_end - span_start + 1
            density = cluster_count / span_length
            if density < word_list_threshold:
                continue

            span_words = words[span_start:span_end + 1]
            extracted_words = [w for w in span_words if w[0] == letter and len(w) > 1 and w != "letter"]
            if len(extracted_words) < min_valid_word_list_matches:
                continue

            cluster_score = (len(extracted_words), density, -span_start)
            if best_cluster is None or cluster_score > best_cluster_score:
                best_cluster = (span_start, span_end, extracted_words)
                best_cluster_score = cluster_score

        if best_cluster is None:
            continue

        span_start, span_end, extracted_words = best_cluster
        matches.append((
            f"word_list_{letter}",
            extracted_words,
            span_start,
            span_end
        ))

    return matches, word_line_indices


def parse_timestamp(line):
    """Extract (start, end) in seconds from a '[start - end] text' line. Returns None if no timestamp."""
    m = re.match(r'\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\]', line)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def strip_timestamp(line):
    """Remove the [start - end] prefix, preserving original casing."""
    return re.sub(r'^\[.*?\]\s*', '', line).strip()


def extract_components(transcript, max_gap_seconds=120, max_segment_seconds=90):
    lines = transcript.split("\n")

    def clean(line):
        return re.sub(r'\[.*?\]\s*', '', line).strip().lower()

    cleaned    = [clean(l) for l in lines]
    originals  = [strip_timestamp(l) for l in lines]   # original casing, no timestamp prefix
    timestamps = [parse_timestamp(l) for l in lines]   # (start, end) or None per line

    segments = []

    def append_segment(cleaned_lines, original_lines, timestamp_lines, expected_type):
        if cleaned_lines:
            segments.append({
                "cleaned": cleaned_lines,
                "original": original_lines,
                "timestamps": timestamp_lines,
                "expected_type": expected_type,
            })

    prompt_hits = []
    seen_types = set()
    for i, _line in enumerate(cleaned):
        expected_type = detect_prompt_type(cleaned, i)
        if expected_type is not None and expected_type not in seen_types:
            prompt_hits.append((i, expected_type))
            seen_types.add(expected_type)

    prompt_hits.sort()

    def next_stop_after(start_idx, limit_idx=None):
        upper = len(cleaned) if limit_idx is None else limit_idx
        for j in range(start_idx + 1, upper):
            if re.search(r'\bstop\b|\bend\b', cleaned[j]):
                return j + 1
        return None

    def next_gap_after(start_idx, limit_idx=None):
        upper = len(cleaned) if limit_idx is None else limit_idx
        prev_end_time = None
        for j in range(start_idx, upper):
            ts = timestamps[j]
            if ts is None:
                continue
            start_time, end_time = ts
            if prev_end_time is not None and (start_time - prev_end_time) > max_gap_seconds:
                return j
            prev_end_time = end_time
        return upper

    def time_limited_end(start_idx, limit_idx=None):
        upper = len(cleaned) if limit_idx is None else limit_idx
        start_time = None
        for j in range(start_idx, upper):
            ts = timestamps[j]
            if ts is not None:
                start_time = ts[0]
                break
        if start_time is None:
            return upper

        for j in range(start_idx, upper):
            ts = timestamps[j]
            if ts is None:
                continue
            if ts[1] - start_time >= max_segment_seconds:
                return j + 1
        return upper

    for idx, (start_idx, expected_type) in enumerate(prompt_hits):
        next_prompt_idx = prompt_hits[idx + 1][0] if idx + 1 < len(prompt_hits) else None
        stop_end_idx = next_stop_after(start_idx, next_prompt_idx)
        if stop_end_idx is not None:
            end = stop_end_idx
        elif next_prompt_idx is not None:
            end = next_prompt_idx
        else:
            end = min(next_gap_after(start_idx), time_limited_end(start_idx))

        end = min(end, time_limited_end(start_idx, end))

        raw_segment_lines = list(range(start_idx, end))
        current_chunk_cleaned    = []
        current_chunk_originals  = []
        current_chunk_timestamps = []
        prev_end_time = None

        for idx in raw_segment_lines:
            ts = timestamps[idx]
            if ts is not None:
                start_time, end_time = ts
                if prev_end_time is not None and (start_time - prev_end_time) > max_gap_seconds:
                    append_segment(
                        current_chunk_cleaned,
                        current_chunk_originals,
                        current_chunk_timestamps,
                        expected_type,
                    )
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

        append_segment(
            current_chunk_cleaned,
            current_chunk_originals,
            current_chunk_timestamps,
            expected_type,
        )

    # Legacy fallback for transcripts that rely on generic begin/start cues.
    for i, line in enumerate(cleaned):
        if not re.search(r'\bbegin\b|\bstart\b', line):
            continue

        next_begin = None
        for j in range(i + 1, len(cleaned)):
            if re.search(r'\bbegin\b|\bstart\b', cleaned[j]):
                next_begin = j
                break

        stop_end_idx = next_stop_after(i, next_begin)
        candidate_end = next_begin if next_begin is not None else len(cleaned)
        time_end_idx = time_limited_end(i + 1, candidate_end)

        if stop_end_idx is not None:
            end = min(stop_end_idx, time_end_idx)
        else:
            end = min(candidate_end, time_end_idx)

        raw_segment_lines = list(range(i + 1, end))
        current_chunk_cleaned = []
        current_chunk_originals = []
        current_chunk_timestamps = []
        prev_end_time = None

        for idx in raw_segment_lines:
            ts = timestamps[idx]
            if ts is not None:
                start_time, end_time = ts
                if prev_end_time is not None and (start_time - prev_end_time) > max_gap_seconds:
                    append_segment(
                        current_chunk_cleaned,
                        current_chunk_originals,
                        current_chunk_timestamps,
                        None,
                    )
                    current_chunk_cleaned = []
                    current_chunk_originals = []
                    current_chunk_timestamps = []
                current_chunk_cleaned.append(cleaned[idx])
                current_chunk_originals.append(originals[idx])
                current_chunk_timestamps.append(ts)
                prev_end_time = end_time
            else:
                current_chunk_cleaned.append(cleaned[idx])
                current_chunk_originals.append(originals[idx])
                current_chunk_timestamps.append(None)

        append_segment(
            current_chunk_cleaned,
            current_chunk_originals,
            current_chunk_timestamps,
            None,
        )

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

            # Collect the first chronological segment per match_type. These tasks
            # occur once each in a fixed order, so later matches are more likely
            # to be incidental recalls than the intended fluency response.
            best_per_type = {}  # match_type -> entry dict

            for segment in segments:
                found_matches, word_line_indices = relevant_components_found(
                    segment, animal_set, vegetable_set, word_list_letters,
                    expected_type=segment.get("expected_type")
                )
                if not found_matches:
                    continue

                for match_type, extracted_answer, start_w, end_w in found_matches:
                    # Determine the line range for this specific task span
                    start_l = word_line_indices[start_w]
                    end_l   = word_line_indices[end_w]

                    # Trim full_response and timestamps to the lines covering the match span
                    task_lines = segment["original"][start_l:end_l + 1]
                    task_timestamps = [ts for ts in segment["timestamps"][start_l:end_l + 1] if ts is not None]

                    if extracted_answer and task_lines:
                        first_word = extracted_answer[0]
                        first_line = task_lines[0]
                        first_match = re.search(rf"\b{re.escape(first_word)}\b", first_line, flags=re.IGNORECASE)
                        if first_match:
                            task_lines[0] = first_line[first_match.start():]

                        last_word = extracted_answer[-1]
                        last_line = task_lines[-1]
                        last_matches = list(re.finditer(rf"\b{re.escape(last_word)}\b", last_line, flags=re.IGNORECASE))
                        if last_matches:
                            task_lines[-1] = last_line[:last_matches[-1].end()]

                    full_response = " ".join(task_lines)

                    if task_timestamps:
                        seg_start = str(task_timestamps[0][0])
                        seg_end   = str(task_timestamps[-1][1])
                    else:
                        seg_start = seg_end = None

                    entry = {
                        "match_type":          match_type,
                        "full_response":       full_response,
                        "response_timestamps": [{"start": seg_start, "end": seg_end}] if seg_start else [],
                        "extracted_answer":    extracted_answer,
                        "pauses":              [],
                    }

                    # Keep the first chronological match for each task type.
                    # These fluency prompts occur once each in a fixed order,
                    # so later matches are often incidental mentions from other tasks.
                    if match_type not in best_per_type:
                        best_per_type[match_type] = entry

            # Always emit the same response slots so downstream code keeps a stable
            # mapping even when one of the four tasks is not detected.
            responses = {
                slot: best_per_type.get(match_type, empty_response(match_type))
                for slot, match_type in RESPONSE_SLOT_ORDER
            }


            # Save JSON — same stem as input file
            stem = os.path.splitext(fname)[0]
            out_path = os.path.join(out_dir, f"{stem}.json")
            with open(out_path, "w") as f:
                json.dump({"responses": responses}, f, indent=4)

            types_found = sorted(best_per_type)
            print(f"  {fname} -> detected {len(best_per_type)}/4 responses {types_found}")
