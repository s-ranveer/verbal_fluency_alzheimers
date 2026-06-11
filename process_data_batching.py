# This is the file for processing the data regarding Alzheimer's speech dataset
import json
import os
import argparse
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Optional
from pathlib import Path
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


class Timestamp(BaseModel):
    start: str
    end: str

class Response(BaseModel):
    full_response: str
    response_timestamps: List[Timestamp]  # {"start": "timestamp", "end": "timestamp"}
    extracted_answer: List[str] # List of words from the response

class OutputSchema(BaseModel):
    responses: Dict[str, Optional[Response]]


def load_transcripts(transcript_dir: str) -> List[tuple[str, str]]:
    """Load all transcript files from directory.
    
    Returns:
        List of tuples: (filename, transcript_content)
    """
    transcript_path = Path(transcript_dir)
    transcripts = []
    
    for file_path in transcript_path.glob("*.txt"):
        with open(file_path, "r") as f:
            transcripts.append((file_path.name, f.read()))
    
    return transcripts

def create_prompts(system_prompt: str, transcripts: List[tuple[str, str]], tokenizer) -> List[tuple[str, str]]:
    """Create formatted prompts for all transcripts.
    
    Returns:
        List of tuples: (filename, formatted_prompt)
    """
    prompts = []
    
    for filename, transcript in transcripts:
        user_prompt = f"Transcript:\n{transcript}\n\nPlease process the above transcript according to the system prompt instructions."
        
        messages = [
            {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append((filename, prompt))
    
    return prompts

def write_failure(
    output_dir: str,
    filename: str,
    reason: str,
    error: str,
    response_text: str = "",
    finish_reason: Optional[str] = None,
    generated_tokens: Optional[int] = None,
) -> None:
    base_name = os.path.splitext(filename)[0]
    output_file = os.path.join(output_dir, f"{base_name}_processed.json")
    raw_file = None

    if response_text:
        raw_file = os.path.join(output_dir, f"{base_name}_raw_response.txt")
        with open(raw_file, "w") as raw:
            raw.write(response_text)

    failure_data = {
        "generation_failed": True,
        "filename": filename,
        "reason": reason,
        "error": error,
        "finish_reason": finish_reason,
        "generated_tokens": generated_tokens,
        "raw_response_file": raw_file,
        "responses": {},
    }
    with open(output_file, "w") as json_file:
        json.dump(failure_data, json_file, indent=4)

def save_response(
    output_dir: str,
    filename: str,
    response_text: str,
    finish_reason: Optional[str] = None,
    generated_tokens: Optional[int] = None,
) -> bool:
    output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_processed.json")
    try:
        json_data = json.loads(response_text)
        OutputSchema.model_validate(json_data)
        with open(output_file, "w") as json_file:
            json.dump(json_data, json_file, indent=4)
        return True
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for file {filename}: {e}")
        print("Saving failure metadata and raw response text.")
        write_failure(
            output_dir,
            filename,
            reason="json_decode_error",
            error=str(e),
            response_text=response_text,
            finish_reason=finish_reason,
            generated_tokens=generated_tokens,
        )
        return False
    except ValidationError as e:
        print(f"Error validating JSON schema for file {filename}: {e}")
        print("Saving failure metadata and raw response text.")
        write_failure(
            output_dir,
            filename,
            reason="schema_validation_error",
            error=str(e),
            response_text=response_text,
            finish_reason=finish_reason,
            generated_tokens=generated_tokens,
        )
        return False

parser = argparse.ArgumentParser()
parser.add_argument("--transcript_dir", type=str, required=True, help="Directory containing transcript files.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed outputs.")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing transcripts.")
parser.add_argument("--model", type=str, default="google/medgemma-27b-text-it", help="vLLM model to use for processing.")
parser.add_argument("--max_model_len", type=int, default=None, help="Optional prompt+output context length reserved by vLLM. If omitted, vLLM uses the model default.")
parser.add_argument("--max_tokens", type=int, default=6000, help="Maximum generated tokens per transcript.")
parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
args = parser.parse_args()

if __name__ == "__main__":
    # Configuration
    BATCH_SIZE = args.batch_size
    TRANSCRIPT_DIR = args.transcript_dir
    OUTPUT_DIR = args.output_dir

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading system prompt for processing Alzheimer's speech dataset...")
    with open("prompts/process_data.md", "r") as file:
        system_prompt = file.read()
    
    print("Loading transcripts...")
    transcripts = load_transcripts(TRANSCRIPT_DIR)
    print(f"Found {len(transcripts)} transcripts to process")
    
    print("Initializing vLLM model...")
    model_path = "google/medgemma-27b-text-it"
    llm_kwargs = {
        "model": model_path,
        "tensor_parallel_size": 4,
        "max_num_seqs": BATCH_SIZE,  # Set to batch size for batching
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    llm = LLM(**llm_kwargs)
    
    tokenizer = llm.get_tokenizer()
    
    print("Creating prompts...")
    prompts = create_prompts(system_prompt, transcripts, tokenizer)
    
    # Sampling parameters
    sampling_params = SamplingParams(
        seed=args.seed,
        max_tokens=args.max_tokens,
        structured_outputs=StructuredOutputsParams(json=OutputSchema.model_json_schema())
    )
    
    # Process in batches
    success_count = 0
    failed_count = 0
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i + BATCH_SIZE]
        batch_filenames = [filename for filename, _ in batch]
        batch_prompts = [prompt for _, prompt in batch]
        
        print(f"Processing batch {i//BATCH_SIZE + 1}/{(len(prompts) + BATCH_SIZE - 1)//BATCH_SIZE} ({len(batch_prompts)} items)...")
        
        # Generate responses for batch
        try:
            outputs = llm.generate(batch_prompts, sampling_params)
        except Exception as e:
            print(f"Generation failed for batch {i//BATCH_SIZE + 1}: {e}")
            for filename in batch_filenames:
                write_failure(
                    OUTPUT_DIR,
                    filename,
                    reason="generation_exception",
                    error=str(e),
                )
                failed_count += 1
                print(f"Failed: {filename}")
            print(f"Saved batch {i//BATCH_SIZE + 1}")
            continue
        
        # Save results after each batch so completed work is preserved.
        for filename, output in zip(batch_filenames, outputs):
            completion = output.outputs[0]
            response_text = completion.text
            token_ids = getattr(completion, "token_ids", None)
            generated_tokens = len(token_ids) if token_ids is not None else None
            is_valid = save_response(
                OUTPUT_DIR,
                filename,
                response_text,
                finish_reason=getattr(completion, "finish_reason", None),
                generated_tokens=generated_tokens,
            )
            if is_valid:
                success_count += 1
                print(f"Completed: {filename}")
            else:
                failed_count += 1
                print(f"Failed: {filename}")
        print(f"Saved batch {i//BATCH_SIZE + 1}")

    print(f"\nProcessed {success_count} transcripts successfully; {failed_count} failed.")
    print(f"Results saved to: {OUTPUT_DIR}")
    
    # Shutdown vllm
    del llm
