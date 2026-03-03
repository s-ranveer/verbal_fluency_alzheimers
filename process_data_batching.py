# This is the file for processing the data regarding Alzheimer's speech dataset
import json
import os
from pydantic import BaseModel
from typing import List, Dict, Optional
from pathlib import Path
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

# Describe the structure for the output that we need to process
class Pause(BaseModel):
    start: str
    end: str

class Timestamp(BaseModel):
    start: str
    end: str

class Response(BaseModel):
    full_response: str
    response_timestamps: List[Timestamp]  # {"start": "timestamp", "end": "timestamp"}
    extracted_answer: List[str] # List of words from the response
    pauses: List[Pause]

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

if __name__ == "__main__":
    # Configuration
    BATCH_SIZE = 4  # Adjust based on your GPU memory
    TRANSCRIPT_DIR = ""
    OUTPUT_DIR = ""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading system prompt for processing Alzheimer's speech dataset...")
    with open("prompts/process_data.md", "r") as file:
        system_prompt = file.read()
    
    print("Loading transcripts...")
    transcripts = load_transcripts(TRANSCRIPT_DIR)
    print(f"Found {len(transcripts)} transcripts to process")
    
    print("Initializing vLLM model...")
    model_path = "google/medgemma-27b-text-it"
    llm = LLM(
        model=model_path, 
        tensor_parallel_size=4, 
        max_num_seqs=BATCH_SIZE  # Set to batch size for batching
    )
    
    tokenizer = llm.get_tokenizer()
    
    print("Creating prompts...")
    prompts = create_prompts(system_prompt, transcripts, tokenizer)
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=30000, 
        structured_outputs=StructuredOutputsParams(json=OutputSchema.model_json_schema())
    )
    
    # Process in batches
    all_results = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i + BATCH_SIZE]
        batch_filenames = [filename for filename, _ in batch]
        batch_prompts = [prompt for _, prompt in batch]
        
        print(f"Processing batch {i//BATCH_SIZE + 1}/{(len(prompts) + BATCH_SIZE - 1)//BATCH_SIZE} ({len(batch_prompts)} items)...")
        
        # Generate responses for batch
        outputs = llm.generate(batch_prompts, sampling_params)
        
        # Store results with filenames
        for filename, output in zip(batch_filenames, outputs):
            response_text = output.outputs[0].text
            all_results.append((filename, response_text))
            print(f"Completed: {filename}")
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for filename, response_text in all_results:
        output_file = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}_processed.json")
        try:
            json_data = json.loads(response_text)  # Assuming response_text is a JSON string
            with open(output_file, "w") as json_file:
                json.dump(json_data, json_file, indent=4)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for file {filename}: {e}")
            print("Saving raw response text instead.")
            with open(output_file, "w") as json_file:
                json_file.write(response_text)

    print(f"\nProcessed {len(all_results)} transcripts successfully!")
    print(f"Results saved to: {OUTPUT_DIR}")
    
    # Shutdown vllm
    del llm
