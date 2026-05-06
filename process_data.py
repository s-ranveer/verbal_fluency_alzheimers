# This is the file for processing the data regarding Alzheimer's speech dataset
import json
from pydantic import BaseModel
from typing import List, Dict, Optional
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

save_path = ""
transcript_path = ""

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



if __name__ == "__main__":
    print("Loading system prompt for processing Alzheimer's speech dataset...")
    with open("prompts/process_data.md", "r") as file:
        system_prompt = file.read()
    
    print("Loading the transcript...")
    with open("transcript_path", "r") as file:
        transcript = file.read()
    
    print("Creating the user prompt...")
    user_prompt = f"Transcript:\n{transcript}\n\nPlease process the above transcript according to the system prompt instructions."
    
    print("Running Inference on vllm")
    
    # Initialize vLLM model
    model_path = "google/medgemma-27b-text-it"
    llm = LLM(model=model_path, tensor_parallel_size=4, max_num_seqs=1)
    
    # Get tokenizer for chat template
    tokenizer = llm.get_tokenizer()
    
    # Prepare messages for chat
    messages = [
        {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Sampling parameters
    sampling_params = SamplingParams(temperature=0.0, max_tokens=30000, structured_outputs=StructuredOutputsParams(json=OutputSchema.model_json_schema()))
    
    # Generate response
    outputs = llm.generate([prompt], sampling_params)
    
    # Extract the response text
    response_text = outputs[0].outputs[0].text
    
    print(response_text)

    # Save the response to a JSON file

    with open(save_path, "w") as json_file:
        json_text = json.loads(response_text)  # Convert string to JSON
        json.dump(json_text, json_file, indent=4)
    
    # Shutdown vllm
    del llm
