"""
This is the file where we would take the LLM extraction, as well as the symbolic feedback, and then use that to correct the LLM's output if possible.
"""
import argparse
import copy
import json
import os
import re
from typing import Optional
from pydantic import BaseModel, Field, ValidationError

class Correction(BaseModel):
    incorrect_word: str
    correct_word: str
    confidence: int = Field(ge=1, le=5)
    justification: str

class ResponseCorrections(BaseModel):
    R1: list[Correction] = Field(default_factory=list)
    R2: list[Correction] = Field(default_factory=list)
    R3: list[Correction] = Field(default_factory=list)
    R4: list[Correction] = Field(default_factory=list)

class PatientCorrection(BaseModel):
    corrections: ResponseCorrections

def parse_args():
    parser = argparse.ArgumentParser(description="This script takes the LLM extraction and symbolic feedback, and then uses that to correct the LLM's output if possible.")
    parser.add_argument("--llm_output_dir", type=str, required=True, help="The directory where the LLM extractions are stored.")
    parser.add_argument("--symbolic_feedback_file", type=str, required=True, help="The file where the symbolic feedback is stored.")
    parser.add_argument("--corrected_output_dir", type=str, required=True, help="The directory where the corrected outputs will be stored.")
    parser.add_argument("--prompt_template_file", type=str, required=True, help="The file where the prompt template is stored.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for correction generation.")
    parser.add_argument("--model", type=str, default="google/medgemma-27b-text-it", help="vLLM model to use for correction generation.")
    parser.add_argument("--max_model_len", type=int, default=None, help="Optional prompt+output context length reserved by vLLM. If omitted, vLLM uses the model default.")
    parser.add_argument("--seed", type=int, default=0, help="The random seed for reproducibility.")
    parser.add_argument("--max_tokens", type=int, default=6000, help="The maximum number of tokens to generate for the correction.")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="Tensor parallel size for vLLM.")
    parser.add_argument("--build_dataset_only", action="store_true", help="Only build and save reprompt_dataset.json; do not run vLLM correction generation.")
    parser.add_argument("--debug_single_case", action="store_true", help="Run only one patient case for debugging.")
    parser.add_argument("--debug_patient_id", type=str, default=None, help="Patient id to use with --debug_single_case. If omitted, the first reprompt-needed patient is used.")
    return parser.parse_args()

# This is the method where we will apply the corrections to the LLM output. This involves mainly updating the extracted answer in the original LLM output with the corrected answer, and also adding a new field for the corrections that were made, which would include the incorrect word, the correct word, the confidence and the justification for the correction.
def apply_corrections(llm_output, corrections):
    corrected_output = copy.deepcopy(llm_output)
    responses = corrected_output.get("responses") or {}
    correction_map = normalize_correction_map(corrections)

    for response_key, response_corrections in correction_map.items():
        response = responses.get(response_key)
        if not response:
            continue

        extracted_answer = response.get("extracted_answer") or []
        updated_answer = list(extracted_answer)

        for correction in response_corrections:
            incorrect_word = correction["incorrect_word"].strip().lower()
            correct_word = correction["correct_word"].strip()
            for idx, word in enumerate(updated_answer):
                if str(word).strip().lower() == incorrect_word:
                    updated_answer[idx] = correct_word

        response["extracted_answer"] = updated_answer
        response["symbolic_feedback_corrections"] = response_corrections

    corrected_output["symbolic_feedback_corrections"] = correction_map
    return corrected_output

def normalize_correction_map(corrections):
    if isinstance(corrections, PatientCorrection):
        corrections = corrections.model_dump()

    if isinstance(corrections, dict) and "corrections" in corrections:
        corrections = corrections["corrections"]

    normalized = {}
    for response_key in ["R1", "R2", "R3", "R4"]:
        response_corrections = corrections.get(response_key, []) if isinstance(corrections, dict) else []
        normalized[response_key] = [
            correction.model_dump() if isinstance(correction, Correction) else correction
            for correction in response_corrections
        ]
    return normalized

def empty_patient_correction():
    return PatientCorrection(corrections={response_key: [] for response_key in ["R1", "R2", "R3", "R4"]})

def construct_prompt_input(llm_output, symbolic_feedback):
    final_prompt_dict = {"R1": {}, "R2": {}, "R3": {}, "R4": {}}
    prompt = {
        "R1": "Let's begin. Tell me all the words you can, as quickly as you can, that begin with the letter 'F'. Ready? Begin.",
        "R2": "Now I want you to do the same for another letter. The next letter is 'L.' Ready? Begin.",
        "R3": "Now I want you to name things that belong to another category: Animals. You will have one minute. I want you to tell me all the animals you can think of in one minute. Ready? Begin.",
        "R4": "Now I want you to name things that belong to another category: Vegetables. You will have one minute. I want you to tell me all the vegetables you can think of in one minute. Ready? Begin."
    }
    llm_output = llm_output.get("responses", "")
    for k in llm_output.keys():
        if k in final_prompt_dict.keys():
            final_prompt_dict[k]["prompt"] = prompt.get(k, "")
            final_prompt_dict[k]["full_response"] = llm_output[k].get("full_response", "")
            final_prompt_dict[k]["word_list"] = llm_output[k].get("extracted_answer", [])
            final_prompt_dict[k]["rejected_words"] = symbolic_feedback.get(f"{k}_word_frequency_not_words", [])

    return final_prompt_dict

def create_dataset_for_correction(llm_dir, symbolic_feedback_file, prompt_template_file):
    with open(prompt_template_file, 'r') as f:
        prompt_template = f.read().strip()

    with open(symbolic_feedback_file, 'r') as f:
        symbolic_feedback = json.load(f)

    prompt_dataset = {}
    for file in sorted(os.listdir(llm_dir)):
        if not file.endswith(".json"):
            continue

        # Process each LLM output file
        patient_id_match = re.search(r'\d+', file)
        if patient_id_match is None:
            continue

        p_id = patient_id_match.group()  # The numerical component of the file name, which we would use to find the corresponding symbolic feedback file
        llm_output_path = os.path.join(llm_dir, file)
        # We would load the corresponding jsons
        with open(llm_output_path, 'r') as f:
            llm_output = json.load(f)
        
        # Use the patient id to find the corresponding symbolic feedback for this LLM output
        p_dict = symbolic_feedback.get(p_id, {})
        # Now, we would look at the different keys in the nested dictionary keyed on word_frequency_not_words
        p_dict = p_dict.get("word_frequency_not_words", {})

        # If any of the keys in the nested dictionary are not an empty list, we will consider that the LLM output is not correct, and we will need to reprompt
        reprompt_needed = False
        for k in p_dict.keys():
            if p_dict[k]:
                reprompt_needed = True
                break
        
        if reprompt_needed:
            # If a reproompt is needed, we woiuld construct the prompt input using the original prompt template, the LLM output and the symbolic feedback
            prompt_input = construct_prompt_input(llm_output, p_dict)
            full_prompt = (
                f"{prompt_template}\n\n"
                f"{'*' * 50}\n"
                f"Please provide the corrections for the following patient responses:\n\n"
                f"{json.dumps(prompt_input, indent=2)}"
            )
        else:
            prompt_input = None  # This is the case when no reprompt is needed, and we can consider the LLM output to be correct as is.
            full_prompt = None
        
        prompt_dataset[p_id] = {
            "source_file": file,
            "original_llm_output": llm_output,
            "prompt_input": prompt_input,
            "full_prompt": full_prompt,
            "reprompt_needed": reprompt_needed
        }
    return prompt_dataset

def select_debug_case(prompt_dataset, patient_id=None):
    if patient_id is not None:
        normalized_patient_id = patient_id.zfill(3) if patient_id.isdigit() else patient_id
        if normalized_patient_id not in prompt_dataset:
            raise ValueError(f"Debug patient id not found in prompt dataset: {patient_id}")
        return {normalized_patient_id: prompt_dataset[normalized_patient_id]}

    for candidate_patient_id, item in prompt_dataset.items():
        if item["reprompt_needed"]:
            return {candidate_patient_id: item}

    for candidate_patient_id, item in prompt_dataset.items():
        return {candidate_patient_id: item}

    raise ValueError("Prompt dataset is empty; no debug case is available.")

def create_chat_prompts(prompt_dataset, tokenizer):
    prompts = []
    for patient_id, item in prompt_dataset.items():
        if not item["reprompt_needed"]:
            continue

        messages = [
            {"role": "user", "content": item["full_prompt"]}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append((patient_id, prompt))
    return prompts

def write_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def write_failure(
    output_dir: str,
    patient_id: str,
    source_file: str,
    reason: str,
    error: str,
    response_text: str = "",
    finish_reason: Optional[str] = None,
    generated_tokens: Optional[int] = None,
) -> None:
    base_name = os.path.splitext(source_file)[0]
    failure_path = os.path.join(output_dir, "failures", f"{base_name}_correction_failure.json")
    raw_path = None

    if response_text:
        raw_path = os.path.join(output_dir, "raw_responses", f"{base_name}_correction_raw_response.txt")
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        with open(raw_path, "w") as raw_file:
            raw_file.write(response_text)

    write_json(
        failure_path,
        {
            "generation_failed": True,
            "patient_id": patient_id,
            "source_file": source_file,
            "reason": reason,
            "error": error,
            "finish_reason": finish_reason,
            "generated_tokens": generated_tokens,
            "raw_response_file": raw_path,
        },
    )

def save_patient_outputs(
    output_dir: str,
    patient_id: str,
    patient_item: dict,
    patient_correction: PatientCorrection,
    response_text: str = "",
    finish_reason: Optional[str] = None,
    generated_tokens: Optional[int] = None,
) -> None:
    source_file = patient_item["source_file"]
    corrected_output = apply_corrections(patient_item["original_llm_output"], patient_correction)
    corrected_output["correction_generation_metadata"] = {
        "patient_id": patient_id,
        "source_file": source_file,
        "reprompt_needed": patient_item["reprompt_needed"],
        "finish_reason": finish_reason,
        "generated_tokens": generated_tokens,
    }

    write_json(os.path.join(output_dir, source_file), corrected_output)

    correction_record = {
        "patient_id": patient_id,
        "source_file": source_file,
        "reprompt_needed": patient_item["reprompt_needed"],
        "corrections": patient_correction.model_dump()["corrections"],
        "finish_reason": finish_reason,
        "generated_tokens": generated_tokens,
    }
    write_json(
        os.path.join(output_dir, "corrections", f"{os.path.splitext(source_file)[0]}_corrections.json"),
        correction_record,
    )

    if response_text:
        raw_path = os.path.join(output_dir, "raw_responses", f"{os.path.splitext(source_file)[0]}_correction_raw_response.txt")
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        with open(raw_path, "w") as raw_file:
            raw_file.write(response_text)

def save_generated_response(output_dir, patient_id, patient_item, response_text, finish_reason=None, generated_tokens=None):
    source_file = patient_item["source_file"]
    try:
        json_data = json.loads(response_text)
        patient_correction = PatientCorrection.model_validate(json_data)
        save_patient_outputs(
            output_dir,
            patient_id,
            patient_item,
            patient_correction,
            response_text=response_text,
            finish_reason=finish_reason,
            generated_tokens=generated_tokens,
        )
        return True
    except json.JSONDecodeError as e:
        print(f"Error decoding correction JSON for patient {patient_id}: {e}")
        write_failure(
            output_dir,
            patient_id,
            source_file,
            reason="json_decode_error",
            error=str(e),
            response_text=response_text,
            finish_reason=finish_reason,
            generated_tokens=generated_tokens,
        )
        save_patient_outputs(output_dir, patient_id, patient_item, empty_patient_correction())
        return False
    except ValidationError as e:
        print(f"Error validating correction schema for patient {patient_id}: {e}")
        write_failure(
            output_dir,
            patient_id,
            source_file,
            reason="schema_validation_error",
            error=str(e),
            response_text=response_text,
            finish_reason=finish_reason,
            generated_tokens=generated_tokens,
        )
        save_patient_outputs(output_dir, patient_id, patient_item, empty_patient_correction())
        return False

def save_uncorrected_outputs(prompt_dataset, corrected_output_dir):
    for patient_id, item in prompt_dataset.items():
        if item["reprompt_needed"]:
            continue
        save_patient_outputs(
            corrected_output_dir,
            patient_id,
            item,
            empty_patient_correction(),
        )

def run_batched_corrections(prompt_dataset, corrected_output_dir, llm, tokenizer, sampling_params, batch_size):
    prompts = create_chat_prompts(prompt_dataset, tokenizer)
    success_count = 0
    failed_count = 0
    total_batches = (len(prompts) + batch_size - 1) // batch_size

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        batch_patient_ids = [patient_id for patient_id, _ in batch]
        batch_prompts = [prompt for _, prompt in batch]
        batch_num = i // batch_size + 1

        print(f"Processing correction batch {batch_num}/{total_batches} ({len(batch_prompts)} items)...")

        try:
            outputs = llm.generate(batch_prompts, sampling_params)
        except Exception as e:
            print(f"Generation failed for correction batch {batch_num}: {e}")
            for patient_id in batch_patient_ids:
                patient_item = prompt_dataset[patient_id]
                write_failure(
                    corrected_output_dir,
                    patient_id,
                    patient_item["source_file"],
                    reason="generation_exception",
                    error=str(e),
                )
                save_patient_outputs(corrected_output_dir, patient_id, patient_item, empty_patient_correction())
                failed_count += 1
            print(f"Saved correction batch {batch_num}")
            continue

        for patient_id, output in zip(batch_patient_ids, outputs):
            patient_item = prompt_dataset[patient_id]
            completion = output.outputs[0]
            response_text = completion.text
            token_ids = getattr(completion, "token_ids", None)
            generated_tokens = len(token_ids) if token_ids is not None else None
            is_valid = save_generated_response(
                corrected_output_dir,
                patient_id,
                patient_item,
                response_text,
                finish_reason=getattr(completion, "finish_reason", None),
                generated_tokens=generated_tokens,
            )
            if is_valid:
                success_count += 1
                print(f"Completed correction: {patient_id}")
            else:
                failed_count += 1
                print(f"Failed correction: {patient_id}")

        print(f"Saved correction batch {batch_num}")

    return success_count, failed_count

def main():
    args = parse_args()
    # This is where we would implement the logic to take the LLM extraction and symbolic feedback, and then use that to correct the LLM's output if possible.
    reprompt_dataset = create_dataset_for_correction(args.llm_output_dir, args.symbolic_feedback_file, args.prompt_template_file)
    if args.debug_single_case:
        reprompt_dataset = select_debug_case(reprompt_dataset, args.debug_patient_id)
        debug_patient_id = next(iter(reprompt_dataset))
        args.batch_size = 1
        print(f"Debug single-case mode enabled for patient {debug_patient_id}")

    os.makedirs(args.corrected_output_dir, exist_ok=True)
    output_path = os.path.join(args.corrected_output_dir, "reprompt_dataset.json")
    with open(output_path, 'w') as f:
        json.dump(reprompt_dataset, f, indent=2)

    reprompt_count = sum(1 for item in reprompt_dataset.values() if item["reprompt_needed"])
    print(f"Wrote {len(reprompt_dataset)} patient records to {output_path}")
    print(f"Reprompt needed for {reprompt_count} patient records")

    if args.build_dataset_only:
        print("Dataset-only mode enabled; skipping vLLM correction generation.")
        raise SystemExit(0)

    # Once we have the dataset, we will use it to correct the LLM's output using vllm, We would create a batch of prompts for the cases where reprompting is needed, and then we would use vLLM to generate the corrected outputs. We would then write the corrected outputs to a new directory.
    # In case, the reprompting is not needed, we would simply copy the original LLM output to the new directory as the corrected output, since we are considering it to be correct as is.
    save_uncorrected_outputs(reprompt_dataset, args.corrected_output_dir)

    if reprompt_count == 0:
        print("No records need reprompting; copied original outputs to corrected output directory.")
        raise SystemExit(0)

    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_num_seqs": args.batch_size,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    
    sampling_params = SamplingParams(
        seed=args.seed,
        max_tokens=args.max_tokens,
        structured_outputs=StructuredOutputsParams(json=PatientCorrection.model_json_schema())
    )

    success_count, failed_count = run_batched_corrections(
        reprompt_dataset,
        args.corrected_output_dir,
        llm,
        tokenizer,
        sampling_params,
        args.batch_size,
    )

    print(f"\nGenerated corrections for {success_count} patients successfully; {failed_count} failed.")
    print(f"Corrected outputs saved to: {args.corrected_output_dir}")

    del llm

if __name__ == "__main__":
    main()
