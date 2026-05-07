# TODO: Implement the file where the feedback from the symbolic grounding, mainly the invalidated words along with the llm extracted json and transcript section would be passed for possible correction and re-extraction of features. This would be used in the second pass of the feature extraction process where we can use the feedback to refine our feature extraction and possibly improve the performance of our models.
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_prompt_path", type=str, required=True, help="Path to the input data containing the LLM extracted JSON and transcript sections.")
parser.add_argument("--input_transcriptions_path", type=str, required=True, help="Path to the input data containing the LLM extracted JSON and transcript sections.")
parser.add_argument("--input_extracted_response_path", type=str, required=True, help="Path to the input data containing the LLM extracted JSON and transcript sections.")
parser.add_argument("--input_invalidated_words_path", type=str, required=True, help="Path to the input data containing the invalidated words from the symbolic grounding.")

args = parser.parse_args()


def create_full_prompt(prompt_data, transcript_data, extracted_response_data, invalidated_words):
    return f"{prompt_data}\n{transcript_data}\n{extracted_response_data}\nInvalidated Words: {invalidated_words}"

if __name__ == "__main__":
    # Load the input prompt to be used with the LLM
    with open(args.input_prompt_path, "r") as f:
        prompt_data = f.read()
    
    # Load the binned data which contains the invalidated words
    invalidated_words_data = pd.read_csv(args.input_invalidated_words_path)
    
    # The only columns relevant here are the patient_id and any columns which contain not_words in the column name, which would contain the invalidated words for each response key
    invalidated_words_data = invalidated_words_data[["patient_id"] + [col for col in invalidated_words_data.columns if "not_words" in col]]

    # We would rename the columns other than patient_id to extract the response key from the column name, which would be in the format {response_key}_word_frequency_not_words
    for col in invalidated_words_data.columns:
        if col != "patient_id":
            response_key = col.split("_")[0]
            invalidated_words_data.rename(columns={col: f"{response_key}"}, inplace=True)    
    
    # We would convert the dataframe to a dictionary with patient_id as the key and the invalidated words for each response key as the value
    invalidated_words_dict = invalidated_words_data.set_index("patient_id").T.to_dict()


    # Create the dataset for the second pass of feature extraction using the feedback from the symbolic grounding
    df = None
    for patient_id, invalidated_words in invalidated_words_dict.items():
        # We would load the corresponding transcript for the patient
        transcript_path = os.path.join(args.input_transcriptions_path, f"{patient_id}.txt")
        with open(transcript_path, "r") as f:
            transcript_data = f.read()
        
        # We would also load the corresponding extracted response for the patient
        extracted_response_path = os.path.join(args.input_extracted_response_path, f"{patient_id}.json")
        with open(extracted_response_path, "r") as f:
            extracted_response_data = f.read()
        

        # We would then create the full prompt for the LLM using the input prompt, transcript data, extracted response data and the invalidated words
        full_prompt = create_full_prompt(prompt_data, transcript_data, extracted_response_data, invalidated_words)
        if df is None:
            df = pd.DataFrame({"patient_id": [patient_id], "full_prompt": [full_prompt]})
        else:
            df = pd.concat([df, pd.DataFrame({"patient_id": [patient_id], "full_prompt": [full_prompt]})], ignore_index=True)
    

    # Once we have the full dataset ready, we would use vllm to generate the corrected and refined features based on the full prompt for each patient. This would involve passing the full prompt to the LLM and extracting the relevant features from the generated response. The generated response would be expected to contain the corrected and refined features based on the feedback provided in the full prompt.
    # TODO: Implement the code for vllm later
