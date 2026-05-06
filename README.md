## NeSyQuaKE Framework
This repository contains the files required for running the NeSyQuaKE framework to extract qualitative influences from raw audio dataset.
The dataset will not be made available at the moment

## Files Included
### Information Extraction
1) ```transcribe_whisperx.py```: Run this file to transcribe the audio files to text and generate the transcript with timestamps
2) ```process_data_batching.py```:  Run this file next to extract the structured output from the text using the prompts provided in the prompt folder
3) ```construct_features.py```: Run this file after processing the data using an LLM to create the feature dataset

### QuaKE
1) ```phonemic.py.py```: This is the file storing the influence graph elicited from the expert.
2) ```compute_QI.py```: This is the file required to compute the monotonic influences
3) ```test_quake.py```: This is the file for loading the dataset, fitting the network adn extracting the qualitative influences.
To run the QuaKE code, ```python test_quake.py --fluency phonemic```

