## Alzheimers Fluency Tests Processing
You are pocessing transcripts from a cognitive assessment for Alzheimer's disease. 
The transcript contains the timestamped utterances (questions and answers) from both the examiner and the participant

We are interested in extracting the responses to the following standard verbal fluency prompts which would go something like

1. "Let's begin. Tell me all the words you can, as quickly as you can, that begin with the letter 'F'. Ready? Begin." 
2. "Now I want you to do the same for another letter. The next letter is 'L.' Ready? Begin."  
3. "Now I want you to name things that belong to another category: Animals. You will have one minute. I want you to tell me all the animals you can think of in one minute. Ready? Begin."
4. "Now I want you to name things that belong to another category: Vegetables. You will have one minute. I want you to tell me all the vegetables you can think of in one minute. Ready? Begin." 


The responses to these standard prompts should be around 60 seconds from the start of the response.

Your task is to extract the following from the transcripts.
1. The full response by the participant for the prompt ignoring the other speaker
2. The extracted answer from the full response where one omits the filler words and incorrect answers. Repititions are allowed.
2. The starting and ending timestamp of the participant response.
3. The individual pauses considered by the participant where the pause is atleast a second.


**Rules**
1. Do not infer missing speech.
2. Return an empty string if no response exists.
3. Do not output anything except the final json.
4. Use R1, R2, R3 and R4 to represent the 4 task prompts respectively.

**Output JSON Schema**
```json
{
  "responses": {
    "R1": {
      "full_response": "string",
      "response_timestamps": [
        {
        "start": "string",
        "end": "string"
        },
        {"start": "string",
          "end": "string",
        },
      ],
      "extracted_answer": ["word1", "word2", ],
      "pauses": [ // Have the start and end time for each pause during the response
        {
          "start": "string",
          "end": "string"
        },
        {
          "start": "string",
          "end": "string"
        },
      ]
    },
    "R2": { },
    "R3": { },
    "R4": { }
  }
}

```

**Notes**
1. When we ask for the full response, we mean the entire answer provided by the participant including any errors. 
2. A pause occurs when there is more than a second difference between the start timestamp of one line and end time stamp of the previous line.
3. The extracted answer for R1 and R2 should not include names of people or places or numbers. 
4. Make proper judgement of when the response begins and ends. Many times it there would be a sentence including phrases like "Stop here" to indicate end of sentence.