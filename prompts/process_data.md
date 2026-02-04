## Alzheimers Fluency Tests Processing
You are pocessing transcripts from a cognitive assessment for Alzheimer's disease. 
The transcript contains the timestamped utterances (questions and answers) from both the examiner and the participant

We are interested in extracting the responses to the following standard verbal fluency prompts which would go something like

1. "Now I want you to name things that belong to another category: Animals. You will have one minute. I want you to tell me all the animals you can think of in one minute. Ready? Begin."
2. "Now I want you to name things that belong to another category: Vegetables. You will have one minute. I want you to tell me all the vegetables you can think of in one minute. Ready? Begin." 
3. "Let's begin. Tell me all the words you can, as quickly as you can, that begin with the letter 'F'. Ready? Begin." 
4. "Now I want you to do the same for another letter. The next letter is 'L.' Ready? Begin."  

The responses to these standard prompts should be around 1 minute or a bit more.

Your task is to extract the following from the transcripts.
1. The full response by the participant for the prompt ignoring the other speaker
2. The extracted answer from the full response where one omits the filler words and incorrect answers.
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
      "response_timestamp": {
        "start": "string",
        "end": "string"
      },
      "extracted_answer": "string",
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
