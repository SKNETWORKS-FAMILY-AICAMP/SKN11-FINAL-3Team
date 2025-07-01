import json
import re

def convert_dialogues_efficient(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f) 

    converted = []
    for item in raw_data:
        content = item["content"].replace('\n', ' ')
        turns = re.split(r'(고객:|상담원:)', content)

        dialogue = []
        for i in range(1, len(turns), 2):
            role = "user" if turns[i] == "고객:" else "assistant"
            utterance = turns[i+1].strip()
            dialogue.append({"role": role, "content": utterance})

        converted.append(dialogue)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

convert_dialogues_efficient("../data/asia_culture.json", "../data/multiturn.json")
