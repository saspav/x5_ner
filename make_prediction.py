import json
from pathlib import Path
from tqdm import tqdm
from data_process import DataProcessBert

with open("data/test.json", encoding='utf-8') as file:
    data = json.load(file)

dcp = DataProcessBert(load_model=True)

# Make predictions on the test set and format the results
predictions_output = []
for item in tqdm(data, total=len(data)):
    item_text = item["text"]

    # поиск информации в тексте
    all_results = dcp.get_all_entity_groups(item_text)

    predictions_output.append({'text': item_text, 'entities': all_results})

# Save the predictions_output to a JSON file with UTF-8 encoding
output_file = "data/submission.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(predictions_output, f, ensure_ascii=False, indent=4)

print(f"Predictions saved to {output_file}")
