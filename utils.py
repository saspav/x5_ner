import json
import torch

# Define the unique tags in the dataset based on the provided entity groups
entity_groups = ['ORG', 'NUM', 'NAME_EMPLOYEE', 'LINK', 'DATE', 'ACRONYM', 'MAIL',
                 'TELEPHONE', 'TECH', 'NAME', 'PERCENT']
unique_tags = ['O'] + [f'B-{entity}' for entity in entity_groups] + [f'I-{entity}' for entity
                                                                     in entity_groups]

# Create tag2id and id2tag dictionaries
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}

# Verify mappings
print("tag2id:", tag2id)
print("id2tag:", id2tag)

# maximum len
max_len = 512


# Function to create labels for each symbol
def create_label(text, entities):
    label = ['O'] * len(text)
    for entity in entities:
        for i in range(entity['start'], min(entity['end'], max_len)):
            if i == entity['start']:
                label[i] = 'B-' + entity['entity_group']
            else:
                label[i] = 'I-' + entity['entity_group']
    return label


# Data collator for token classification without tokenizer
class CustomDataCollator:
    def __init__(self, padding_token_id):
        self.padding_token_id = padding_token_id

    def __call__(self, features):
        max_length = max(len(feature['input_ids']) for feature in features)
        batch = {'input_ids': [], 'labels': [], 'attention_mask': []}

        for feature in features:
            input_ids = feature['input_ids']
            labels = feature['labels']
            attention_mask = [1] * len(input_ids)

            # Padding
            padding_length = max_length - len(input_ids)
            input_ids = input_ids + [self.padding_token_id] * padding_length
            labels = labels + [-100] * padding_length
            attention_mask = attention_mask + [0] * padding_length

            batch['input_ids'].append(input_ids)
            batch['labels'].append(labels)
            batch['attention_mask'].append(attention_mask)

        batch = {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
        return batch


# Function to convert predictions to entities
def predictions_to_entities(text, predictions):
    entities = []
    current_entity = None
    for idx, label in enumerate(predictions):
        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {"entity_group": label[2:], "start": idx, "end": idx + 1}
        elif label.startswith("I-") and current_entity and current_entity[
            "entity_group"] == label[2:]:
            current_entity["end"] = idx + 1
        else:
            if current_entity:
                current_entity['word'] = text[current_entity['start']:current_entity['end']]
                entities.append(current_entity)
                current_entity = None
    if current_entity:
        current_entity['word'] = text[current_entity['start']:current_entity['end']]
        entities.append(current_entity)
    return entities


# Function to apply post-processing to ensure correct B- and I- tagging
def post_process_predictions(predictions):
    corrected_predictions = []
    previous_label = 'O'
    for label in predictions:
        if label.startswith('I-') and previous_label == 'O':
            # If we see an I- without a preceding B-, change it to B-
            corrected_predictions.append('B-' + label[2:])
        else:
            corrected_predictions.append(label)
        previous_label = label
    return corrected_predictions


def store_char_vocab(train_json, test_json=None, output_file='vocab.json'):
    # Create a character vocabulary
    def create_char_vocab(dataset):
        characters = set()
        for item in dataset:
            characters.update(item)
        char2id = {char: idx for idx, char in enumerate(sorted(characters), start=1)}
        char2id['<pad>'] = 0  # Adding a padding token
        id2char = {idx: char for char, idx in char2id.items()}
        return char2id, id2char

    # Prepare dataset for creating the vocabulary
    dataset_dict = {
        'tokens': []
    }

    # Load training JSON data
    for item in train_json:
        tokens = list(item['text'])
        dataset_dict['tokens'].append(tokens)

    # Load test JSON data if provided
    if test_json is not None:
        for item in test_json:
            tokens = list(item['text'])  # Assuming the 'text' key contains the tokens
            dataset_dict['tokens'].append(tokens)

    # Create character vocabulary
    char2id, id2char = create_char_vocab(dataset_dict['tokens'])

    # Store the vocabulary to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({'char2id': char2id, 'id2char': id2char}, f, ensure_ascii=False, indent=4)
    return char2id, id2char


def load_char_vocab(input_file='vocab.json'):
    # Load character vocabulary from a JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
        char2id = vocab['char2id']
        id2char = vocab['id2char']
    return char2id, id2char
