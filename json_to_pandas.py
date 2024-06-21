import json
import pandas as pd
from pathlib import Path


def json_to_pandas(path_to_json=None, print_text=False, save_to_excel=False):
    """
    Чтение json-файла и сохранение его в ДФ
    :param path_to_json: путь к файлу
    :param print_text: печатать содержимое json-файла
    :param save_to_excel: сохранить в эксель
    :return: ДФ
    """
    # если файл не находится -> смотрим в текущем каталоге
    if path_to_json is None or not Path(path_to_json).is_file():
        path_to_json = 'train.json'

    # Открываем файл train.json
    with open(path_to_json, encoding='utf-8') as file:
        data = json.load(file)
        print(f'Количество записей в train.json: {len(data)}')

    data_df = pd.DataFrame(columns=['idx', 'text', 'entity_group', 'word', 'start', 'end'])

    idx_text = 0
    # Перебираем каждую запись в файле
    for record in data:
        idx_text += 1
        # Получаем значения полей text и entities
        item_text = record['text']
        entities = record['entities']

        # Выводим значения полей
        if print_text:
            print('Text:', item_text)

        for entity in entities:
            if 'word' not in entity:
                print('Text:', item_text)
                print('entity:', entity)

            entity_group = entity['entity_group']
            word = entity.get('word', 'NONE')
            start = entity['start']
            end = entity['end']

            if print_text:
                print('Entity Group:', entity_group)
                print('Word:', word)
                print('Start:', start)
                print('End:', end)

            data_df.loc[len(data_df)] = [idx_text, item_text, entity_group, word, start, end]

        if print_text:
            print('-' * 60)

    if save_to_excel:
        data_df.to_excel(path_to_json.replace('.json', '.xlsx'), index=False)

    return data_df


if __name__ == "__main__":
    # json_file = 'output_xml-roberta-large-ner-russian.json'
    # json_file = 'data/test.json'
    json_file = 'data/submission.json'
    df = json_to_pandas(path_to_json=json_file, save_to_excel=True)

    print(df.head(7))
