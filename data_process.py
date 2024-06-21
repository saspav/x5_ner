import os
import re
import random
import json
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoModelForTokenClassification, pipeline
from transformers import BertTokenizer, BertForTokenClassification
from faker import Faker
from utils import id2tag, load_char_vocab, max_len
from utils import predictions_to_entities, post_process_predictions

# пытаемся импортировать самодельный экспорт в эксель с красивостями
try:
    from df_addons import df_to_excel
except ModuleNotFoundError:
    df_to_excel = lambda sdf, spt, *args, **kwargs: sdf.to_excel(spt, index=False)

__import__('warnings').filterwarnings("ignore")


class DataABC:
    def __init__(self):
        self.entity_groups = ('ACRONYM', 'DATE', 'LINK', 'MAIL', 'NAME', 'NAME_EMPLOYEE',
                              'NUM', 'ORG', 'PERCENT', 'TECH', 'TELEPHONE')


class DataProcess(DataABC):
    """ Класс для поиска сущностей"""

    def __init__(self, load_model=True, model_path=None):
        """
        Инициализация экземпляра класса
        :param load_model: Загружать в память NER модель
        """
        super().__init__()
        self.init_entity_groups()

        # Используем GPU если доступно
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # задаем путь для сохранения модели RoBERTa
        save_directory = os.environ.get('MODEL_SAVE_DIR', r"G:\python-datasets\cache")
        if not Path(r'G:\python-datasets').is_dir():
            save_directory = './models'

        os.makedirs(save_directory, exist_ok=True)

        # Печать промежуточных результатов
        self.print_ner_classifier_results = None
        # модель, какую будем использовать для NER
        self.ner_classifier = None
        self.model_name = "yqelz/xml-roberta-large-ner-russian"
        # self.model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
        # self.model_name = "Jean-Baptiste/roberta-large-ner-english"
        # self.model_name = "DeepPavlov/rubert-base-cased-conversational"

        # путь к локальной модели
        if model_path is None:
            self.model_path = "./models/xml-roberta-large-ner-russian"
        else:
            self.model_path = model_path
        if Path(self.model_path).is_dir():
            self.model_name = self.model_path

        # стратегия для объединения токенов
        self.strategy = 'none'
        # self.strategy = 'simple'
        # self.strategy = 'first'
        # self.strategy = 'average'
        # self.strategy = 'max'
        if load_model:
            self.init_ner_classifier(strategy=self.strategy)

    def init_ner_classifier(self, strategy=None):
        """
        Создаем экземпляр NER классификатора
        :param strategy: стратегия для объединения токенов
        :return: экземпляр NER классификатора
        """
        if strategy is None:
            strategy = self.strategy
        else:
            self.strategy = strategy

        self.ner_classifier = pipeline(
            # "ner",
            "token-classification",
            model=self.model_name,
            tokenizer=self.model_name,
            aggregation_strategy=strategy,
            device=self.device
        )
        return self.ner_classifier

    @staticmethod
    def get_name_group(entity_group):
        """Формирование имени атрибута сущности"""
        return f'TEMP_{entity_group}'

    def init_entity_groups(self):
        """Очистка временных списков со словарями сущностей"""
        for entity_group in self.entity_groups + ('NAME_ORG',):
            setattr(self, self.get_name_group(entity_group), list())

    @staticmethod
    def merge_tokens(results):
        """
        Объединение токенов и корректировка меток после работы TokenClassification
        для всех стратегий, кроме "none"
        :param results: выход TokenClassification
        :return: обновленный результат
        """
        merged = []
        current_entity = None
        current_word = ""
        current_start = None
        current_end = None

        for item in results:
            entity = item.get('entity_group')
            word = item['word']
            start = item['start']
            end = item['end']

            if entity == current_entity and current_end + 1 == start:
                # Продолжаем объединять токены
                current_word += word
                current_end = end
            else:
                if current_word:
                    merged.append({
                        'entity_group': current_entity,
                        'word': current_word,
                        'start': current_start,
                        'end': current_end
                    })
                # Начинаем новый токен
                current_entity = entity
                current_word = word
                current_start = start
                current_end = end

        if current_word:
            merged.append({
                'entity_group': current_entity,
                'word': current_word,
                'start': current_start,
                'end': current_end
            })

        return merged

    @staticmethod
    def merge_tokens_none(results):
        """
        Объединение токенов и корректировка меток после работы TokenClassification
        для стратегии "none"
        :param results: выход TokenClassification
        :return: обновленный результат
        """
        # Объединение сущностей
        merged_entities = []
        current_entity = None

        for result in results:
            entity = result['entity'][-3:]
            word = result['word']
            start = result['start']
            end = result['end']

            # Проверяем, является ли текущая сущность новой или продолжающейся
            if (current_entity and current_entity['entity'][-3:] == entity
                    and (start - current_entity['end']) < 2):
                # Добавим пробел, если начало и конец различаются (глюк токенизатора)
                space = ' ' * (start - current_entity['end']) * (word != '▁')
                # Добавляем слово к текущей сущности
                current_entity['word'] += space + word
                current_entity['end'] = end
            else:
                # Завершаем текущую сущность, если она есть
                if current_entity:
                    merged_entities.append(current_entity)
                # Начинаем новую сущность
                current_entity = {
                    'entity': entity,
                    'word': word,
                    'start': start,
                    'end': end
                }

        # Добавляем последнюю сущность
        if current_entity:
            merged_entities.append(current_entity)

        # Постобработка результатов объединения токенов: ФИО и организации
        # не могут иметь длину менее 3 символов,
        # если это не аббревиатура заглавными буквами
        merged = []
        for result in merged_entities:
            word = result['word'].replace('▁', ' ').replace('  ', ' ').strip()
            match = re.fullmatch(r'^[А-ЯЁ]+$', word)
            digit = re.search(r'\d+', word.replace(' ', ''))
            digit = len(digit.group()) if digit else 0
            if (len(word) > 5 and digit < 3) or (match and len(match.group()) > 2):
                merged.append({
                    'entity_group': result['entity'],
                    'word': word,
                    'start': result['start'],
                    'end': result['end']
                })
        return merged

    def get_persons_orgs(self, input_text):
        if self.ner_classifier is None:
            print('Модель для NER недоступна!!!')
            return []

        # Замена символов для лучшей токенизации
        input_text = str(input_text)
        input_text = input_text.replace('*', ' ')
        # input_text = input_text.replace('|', ' ')
        input_text = input_text.replace('•', ' ')
        input_text = input_text.replace('«', ' ').replace('»', ' ')
        input_text = input_text.replace('"', ' ')

        # поиск информации в тексте
        results = self.ner_classifier(input_text)

        if self.print_ner_classifier_results:
            print('results = self.ner_classifier(input_text):')
            print(*results, sep='\n')
            print()

        if self.strategy != "none":
            results = self.merge_tokens(results)
        else:
            results = self.merge_tokens_none(results)

        organizations = [item for item in results if item.get('entity_group') == 'ORG']
        locations = [item for item in results if item.get('entity_group') == 'LOC']

        patt_employee = re.compile(r'ответсвен|контакт|филиал|менеджер|сотрудник|уважением',
                                   flags=re.IGNORECASE)

        # если есть слово из шаблона - будем считать, что это сотрудники, а так просто персоны
        if patt_employee.search(input_text):
            name_entity_group = 'NAME_EMPLOYEE'
        else:
            name_entity_group = 'NAME'

        # name_entity_group = 'NAME'

        # получим список границ сущностей, чтобы исключить выбор из них групп цифр
        borders = self.get_entity_borders(['DATE', 'LINK', 'MAIL', 'PERCENT'])

        found_org = []
        for item in organizations:
            word, start, end = item['word'], item['start'], item['end']
            if '(' in word and ')' in word:
                continue

            new_word = word
            if '(' in word:
                new_word = word.split('(')[0].strip()

            pattern_org = re.compile(fr'{re.escape(new_word)} \([a-zA-Z -]+\)')
            for found in pattern_org.finditer(input_text):
                start, end = found.span()
                new_word = found.group()
                # print('founds:', found)
                found_org.append({
                    'entity_group': 'ORG',
                    'word': word,
                    'new_word': new_word,
                    'start': start,
                    'end': end})

        # выбираем из результатов только персон и организации
        pers_orgs = []
        for item in results:
            if item.get('entity_group') in ('PER', 'ORG'):
                entity_group = ('ORG', name_entity_group)[item['entity_group'] == 'PER']

                word, start, end = item['word'], item['start'], item['end']

                # Проверка для организаций на пересечение с 'LINK', 'MAIL' ->
                # если есть пересечение - это кусок от 'LINK', 'MAIL'
                # и 'ORG' должен содержать русские буквы или латинские,
                # но только Одну заглавную латинскую и остальные прописные буквы
                if entity_group == 'ORG':

                    # корректировка word с учетом окончания (english)
                    for found in found_org:
                        if word == found['word'] and start == found['start']:
                            word = found['new_word']
                            end = found['end']
                            break

                    # Нужно убрать захваченные первые латинские символы
                    # Паттерн для поиска заглавных латинских букв в начале строки
                    pattern = r'^[A-Z ]+(?=[А-ЯЁ])'
                    match = re.search(pattern, word)
                    if match:
                        edited_word = word.replace(match.group(0), "")
                        delta_start = match.end() - match.start()
                        if delta_start:
                            word = edited_word
                            start += delta_start

                    # Регулярные выражения для поиска заглавных и строчных латинских букв
                    upp_lat = re.compile(r'[A-Z]')
                    low_lat = re.compile(r'[a-z]')
                    upp_chr = re.compile(r'[A-ZА-ЯЁ]')
                    # Подсчёт заглавных и строчных латинских букв
                    upp_count = len(upp_lat.findall(word))
                    low_count = len(low_lat.findall(word))
                    # Первая буква заглавная
                    upp_start = len(upp_chr.findall(word[0]))
                    # Регулярное выражение для проверки наличия русских букв
                    rus_let = re.compile(r'[а-яА-ЯЁё]')
                    rus_count = len(rus_let.findall(word))

                    # NER выдает обычные слова как ORG: нужно сверить со словарем и их убрать
                    #

                    # проверка на пересечение: если границы изменились или
                    # нет русских букв и больше одной заглавной - это не 'ORG'
                    # или 'ORG' начинается с маленькой буквы
                    _start, _end = self.check_in_borders(borders, start, end)
                    if ((not rus_let.search(word) and upp_count > 1)
                            or not upp_start or (_start, _end) != (start, end)):
                        continue

                # костыль для корректировки слов, начинающих с точки
                dot_space = re.search(r'^[. ]+(?=\w)', word)
                if dot_space:
                    len_space = len(dot_space.group())
                    start += len_space
                    word = word[len_space:]

                # костыль для корректировки слов, оканчивающихся на точку
                dot_space = re.search(r'(?<=\w)(?:\.\w|[. ]+)$', word)
                if dot_space:
                    len_space = len(dot_space.group())
                    end -= len_space
                    word = word[:-len_space]

                pers_orgs.append({
                    'entity_group': entity_group,
                    'word': word,
                    'start': start,
                    'end': end})

        return pers_orgs

    def get_entity_borders(self, entity_groups):
        """
        Получение списка кортежей с границами сущностей
        :param entity_groups: список названий сущностей из self.entity_groups
        :return: список из кортежей начальной и конечной позиций сущностей
        """
        merged = []
        for entity_group in entity_groups:
            name_entity_attr = self.get_name_group(entity_group)
            if hasattr(self, name_entity_attr):
                merged.extend(getattr(self, name_entity_attr))
        borders = [(item['start'], item['end']) for item in merged]
        return borders

    @staticmethod
    def get_urls(input_text):
        """
        Получение URL адреса из текста
        :param input_text: текст
        :return: список словарей с адресом и границами
        """
        pattern = r'https?://[\w\.-]+(?:/[\w\.-]*)*\w/?'
        result = []
        for item in re.finditer(pattern, input_text):
            start, end = item.span()
            result.append({"entity_group": "LINK",
                           'word': item.group(), 'start': start, 'end': end})
        return result

    @staticmethod
    def get_emails(input_text):
        """
        Получение email адреса из текста
        :param input_text: текст
        :return: список словарей с адресом и границами
        """
        pattern = r'[\w.+-]+@[\w\.-]+\.[\w-]+\w'
        result = []
        for item in re.finditer(pattern, input_text):
            start, end = item.span()
            result.append({"entity_group": "MAIL",
                           'word': item.group(), 'start': start, 'end': end})
        return result

    @staticmethod
    def check_in_borders(borders, start, end):
        """
        Проверка попадания границ найденной сущности в диапазон значений других сущностей
        :param borders: список кортежей границ других сущностей
        :param start: начальная позиция
        :param end: конечная позиция
        :return: исходные границы или откорректированные, если они пересекаются
        """
        for a, b in sorted(borders):
            if a <= start <= b or a <= end <= b:
                if a <= start <= b:
                    start = b  # корректируем начальную позицию
                if a <= end <= b:
                    end = a - 1  # корректируем конечную позицию
                break
            #  другая сущность целиком лежит в границах найденной сущности
            elif start <= a <= end and start <= b <= end:
                start = end = -1
                break
        return start, end

    def find_pattern_text(self, entity_group, pattern, input_text, entity_groups):
        """
        Поиск текста по шаблону
        :param entity_group: группа сущности из self.entity_groups
        :param pattern: шаблон
        :param input_text: текст
        :param entity_groups: список групп для получения границ для проверки на пересечения
        :return: список из словарей с найденным текстом и границами
        """
        # получим список границ сущностей, чтобы исключить выбор из них групп цифр
        borders = self.get_entity_borders(entity_groups)
        result = []
        for item in re.finditer(pattern, input_text):
            start, end = item.span()
            new_start, new_end = self.check_in_borders(borders, start, end)
            if new_end - new_start > 2:
                word = item.group()
                result.append({"entity_group": entity_group,
                               'word': word[new_start - start:new_end - start],
                               'start': new_start, 'end': new_end})
        return result

    def get_nums(self, input_text):
        """
        Получение чисел из текста
        :param input_text: текст
        :return: список словарей с процентами и границами
        """
        # Отбираем "числа": сначала могут идти латинские буквы и затем не менее 3-х цифр и
        # не должно заканчиваться на маленькие русские буквы
        pattern = r'[a-zA-Z]*\d{3,}(?!0[а-яё])'
        entity_groups = ['DATE', 'LINK', 'MAIL', 'TELEPHONE']
        result = self.find_pattern_text("NUM", pattern, input_text, entity_groups)
        return result

    @staticmethod
    def get_percents(input_text):
        """
        Получение процентов из текста
        :param input_text: текст
        :return: список словарей с процентами и границами
        """
        pattern = r'[1-9]\d?[\.,]?\d*%'
        result = []
        for item in re.finditer(pattern, input_text):
            start, end = item.span()
            result.append({"entity_group": "PERCENT",
                           'word': item.group(), 'start': start, 'end': end})
        return result

    @staticmethod
    def get_dates(input_text):
        """
        Получение дат из текста
        :param input_text: текст
        :return: список словарей с датами и границами
        """
        pattern = r'(?:\d{2}[\./\\]){2}\d{4}'
        result = []
        for item in re.finditer(pattern, input_text):
            start, end = item.span()
            result.append({"entity_group": "DATE",
                           'word': item.group(), 'start': start, 'end': end})
        return result

    @staticmethod
    def get_phones(input_text):
        """
        Получение телефонов из текста
        :param input_text: текст
        :return: список словарей с телефонами и границами
        """
        pattern = r'(?:\+?7|8) ?\(?\d{3}\)? ?\d{3}[ -]?\d{2}[ -]?\d{2}'
        result = []
        for item in re.finditer(pattern, input_text):
            start, end = item.span()
            result.append({"entity_group": "TELEPHONE",
                           'word': item.group(), 'start': start, 'end': end})
        return result

    def get_techs(self, input_text):
        """
        Получение TECH из текста (чтобы верно искать его регуляркой,
        нужно сначала найти сущности 'NUM', 'ORG')
        :param input_text: текст
        :return: список словарей с адресом и границами
        """
        # Начинается с заглавной латинской буквы, затем 3 и более латинских букв
        # или 3 заглавных латинских буквы, за которыми не пробел и не буква
        add_ptn = ('APL', 'BGI', 'DQY', 'EYR', 'FME', 'GVR', 'HBC', 'HGT', 'JHX', 'KTD',
                   'LRS', 'LYD', 'NXI', 'OTX', 'OZP', 'PLQ', 'QBH', 'QQA', 'RQN', 'SFB',
                   'SKC', 'TLP', 'UMC', 'WYK', 'XQM', 'YGL', 'ZLO', 'ZMV')
        add_ptn = r'|' + r'|'.join(add_ptn)
        pattern = r'[A-Z](?:[a-zA-Z\d]){3,}|\b[A-ZА-ЯЁ]{3}(?=[^\s\w])' + add_ptn
        entity_groups = ['NUM', 'NAME_ORG']
        result = self.find_pattern_text("TECH", pattern, input_text, entity_groups)
        return result

    def get_acronym(self, input_text):
        """
        Получение ACRONYM из текста (чтобы верно искать его регуляркой,
        нужно сначала найти сущности 'NUM', 'ORG', 'TECH')
        :param input_text: текст
        :return: список словарей с адресом и границами
        """
        # 3 заглавных латинских буквы или 3 русских буквы
        # pattern = r'\b[A-ZА-ЯЁ]{3}\b'  # вторая попытка
        pattern = r'\b[A-Z]{3}\b'  # первая попытка
        entity_groups = ['NUM', 'NAME_ORG', 'TECH']
        result = self.find_pattern_text("ACRONYM", pattern, input_text, entity_groups)
        return result

    def get_all_entity_groups(self, input_text):
        self.init_entity_groups()
        entities = []
        find_ent_func = {'DATE': self.get_dates,
                         'LINK': self.get_urls,
                         'MAIL': self.get_emails,
                         'TELEPHONE': self.get_phones,
                         'PERCENT': self.get_percents,
                         'NAME_ORG': self.get_persons_orgs,
                         'NUM': self.get_nums,
                         'TECH': self.get_techs,
                         'ACRONYM': self.get_acronym,
                         }

        for entity, entity_function in find_ent_func.items():
            # поиск информации в тексте
            results = entity_function(input_text)
            # print(results)
            entities.extend(results)
            setattr(self, self.get_name_group(entity), results.copy())

        entities = sorted(entities, key=lambda x: x.get('start', 0))
        return entities


class DataReplace(DataABC):
    """ Класс для маскирования сущностей"""

    def __init__(self):
        super().__init__()

        self.fake = Faker("ru_RU")
        self.fake_en = Faker("en_US")
        self.max_faker_iterations = 10 ** 4
        self.temp_word = '9.3'
        self.temp_float = 9.3

        # Инициализация словаря для замены сущностей
        self.replacement = self.init_replacement()

    def init_replacement(self):
        """
        Инициализация словаря для замены сущностей
        :return: Очищенный словарь
        """
        self.replacement = {key: dict() for key in self.entity_groups}
        return self.replacement

    @staticmethod
    def determine_gender(full_name):
        """
        Определение пола по отчеству и фамилии
        :param full_name: полное ФИО
        :return: пол: 0-Ж, 1-М
        """
        parts = full_name.split()[-3:]
        if len(parts) < 3:
            return 1  # Если не состоит из трех частей

        # Определение по отчеству
        if any(part.endswith('ич') for part in parts[1:]):
            return 1
        elif any(part.endswith('на') for part in parts[1:]):
            return 0

        # Определение по фамилии (дополнительно)
        if any(part.endswith('а') or part.endswith('я') for part in (parts[0], parts[-1])):
            return 0

        return 1

    def generate_int_float(self):
        """
        Генерация процентов (целочисленных или вещественных)
        :return: строка с процентами
        """
        #  если исходные проценты были целым числом -> генерим целое число
        if len(self.temp_word) < 4 and '.' not in self.temp_word:
            result = random.randint(1, 10 ** len(self.temp_word) - 1)
        else:
            result = round(random.uniform(1 + 10 * (self.temp_float >= 10),
                                          2 * 10 ** (1 + (self.temp_float >= 10)) - 1),
                           2 - (10 <= self.temp_float < 100 and len(self.temp_word) == 4))
        return f'{result}%'

    def make_fake_entity(self, entity_group, word):
        """
        Получение фейковых данных для каждой сущности и сохранение их в словарь,
        чтобы не генерить новое значение для одинаковых значений сущностей
        :param entity_group: сущность
        :param word: исходное значение
        :return: фейковое значение
        """

        func = self.fake.name

        if entity_group in ('NAME', 'NAME_EMPLOYEE'):
            func = (self.fake.name_female, self.fake.name_male)[self.determine_gender(word)]

        elif entity_group == 'TELEPHONE':
            func = self.fake.phone_number

        elif entity_group == 'PERCENT':
            self.temp_word = word.replace(',', '.').strip('%')
            try:
                self.temp_float = float(self.temp_word)
            except ValueError:
                self.temp_float = 10 ** (len(self.temp_word) - 2) - 1
            func = self.generate_int_float

        elif entity_group == 'MAIL':
            func = self.fake.ascii_email

        elif entity_group == 'LINK':
            func = self.fake.url

        elif entity_group == 'ORG':
            func = self.fake.company

        elif entity_group == 'ACRONYM':
            func = lambda: self.fake.swift()[:len(word)]

        elif entity_group == 'TECH':
            func = self.fake_en.words

        # достаем уже подмененное значение
        entity_word = self.replacement[entity_group].get(word)

        # нет такого? генерим его и сохраняем в словарь
        if entity_word is None:
            num_iter = 1

            # даты и номера обрабатываются по особому
            if entity_group == 'DATE':
                entity_word = self.fake.date_this_century().strftime('%d.%m.%Y')
            elif entity_group == 'NUM':
                entity_word = self.fake.checking_account() * 2
                if not word.isnumeric():
                    entity_word = self.fake.bban() + entity_word
                entity_word = entity_word[:len(word)]

            #  для этих сущностей свой алгоритм обработки
            if entity_group in ('DATE', 'NUM'):
                num_iter = self.max_faker_iterations + 1

            max_word_len = min_diff = 0
            max_word = ''
            while num_iter < self.max_faker_iterations:
                if entity_group == 'TECH':
                    if len(word) == 3 and word.isupper():
                        entity_word = self.fake.swift()[:len(word)]
                    else:
                        nb = len(word) // 7 + 1
                        entity_word = ''.join(map(str.title, self.fake_en.words(nb=nb)))
                else:
                    entity_word = func()

                if len(entity_word) == len(word):
                    break

                num_iter += 1

                # костыль для почты
                if max_word_len < len(entity_word) < len(word):
                    if len(word) - len(entity_word) < min_diff:
                        min_diff = len(word) - len(entity_word)
                        max_word_len = len(entity_word)
                        max_word = entity_word

            if num_iter == self.max_faker_iterations:
                print(f'Достигнуто МАХ число итераций для {entity_group} подмена: {word}')
                print(max_word, max_word_len, len(word))
                # костыль для почты
                if entity_group == 'MAIL' and len(max_word) < len(word):
                    print(max_word)
                    mail = max_word.rsplit('.', 1)
                    temp = word.split('@')[1].rsplit('.', 1)
                    mail.insert(1, temp[0][:len(word) - len(max_word) - 1])
                    entity_word = '.'.join(mail)

            self.replacement[entity_group][word] = entity_word

        return entity_word


class DataProcessBert(DataProcess):
    """ Класс для поиска сущностей использует Bert"""

    def __init__(self, load_model=True):
        """
        Инициализация экземпляра класса
        :param load_model: Загружать в память NER модель
        """
        super().__init__(load_model=load_model)

        self.char2id, self.id2char = load_char_vocab()
        # модель, какую будет использовать Bert
        self.bert_сlf = None
        self.bert_model = "./results/checkpoint-last"
        if load_model:
            # Load the fine-tuned model
            self.bert_сlf = AutoModelForTokenClassification.from_pretrained(self.bert_model)
            self.bert_сlf.to(self.device)

    def init_ner_classifier(self, strategy=None):
        """
        Создаем экземпляр NER классификатора
        :param strategy: стратегия для объединения токенов
        :return: экземпляр NER классификатора
        """
        if strategy is None:
            strategy = self.strategy
        else:
            self.strategy = strategy

        self.ner_classifier = pipeline(
            # "ner",
            "token-classification",
            model=self.model_name,
            tokenizer=self.model_name,
            aggregation_strategy=strategy,
            device=self.device
        )
        self.bert_сlf = self.ner_classifier
        return self.ner_classifier

    def get_persons_orgs(self, input_text):
        if self.ner_classifier is None or self.bert_сlf is None:
            print('Модели для NER недоступны!!!')
            return []

        # получим персон обычным способом
        pers_orgs = super().get_persons_orgs(input_text)
        # удалим организации
        pers_orgs = [item for item in pers_orgs if item.get('entity_group') != 'ORG']

        text = input_text.replace('−', '-')[:max_len]
        tokens = list(text)
        input_ids = [self.char2id[char] for char in tokens]
        attention_mask = [1] * len(input_ids)

        # Convert to tensors (without padding)
        inputs = {
            'input_ids': torch.tensor([input_ids], device=self.device),
            'attention_mask': torch.tensor([attention_mask], device=self.device)
        }

        # Perform inference
        self.bert_сlf.eval()
        with torch.no_grad():
            outputs = self.bert_сlf(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)

        # Convert predictions to labels
        predicted_labels = [id2tag[p.item()] for p in predictions[0][:len(text)] if
                            p.item() != -100]

        # Apply post-processing to correct I- labels without preceding B- labels
        corrected_labels = post_process_predictions(predicted_labels)

        entities = predictions_to_entities(text, corrected_labels)

        # print(entities)

        # predictions_output.append({"text": text, "entities": entities})
        # возьмем из предсказаний только организации
        organizations = [item for item in entities if item.get('entity_group') == 'ORG'
                         and len(item.get('word', '')) > 2]

        pers_orgs.extend(organizations)

        return pers_orgs


class DataProcessBertTuned(DataProcessBert):
    """ Класс для поиска сущностей использует Bert"""

    def __init__(self, load_model=True):
        """
        Инициализация экземпляра класса
        :param load_model: Загружать в память NER модель
        """
        DataProcess.__init__(self, load_model=load_model)

        self.char2id, self.id2char = load_char_vocab()

        # модель, какую будет использовать Bert
        self.bert_сlf = None
        self.bert_model = "./models/sberbank-ai--ruBert-base"
        if load_model:
            # Load the fine-tuned model
            self.bert_сlf = AutoModelForTokenClassification.from_pretrained(self.bert_model)
            self.bert_сlf.to(self.device)


if __name__ == "__main__":
    pass

    # Открываем файл train.json
    json_file = 'train.json'
    with open(json_file, encoding='utf-8') as file:
        data = json.load(file)
        print(f'Количество записей в train.json: {len(data)}')

    TECH = set()
    ACRONYM = set()
    # Перебираем каждую запись в файле
    for record in data:
        for entity in record['entities']:
            if entity['entity_group'] == 'TECH':
                if len(entity['word']) == 3:
                    TECH.add(entity['word'])
            if entity['entity_group'] == 'ACRONYM':
                ACRONYM.add(entity['word'])
    print(sorted(TECH))
    print(sorted(ACRONYM))
    print(TECH & ACRONYM)
