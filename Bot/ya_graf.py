from typing import List, TypedDict, Annotated, Optional
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
import re
import json
import pandas as pd

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.llms import YandexGPT


model = YandexGPT(iam_token="",
                   folder_id="b1g08itthrid4mko59gr",
                   model_name='yandexgpt',
                   model_version="rc",
                   temperature=0.1,
                   verbose=True)
# json_parser = JsonOutputParser()

operations = """
    1-я междурядная культивация
    2-я междурядная культивация
    Боронование довсходовое
    Внесение минеральных удобрений
    Выравнивание зяби
    2-е Выравнивание зяби
    Гербицидная обработка
    1 Гербицидная обработка
    2 Гербицидная обработка
    3 Гербицидная обработка
    4 Гербицидная обработка
    Дискование
    Дискование 2-е
    Инсектицидная обработка
    Культивация
    Пахота
    Подкормка
    Предпосевная культивация
    Прикатывание посевов
    Сев
    Сплошная культивация
    Уборка
    Функицидная обработка
    Чизлевание
    """

crop_сulture_json = {
    "Пшеница озимая товарная": [
        "оз пш",
        "пшеница",
        "озимые",
        "оз пшеница",
        "пш",
        "пшеница озимая"
    ],
    "Пшеница озимая семенная": [
        "пшеница озимая семенная",
        "оз пш семенная"
    ],
    "Пшеница озимая на зеленый корм": [
        "пшеница озимая на зеленый корм",
        "оз пш корм"
    ],
    "Многолетние травы текущего года": [
        "мн тр",
        "многолетние",
        "травы",
        "многолетние травы"
    ],
    "Многолетние травы прошлых лет": [
        "многолетние травы прошлых лет"
    ],
    "Многолетние злаковые травы": [
        "многолетние злаковые травы"
    ],
    "Соя товарная": [
        "сои",
        "соя",
        "под сою",
        "соя товарная"
    ],
    "Соя семенная": [
        "соя семенная"
    ],
    "Свекла сахарная": [
        "сах св",
        "свекла",
        "сах.св",
        "свёкла",
        "свекла сахарная"
    ],
    "Кукуруза товарная": [
        "кук зерно",
        "кукуруза",
        "кук",
        "под кук",
        "кукуруза товарная"
    ],
    "Кукуруза семенная": [
        "кук семян",
        "кукуруза семенная"
    ],
    "Кукуруза кормовая": [
        "кук силос",
        "кукуруза силос",
        "кук/силос",
        "кукуруза кормовая"
    ],
    "Подсолнечник товарный": [
        "подсол",
        "подсолнечник",
        "подс",
        "подсолнечник товарный"
    ],
    "Подсолнечник кондитерский": [
        "подсолнечник кондитерский"
    ],
    "Подсолнечник семенной": [
        "подсолнечник семенной"
    ],
    "Рапс озимый": [
        "оз рапс",
        "рапс",
        "под рапс",
        "рапс озимый"
    ],
    "Рапс яровой": [
        "рапс яровой"
    ],
    "Ячмень озимый": [
        "оз ячмень",
        "ячмень",
        "оз ячм",
        "ячмень озимый"
    ],
    "Ячмень озимый семенной": [
        "ячмень озимый семенной"
    ],
    "Овес": [
        "овес",
        "посев овса"
    ],
    "Горох товарный": [
        "горох",
        "горох товарный"
    ],
    "Горох на зерно": [
        "горох на зерно"
    ],
    "Люцерна": [
        "люцерна"
    ],
    "Чистый пар": [
        "чистый пар"
    ],
    "Вика+Тритикале": [
        "вика+тритикале"
    ],
    "Сорго": [
        "сорго"
    ],
    "Сорго кормовой": [
        "сорго кормовой"
    ],
    "Сорго-суданковый гибрид": [
        "сорго-суданковый гибрид"
    ],
    "Кориандр": [
        "кориандр"
    ],
    "Конопля": [
        "конопля"
    ],
    "Гуар": [
        "гуар"
    ],
    "Чумиза": [
        "чумиза"
    ],
    "Просо": [
        "просо"
    ]
} 

class State(TypedDict):
    input_message: Optional[str]  # Contains subject, sender, body, etc.
    extract_data: Optional[str]
    messages: List[Dict[str, Any]] 
    normalize_message_: Optional[str]  


def normalize_message_node(state:State) -> State:
    input_message = state['input_message']
    normalize_prompt = """
    Вы - специализированный ассистент по обработке агрономических данных. Ваша задача - нормализовать сокращенные сообщения, преобразуя сокращенные термины в их полные формы с использованием предоставленных справочных данных.
    Правила нормализации:
    • Культуры: Замените сокращенные названия сельскохозяйственных культур на их полные наименования согласно предоставленному JSON-словарю {crop_сulture_json}. Например:
    "сах св" → "сахарная свекла"
    "оз ячмень" → "озимый ячмень"
    "оз зел корм" → "озимый зеленый корм"
    • Операции: Замените сокращенные названия сельскохозяйственных операций на их полные наименования согласно предоставленному списку {operations}. Например:
    • Структурные элементы:
    "Отд" → "Отделение"
    "По Пу" → "По производственному участку"
    Формат данных: Сохраните исходную структуру сообщения, но с нормализованными терминами. Числовые данные (например, "77/518") оставьте без изменений.
    • Контекст: Учитывайте, что данные представляют собой отчеты о сельскохозяйственных работах с указанием:
    Типа операции
    Культуры
    Подразделения (отделения или производственного участка)
    Числовых показателей (вероятно, площадь/объем работ)
    При нормализации сохраняйте исходное форматирование и структуру данных, изменяя только сокращенные термины на их полные формы согласно предоставленным словарям.
    """
    normalize_message = model.invoke([
    SystemMessage(content=normalize_prompt.format(operations=operations, crop_сulture_json=crop_сulture_json)),
    HumanMessage(content=input_message)
    ])
    # normalize_message = response[0].content
    print(normalize_message)
    state["normalize_message_"] = normalize_message
    return state

# Create the graph
ogr_graph = StateGraph(State)

# Add nodes
ogr_graph.add_node("normalize_message", normalize_message_node)


# Start the edges
ogr_graph.add_edge(START, "normalize_message")
ogr_graph.add_edge("normalize_message", END)

# Compile the graph
compiled_graph = ogr_graph.compile()

message = """
"Пахота зяби под сою
По Пу 15/1382
Отд 16 15/775
"""

compiled_graph.invoke({
    "input_message": message,
    "messages": [],
    "normalize_message": str
})
