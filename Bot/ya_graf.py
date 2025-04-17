from typing import List, TypedDict, Annotated, Optional
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
import re
import json
import pandas as pd
from pydantic import BaseModel

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.llms import YandexGPT

load_dotenv(find_dotenv())
YANDEX_TOKEN = os.environ.get("YANDEX_TOKEN")

model = YandexGPT(iam_token=YANDEX_TOKEN,
                   folder_id="b1g08itthrid4mko59gr",
                   model_name='yandexgpt',
                   model_version="rc",
                   temperature=0.1,
                   verbose=True)

class FormatCheck(BaseModel):
    date: str
    department: str
    operation: str
    culture: str
    areaPerDay: float
    totalArea: float
    yieldPerDay: float
    totalYield: float


# json_parser = JsonOutputParser()
operations = pd.read_csv('data/справка-подразделений.csv').to_json(orient="records", force_ascii=False)

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

department = pd.read_csv('data/справка-операции.csv').to_json(orient="records", force_ascii=False)

class State(TypedDict):
    input_message: Optional[str]  # Contains subject, sender, body, etc.
    extract_data: Optional[str]
    messages: List[Dict[str, Any]] 
    normalize_message_: Optional[str] 
    json_entities: Optional[List[Dict[str, Any]]] 

def normalize_message_node(state:State) -> State:
    input_message = state['input_message']
    normalize_prompt = f"""
    Вы - специализированный ассистент по обработке агрономических данных. Ваша задача - нормализовать сокращенные сообщения, преобразуя сокращенные термины в их полные формы с использованием предоставленных справочных данных.
    Правила нормализации:
    • Культуры: Замените сокращенные названия сельскохозяйственных культур на их полные наименования согласно предоставленному JSON-словарю {crop_сulture_json}. Например:
    "сах св" → "Свекла сахарная"
    "оз ячмень" → "Ячмень озимый"
    "кук зерно" → "Кукуруза товарная"
    • Операции: Замените сокращенные названия сельскохозяйственных операций на их полные наименования согласно предоставленному списку {operations}. Например:
    • Структурные элементы: С помощью json файла {department} тебе нужно поменять написанный структурный элемент на его ПОДРАЗДЕЛЕНИЕ:
    "Отд_X" → "АОР"; 
    "По Пу" → "АОР"

    Формат данных: Сохраните исходную структуру сообщения, но с нормализованными терминами. Числовые данные (например, "77/518") оставьте без изменений.
    • Контекст: Учитывайте, что данные представляют собой отчеты о сельскохозяйственных работах с указанием:
    Типа операции
    Культуры
    Подразделения (отделения или производственного участка)
    Числовых показателей (вероятно, площадь/объем работ)
    При нормализации сохраняйте исходное форматирование и структуру данных, изменяя только сокращенные термины на их полные формы согласно предоставленным словарям.
    """
    normalize_message = model.invoke([
    SystemMessage(content=normalize_prompt),
    HumanMessage(content=input_message)
    ])
    print(normalize_message)
    state["normalize_message_"] = normalize_message
    return state

def extract_entities(state:State) -> State:
    normalize_message = state["normalize_message_"]
    extract_prompt = """
    Ты — сервис извлечения структурированных данных.
    На входе ты получаешь неструктурированное русскоязычное сообщение о полевых работах хозяйства.
    На выходе ты строго возвращаешь только JSON (без пояснений, комментариев и форматирования Markdown).
    JSON состоит из массива объектов; один объект описывает одну уникальную комбинацию
    «Подразделение + Операция + Культура».
    Объект ДОЛЖЕН содержать ВСЕ перечисленные ниже ключи — даже если значение null.
    * date — дата формата ДД.ММ.ГГГГ. • Определи её по любому из паттернов ДД.ММ, ДД‑ММ, ДД/ММ, ДД.ММ.ГГ, ДД.ММ.ГГГГ. Если даты нет, то поставь null .
    * department — название подразделения(например АОР, ТСК, АО Кропоткинское, Восход, Колхоз Прогресс, Мир, СП Коломейцево).
    * operation — название агрономической операции (существительное; например «Пахота», «Предпосевная культивация», «Дискование»).:
    * culture - название культуры (существительное; например «Пшеница озимая товарная», «Пшеница озимая семенная», «Соя товарная»).
    * areaPerDay — площадь (га), выполненная за отчётный день. Число перед косой чертой «/» в записи «XX/YY». Если такого числа нет — null.
    * totalArea — суммарная площадь (га) «с начала операции» — число после «/» в той же записи. Если нет — null.
    * yieldPerDay — валовой сбор за день (центнеры, ц). Определи по числам, за которыми сразу следует «ц»/«центнер» и нет символа «/». Если в тексте не указано — null.
    * totalYield — валовой сбор с начала операции (центнеры, ц). Чаще всего бывает после комбинации «Вал с начала» или второй компонент записи «XX/YY ц». Если не найдено — null.
    Алгоритм разбора чисел
    • Игнорируй пробельные символы и символы «→», «-», «•».
    • Для чисел допускаются запятые как разделитель тысяч «1 234» и точки как десятичная часть «12.5».
    • Преобразуй итоговые числа к целым, если дробной части нет, иначе к float.
    Обработка множественных строк вида «Отд 12 26/221»
    • Если в одном абзаце несколько отделов (несколько «Отд X …»), то формируй JSON ТОЛЬКО для самого первого.
    Если какой‑то атрибут отсутствует — ставь литерал null (без кавычек).
    Строго соблюдай порядок ключей в каждом объекте как указан выше.
    Пример правильного вывода (без форматирования):
    [ message = "
    27.10.день
    Предп культ под оз пш
    По Пу 215/1015"
  {
    "date": "27.10.2024",
    "department": "АОР",
    "operation": "Предпосевная культивация",
    "culture": "Пшеница озимая товарная",
    "areaPerDay": 215,
    "totalArea": 1015,
    "yieldPerDay": null,
    "totalYield": null
  }, 
  message = "
    20.11 Уборка сах св
    Отд 12 16/16
    Вал 473920
    Урож 296,2
    Диг - 19,19
    Оз - 5,33
    "
      {
    "date": "20.11.2024",
    "department": "АОР",
    "operation": "Уборка",
    "culture": "Свекла сахарная",
    "areaPerDay": 16,
    "totalArea": 16,
    "yieldPerDay": 473920,
    "totalYield": null
  },]
    """
    parser = JsonOutputParser(pydantic_object=FormatCheck)
    chain = model | parser
    entities = chain.invoke([
    SystemMessage(content=extract_prompt),
    HumanMessage(content=normalize_message)
    ])
    print(entities)
    state["json_entities"] = entities
    return state



# Create the graph
ogr_graph = StateGraph(State)
# Add nodes
ogr_graph.add_node("normalize_message", normalize_message_node)
ogr_graph.add_node("extract_entities", extract_entities)
extract_entities
# Start the edges
ogr_graph.add_edge(START, "normalize_message")
ogr_graph.add_edge("normalize_message", "extract_entities")

ogr_graph.add_edge("extract_entities", END)

# Compile the graph
compiled_graph = ogr_graph.compile()

message = """
Предп культ под оз пш
По Пу 91/1403
Отд 11 45/373
Отд 12 46/363"
"""

compiled_graph.invoke({
    "input_message": message,
    "messages": [],
    "normalize_message": str,
    "json_entities": {}
})
