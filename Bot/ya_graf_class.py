from typing import List, TypedDict, Annotated, Optional
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
import json
from pathlib import Path
import pandas as pd
from pydantic import BaseModel

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.llms import YandexGPT

load_dotenv(find_dotenv())

file_path = Path("data/crop_culture.json")

# откроем файл и передадим его объект в json.load
with file_path.open("r", encoding="utf-8") as f:
    crops = json.load(f)

class FormatCheck(BaseModel):
    date: str
    department: str
    operation: str
    culture: str
    areaPerDay: float
    totalArea: float
    yieldPerDay: float
    totalYield: float

class State(TypedDict):
    input_message: Optional[str] 
    extract_data: Optional[str]
    messages: List[Dict[str, Any]]
    normalize_message_: Optional[str]
    json_entities: Optional[List[Dict[str, Any]]]

class YaGraph:
    def __init__(self):
        self.model = self.get_model()
        self.parser = JsonOutputParser(pydantic_object=FormatCheck)
        self.operations = pd.read_csv('data/справка-подразделений.csv').to_json(orient="records", force_ascii=False)
        self.crop_сulture_json = crops
        self.department = pd.read_csv('data/справка-операции.csv').to_json(orient="records", force_ascii=False)

    def get_model(self):
        YANDEX_TOKEN = os.environ.get("YANDEX_TOKEN")

        model = YandexGPT(iam_token=YANDEX_TOKEN,
                          folder_id="b1g08itthrid4mko59gr",
                          model_name='yandexgpt',
                          model_version="rc",
                          temperature=0.1)

        return model

    def normalize_message_node(self, state:State) -> State:
        input_message = state['input_message']
        normalize_prompt = f"""
        Вы - специализированный ассистент по обработке агрономических данных. Ваша задача - нормализовать сокращенные сообщения, преобразуя сокращенные термины в их полные формы с использованием предоставленных справочных данных.
        Правила нормализации:
        • Культуры: Замените сокращенные названия сельскохозяйственных культур на их полные наименования согласно предоставленному JSON-словарю {self.crop_сulture_json}. Например:
        "сах св" → "Свекла сахарная"
        "оз ячмень" → "Ячмень озимый"
        "кук зерно" → "Кукуруза товарная"
        • Операции: Замените сокращенные названия сельскохозяйственных операций на их полные наименования согласно предоставленному списку {self.operations}. Например:
        • Структурные элементы: С помощью json файла {self.department} тебе нужно поменять написанный структурный элемент на его ПОДРАЗДЕЛЕНИЕ:
        "Отд_X" → "АОР". ВАЖНО: Для отделений всегда будет "АОР" без чисел; Примеры:"Отд 12" → "АОР", "Отд 17" → "АОР".
        "По Пу" → "АОР по производственному участку".
    
        Формат данных: Сохраните исходную структуру сообщения, но с нормализованными терминами. Числовые данные (например, "77/518") оставьте без изменений.
        • Контекст: Учитывайте, что данные представляют собой отчеты о сельскохозяйственных работах с указанием:
        Типа операции
        Культуры
        Подразделения (отделения или производственного участка)
        Числовых показателей (вероятно, площадь/объем работ)
        При нормализации сохраняйте исходное форматирование и структуру данных, изменяя только сокращенные термины на их полные формы согласно предоставленным словарям.
        """
        normalize_message = self.model.invoke([
        SystemMessage(content=normalize_prompt),
        HumanMessage(content=input_message)
        ])
        state["normalize_message_"] = normalize_message
        return state

    def extract_entities(self, state:State) -> State:
        normalize_message = state["normalize_message_"]
        extract_prompt = """
        Ты — сервис извлечения структурированных данных.
        На входе ты получаешь неструктурированное русскоязычное сообщение о полевых работах хозяйства.
        На выходе ты строго возвращаешь только JSON (без пояснений, комментариев и форматирования Markdown).
        JSON состоит из массива объектов; один объект описывает одну уникальную комбинацию
        «Подразделение + Операция + Культура».
        Объект ДОЛЖЕН содержать ВСЕ перечисленные ниже ключи — даже если значение null.
        * date— дата формата ДД.ММ.ГГГГ. • Определи её по любому из паттернов ДД.ММ, ДД‑ММ, ДД/ММ, ДД.ММ.ГГ, ДД.ММ.ГГГГ. Если даты нет, то поставь null .
        * department— название подразделения(например АОР, ТСК, АО Кропоткинское, Восход, Колхоз Прогресс, Мир, СП Коломейцево).
        * operation— название агрономической операции (существительное; например «Пахота», «Предпосевная культивация», «Дискование»).:
        * culture - название культуры (существительное; например «Пшеница озимая товарная», «Пшеница озимая семенная», «Соя товарная»).
        * areaPerDay— площадь (га), выполненная за отчётный день. Число перед косой чертой «/» в записи «XX/YY». Если такого числа нет — null.
        * totalArea— суммарная площадь (га) «с начала операции» — число после «/» в той же записи. Если нет — null.
        * yieldPerDay— валовой сбор за день (центнеры, ц). Определи по числам, за которыми сразу следует «ц»/«центнер» и нет символа «/». Если в тексте не указано — null.
        * totalYield— валовой сбор с начала операции (центнеры, ц). Чаще всего бывает после комбинации «Вал с начала» или второй компонент записи «XX/YY ц». Если не найдено — null.
        Алгоритм разбора чисел
        • Игнорируй пробельные символы и символы «→», «-», «•», «"».
        • Для чисел допускаются запятые как разделитель тысяч «1 234» и точки как десятичная часть «12.5».
        • Преобразуй итоговые числа к целым, если дробной части нет, иначе к float.
        
        УЧЕТ ПОРЯДКА: 
        • Если в абзаце содержится "АОР по производственному участку", то формируй JSON ТОЛЬКО для данного образца.
        • Если в одном абзаце несколько отделов (несколько «АОР …») и нет "АОР по производственному участку", то формируй JSON ТОЛЬКО для первого найденного образца.
        Если какой‑то атрибут отсутствует — ставь литерал null (без кавычек).
        Строго соблюдай порядок ключей в каждом объекте как указан выше.
        ВАЖНО: Игнорируй все сообщения которые не относятся к:
        {date, department, operation, culture, areaPerDay, totalArea, yieldPerDay, totalYield}
        Пример правильного вывода (без форматирования):
        [ message = "
        27.10.день
        Предп культ под оз пш
        АОР по производственному участку 215/1015
        АОР 18/32"

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
        АОР 16/16
        АОР 18/32
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
        chain = self.model | self.parser
        entities = chain.invoke([
        SystemMessage(content=extract_prompt),
        HumanMessage(content=normalize_message)
        ])
        state["json_entities"] = entities
        return state



    def get_graph(self):
        # Create the graph
        ogr_graph = StateGraph(State)
        # Add nodes
        ogr_graph.add_node("normalize_message", self.normalize_message_node)
        ogr_graph.add_node("extract_entities", self.extract_entities)
        # Start the edges
        ogr_graph.add_edge(START, "normalize_message")
        ogr_graph.add_edge("normalize_message", "extract_entities")

        ogr_graph.add_edge("extract_entities", END)

        # Compile the graph
        compiled_graph = ogr_graph.compile()

        return compiled_graph


# message = """
# "Пахота зяби под Многолетние травы
# АОР 13/540
# АОР 13/273
# """

# yaGraph = YaGraph()
# compiled_graph = yaGraph.get_graph()
# #
# response = compiled_graph.invoke({
#     "input_message": message,
#     "messages": [],
#     "normalize_message": str,
#     "json_entities": {}
# })

# print(response['json_entities'])