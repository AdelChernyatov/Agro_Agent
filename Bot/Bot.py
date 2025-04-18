import asyncio

from aiogram import Dispatcher, Bot, types, executor
from excel_logger import toExcel
from ya_graf_class import YaGraph
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
TOKEN = os.environ.get("TG_TOKEN")

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(content_types=['text'])
async def test(message: types.Message):
    yaGraph = YaGraph()
    compiled_graph = yaGraph.get_graph()
    response = compiled_graph.invoke({
        "input_message": message.text,
        "messages": [],
        "normalize_message": str,
        "json_entities": {}
    })
    json_entities = response['json_entities'][0]   
    await bot.send_message(message.from_user.id, 'Информация обработана')
    toExcel(json_entities)

if __name__ == '__main__':
    executor.start_polling(dp)