import asyncio

from aiogram import Dispatcher, Bot, types, executor
from DeepSeek import deepseek_answer, answ2dict
from excel_logger import toExcel
from prompts import system_prompt_check_message, system_prompt_get_json
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
TOKEN = os.environ.get("TG_TOKEN")

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(content_types=['text'])
async def test(message: types.Message):
    essences = deepseek_answer(system_prompt_check_message, message.text)
    print('check_flag', essences)
    ### что-то типо проверки текста на наличие сущностей, чтобы обрабатывать или не обрабатывать
    if 'True' in essences:
        answ_flag = False
        await bot.send_message(message.from_user.id, 'Информация обработана')
        while not(answ_flag):
            # essences = deepseek_answer(system_prompt_get_json, message.text)
            essences = essences[essences.find('{'): essences.rfind('}') + 1]
            json_data = answ2dict(essences)

            if json_data != 'repeat prompt':
                answ_flag = True

        toExcel(json_data)

    elif 'non-info' in essences:
        await bot.send_message(message.from_user.id, 'Недостаточно информации')
        return


if __name__ == '__main__':
    executor.start_polling(dp)