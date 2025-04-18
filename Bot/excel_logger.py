import pandas as pd
import os
import re
from datetime import datetime

def extract_or_default_date(date) -> datetime:
    date_format = "%d.%m.%Y"
    if date:
        # Ищем дату в сообщении: допускаем . / - в качестве разделителей
        match = re.search(r"\b(\d{1,2})[.\-\/](\d{1,2})(?:[.\-\/](\d{2,4}))?\b", date)
        if match:
            day, month, year = match.groups()
            day = int(day)
            month = int(month)
            if year:
                year = int(year)
                if year < 100:  # например 24 → 2024
                    year += 2000

            try:
                date = datetime(year, month, day)
                return date.strftime(date_format)
            except:
                received_date = datetime.now()
    else:
        received_date = datetime.now()
    # Если дата не найдена или неверная — используем дату получения
    return received_date.strftime(date_format)


def format_data(data):
    data['Дата'] = extract_or_default_date(data['Дата'])
    if data['Вал за день, ц']:
        data['Вал за день, ц'] = data['Вал за день, ц'] / 100 # поменять на запятую
    if data['Вал с начала, ц']:
        data['Вал с начала, ц'] = data['Вал с начала, ц'] / 100 # поменять на запятую
    return data

def toExcel(data, excel_file='excel_test/test_0.xlsx'):
    records = {
        'Дата': data['date'],
        'Подразделение': data['department'],
        'Операция': data['operation'],
        'Культура': data['culture'],
        'Гектар за день': data['areaPerDay'],# делить на 100
        'Гектар с начала операции': data['totalArea'],# делить на 100
        'Вал за день, ц': data['yieldPerDay'], # делить на 100
        'Вал с начала, ц': data['totalYield'] # делить на 100 
    }
    records = format_data(records)
    if os.path.exists(excel_file):
        df = pd.read_excel(excel_file)
    else:
        df = pd.DataFrame(
            columns=[
                'Дата',
                'Подразделение',
                'Операция',
                'Культура',
                'Гектар за день',
                'Гектар с начала операции',
                'Вал за день, ц',
                'Вал с начала, ц'
            ]
        )

    df = pd.concat([df, pd.DataFrame([records])], ignore_index=True)

    df.to_excel(excel_file, index=False)