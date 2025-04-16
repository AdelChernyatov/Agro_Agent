import pandas as pd
import os

from datetime import datetime

def toExcel(data, excel_file='agrologs.xlsx'):
    records = {
        'Дата': datetime.now().strftime('%H:%M'),
        'Подразделение': data['Подразделение'],
        'Операция': data['Операция'],
        'Культура': data['Культура'],
        'Гектар за день': data['Гектар за день'],
        'Гектар с начала операции': data['Гектар с начала операции']
    }

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
                'Гектар с начала операции'
            ]
        )

    df = pd.concat([df, pd.DataFrame([records])], ignore_index=True)

    df.to_excel(excel_file, index=False)