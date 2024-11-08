import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path, sep=';', low_memory=False, na_values=['nan', '?'])
    df['dt'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df = df.set_index('dt').drop(['Date', 'Time'], axis=1)
    return df

