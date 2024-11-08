import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    df = df.fillna(df.mean())
    df_resample = df.resample('h').mean()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_resample)
    return scaled_data