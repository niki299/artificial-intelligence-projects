import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    data_frame = pd.DataFrame(data_scaled, columns=data.columns)

    return data_frame

def read_file(filepath):
    data = pd.read_csv(filepath)
    data.loc[data['MINIMUM_PAYMENTS'].isnull(), 'MINIMUM_PAYMENTS'] = data['MINIMUM_PAYMENTS'].median()
    data.loc[data['CREDIT_LIMIT'].isnull(), 'CREDIT_LIMIT'] = data['CREDIT_LIMIT'].median()
    data = data.drop('CUST_ID', 1)

    return data
