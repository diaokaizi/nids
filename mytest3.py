import numpy as np
import pandas as pd



datasets = {
    'NF-UNSW-NB15' : '/root/work/NIDS/data/NF-UNSW-NB15-FULL.parquet',
    # 'NF-BoT-IoT' : '/root/work/NIDS/data/NF-BoT-IoT.parquet',
    # 'NF-CSE-CIC-IDS2018' : '/root/work/NIDS/data/NF-CSE-CIC-IDS2018.parquet',
    # 'NF-ToN-IoT' : '/root/work/NIDS/data/NF-ToN-IoT.parquet',
}

features_to_remove = ['Attack', 'Label']


x_train = {}
x_val = {}
x_test = {}





# df = pd.read_csv('/root/work/NIDS/data/NF-UNSW-NB15.csv')
# print(df)

# df = pd.read_parquet('/root/work/NIDS/data/NF-UNSW-NB15.parquet')
# print(df)

roc_metrics = {
    'Accuracy': 1.233,
    'Precision': 1,
    'Recall': 1,
    'F1': 1
}

pr_metrics = {
    'Accuracy': 123,
    'Precision': 1,
    'Recall': 1,
    'F1': 1
}
from resultObj import Result
a = Result()
a.save_result(roc_metrics, pr_metrics, "123")