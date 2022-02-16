import pandas as pd
import requests
import time
import random
from tqdm import tqdm

DATA_FRAC = 0.8

# load data
inp = pd.read_pickle('input.pkl').sample(frac=DATA_FRAC)
data = inp.to_dict(orient='records')

# training model
for rec in tqdm(data, desc='Learned obs', unit='obs', colour='blue'):
    time.sleep(random.random())
    r = requests.post('http://localhost:5123/learn', json=rec)
    assert r.json()['status'] == 'ok'

# get AUC
r = requests.get('http://localhost:5123/auc')
print(f'ROC AUC: {r.json()["res"]:.3f}')

# predict with a malignant sample
msam = [ob for ob in data if ob['target'] == 1][0]
msam['worst area'] = msam['worst area'] * 1.1
msam['mean perimeter'] = msam['mean perimeter'] * 0.9

r = requests.post('http://localhost:5123/predict', json=msam)
print(f'Prediction of a pseudo-malignant (target = 1) sample: {r.json()["res"]}')

# predict with a benign sample
bsam = [ob for ob in data if ob['target'] == 0][0]
bsam['worst area'] = bsam['worst area'] * 1.1
bsam['mean perimeter'] = bsam['mean perimeter'] * 0.9

r = requests.post('http://localhost:5123/predict', json=bsam)
print(f'Prediction of a pseudo-benign (target = 0) sample: {r.json()["res"]}')

# save model
r = requests.post('http://localhost:5123/save')
print(f'Save model {r.json()["status"]}')
