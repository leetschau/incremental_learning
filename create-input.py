import pandas as pd
from sklearn import datasets

inp = datasets.load_breast_cancer()
df = pd.DataFrame(data=inp.data, columns=inp.feature_names)
df['target'] = pd.Series(inp.target)
df.to_pickle("input.pkl")
