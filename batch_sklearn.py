from scipy.io import arff
import pandas as pd

def eval_acc(inp: pd.DataFram) -> float:
    pass


inpfile = 'abrupt_drift.arff'
groups = 10

accs = {}
inp = pd.DataFrame(arff.loadarff(inpfile)[0])
for i in range(groups):
    tail = inp.shape[0] * (i + 1) // groups
    accs[i + 1] = eval_acc(inp.head(tail))

print(accs)
