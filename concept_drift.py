from scipy.io import arff
import pandas as pd
import river

from river import preprocessing
scaler = preprocessing.StandardScaler()

inp = arff.loadarff('concept_drift.arff')
df = pd.DataFrame(inp[0])
X = df.drop(columns='class').to_dict('records')
labels = df['class'].transform(lambda x: 1 if x == b'class1' else 0)

metric = river.metrics.Accuracy()

model = river.linear_model.LogisticRegression()

i = 0
for xi, yi in zip(X, labels):
    # print(f'x: {xi}, y: {yi}')
    # xi_scaled = scaler.learn_one(xi).transform_one(xi)
    # print(f'x: {xi_scaled}, y: {yi}')
    y_pred = model.predict_proba_one(xi)
    # print(f'y_pred: {y_pred}')
    metric.update(yi, y_pred[True])
    model.learn_one(xi, yi)
    if i % 10000 == 0:
        print(f'index: {i}, acc: {metric}')
    i = i + 1

# model = river.naive_bayes.MultinomialNB()
# for i in range(len(y)):
    # y_pred = model.predict_proba_one(X_stream[i])
    # metric.update(y[i], y_pred)
    # model.learn_one(X_stream[i], y[i])

    # if i % 1000 == 0:
        # print(f'index: {i}, acc: {metric}')
