import river

metric = river.metrics.Accuracy()
optimizer = river.optim.SGD(lr=0.01)
model = river.linear_model.LogisticRegression(optimizer)
dataset = river.stream.iter_arff("inp.arff", target='class')

for i, (xi, yi) in enumerate(dataset):
    y_pred = 1 if model.predict_proba_one(xi)[True] >= 0.5 else 0
    y_num = 1 if yi == 'groupA' else 0
    metric.update(y_num, y_pred)
    model.learn_one(xi, y_num)
    if (i + 1) % 10000 == 0:
        print(f'Instance {i + 1}: {metric}')

