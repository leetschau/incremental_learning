from flask import Flask, request
import shelve
from datetime import datetime

from river import preprocessing
from river import linear_model
from river import optim
import pandas as pd
from sklearn import metrics

app = Flask(__name__)

scaler = preprocessing.StandardScaler()
optimizer = optim.SGD(lr=0.01)
log_reg = linear_model.LogisticRegression(optimizer)

y_true = []
y_pred = []

@app.route('/predict', methods=['POST'])
def predict():
    '''predict by one observation

    payload: a dict contains all features
    '''
    X = request.json
    X_scaled = scaler.learn_one(X).transform_one(X)
    yi_pred = log_reg.predict_proba_one(X_scaled)
    return {'res': yi_pred[True]}, 201


@app.route('/learn', methods=['POST'])
def learn():
    '''learn by one observation

    payload: a dict contains all features as X and 'target' as y
    '''
    X = request.json
    yi = X.pop('target', None)

    X_scaled = scaler.learn_one(X).transform_one(X)
    yi_pred = log_reg.predict_proba_one(X_scaled)

    log_reg.learn_one(X_scaled, yi)

    y_true.append(yi)
    y_pred.append(yi_pred[True])
    return {'status': "ok"}, 201


@app.route('/auc', methods=['GET'])
def get_auc():
    return {'res': metrics.roc_auc_score(y_true, y_pred)}, 201


@app.route('/save', methods=['POST'])
def save_model():
    with shelve.open('incr-result') as db:
        now = datetime.now()
        db[f'model-{now.strftime("%y-%m-%d %H:%M:%S")}'] = {
                'model': log_reg,
                'AUC': metrics.roc_auc_score(y_true, y_pred),
                'observation number': len(y_true),
                'time': now }
    return {'status': "ok"}, 201

