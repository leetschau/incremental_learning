from flask import Flask

app = Flask(__name__)

@app.route('/', methods=['GET'])
def predict():
    return "<p>Hello, gunicorn!</p>"

@app.route('/', methods=['POST'])
def learn():
    payload = flask.request.json
    river_model = load_model()
    river_model.learn_one(payload['features'], payload['target'])
    return {}, 201

