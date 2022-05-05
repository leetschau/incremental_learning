# Incremental ETA

ETA (encrypted traffic analysis) with incremental training based on River.

## HTTP C/S Scenario

The incremental training is build as a HTTP server.
After build the demo dataset, you can modify the dataset size by `DATA_FRAC` in data-sender.py.

1. Clone the repo;

1. Install dependencies: `pdm install`;

1. Build demo dataset: `pdm run python create-input.py`;

1. Start server: `pdm run gunicorn -b 0.0.0.0:5123 wsgi:app`;

1. Client requests: `pdm run python data-sender.py`.

Model and relative metadata are saved in a shelve database defined in server.py.

# Concept Drift Demo

Create dataset:
```
java -Xmx40g -cp /cyberange/apps/moa-release-2021.07.0/lib/moa.jar \
  -javaagent:/cyberange/apps/moa-release-2021.07.0/lib/sizeofag-1.0.4.jar \
  moa.DoTask 'WriteStreamToARFFFile \
  -s (generators.HyperplaneGenerator -k 10 -t 0.001 -s 10) -f concept_drift.arff -m 100000'
```

Get stream classifier accuracy: `pdm run python concept_drift.py`.

