# Incremental ETA

ETA (encrypted traffic analysis) with incremental training based on River.

## Prerequisites

* [PDM](https://pdm.fming.dev/)
* [River](https://riverml.xyz/)
* [moa](https://moa.cms.waikato.ac.nz/)
* [Weka](https://www.cs.waikato.ac.nz/ml/weka/)

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
  moa.DoTask 'WriteStreamToARFFFile -s (ConceptDriftStream \
    -s (generators.SEAGenerator -f 3) -d (generators.SEAGenerator -f 2) -p 50000 -w 1) \
    -f abrupt_drift.arff -m 100000'
```

