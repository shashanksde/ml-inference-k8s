# ml-inference-k8s

A small, self-contained example that ties together three things:

1. **ML inference** — a handwritten-digit (0–9) classifier served over HTTP with FastAPI.
2. **Containerization** — a Dockerfile that bakes the trained model into the image.
3. **Kubernetes** — Deployment, Service, HPA, and ConfigMap manifests, with the app's
   health endpoints wired to K8s liveness/readiness probes.

## What the model is

The classifier is a `StandardScaler → LogisticRegression` scikit-learn pipeline trained on
scikit-learn's bundled [`load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
dataset: 8×8 grayscale handwritten digits with pixel values 0–16. It's the same task as MNIST —
recognize a digit 0–9 — at a smaller resolution. This choice keeps the project **fully offline and
reproducible**: training takes a couple of seconds and needs no dataset download. Test accuracy is
~0.97.

## Project layout

```
app/            FastAPI service + model training/inference
  model.py      train / save / load / predict
  train.py      `python -m app.train` — writes digits_model.joblib
  main.py       FastAPI app (/healthz, /readyz, /metadata, /predict)
tests/          pytest API tests (TestClient)
k8s/            Deployment, Service, HPA, ConfigMap
Dockerfile      slim image; trains the model at build time
.github/        CI: pytest + docker build validation
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train and persist the model artifact
python -m app.train

# Run the tests
pytest -q

# Start the API
uvicorn app.main:app --reload --port 8000
```

Then:

```bash
curl localhost:8000/healthz
curl localhost:8000/metadata

# Predict on a real sample (first digit from the dataset):
python - <<'PY'
import json, urllib.request
from sklearn.datasets import load_digits
d = load_digits()
body = json.dumps({"pixels": d.data[0].tolist()}).encode()
req = urllib.request.Request("http://localhost:8000/predict", body,
                             {"Content-Type": "application/json"})
print(urllib.request.urlopen(req).read().decode())
print("true label:", int(d.target[0]))
PY
```

Interactive API docs are at `http://localhost:8000/docs`.

## API

| Method | Path        | Description                                            |
|--------|-------------|--------------------------------------------------------|
| GET    | `/healthz`  | Liveness — 200 while the process is up                 |
| GET    | `/readyz`   | Readiness — 200 once the model is loaded (503 while loading) |
| GET    | `/metadata` | Model type, feature count, labels, training accuracy   |
| POST   | `/predict`  | Body `{"pixels": [64 floats]}` → `{prediction, confidence, probabilities}` |

## Docker

```bash
docker build -t ghcr.io/shashanksde/ml-inference-k8s:latest .
docker run -p 8000:8000 ghcr.io/shashanksde/ml-inference-k8s:latest
```

The model is trained during `docker build` (`RUN python -m app.train`), so the image ships ready to
serve. The container runs as a non-root user.

## Kubernetes

Push an image to a registry your cluster can pull from (the manifests reference
`ghcr.io/shashanksde/ml-inference-k8s:latest`), then:

```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

kubectl port-forward svc/ml-inference-k8s 8000:80
curl localhost:8000/healthz
```

Highlights:

- **Liveness probe → `/healthz`** restarts a wedged pod.
- **Readiness probe → `/readyz`** keeps a pod out of the Service until the model has loaded.
- **HPA** scales 2→5 replicas at 70% CPU.
- Hardened `securityContext` (non-root, read-only root FS, dropped capabilities).

## Notes / limitations

- Uses sklearn `load_digits` (8×8) rather than downloading full 28×28 MNIST, for offline reproducibility.
- The Kubernetes and Docker layers are provided as working, validated config; deploying to a live
  cluster and pushing the image to a registry are the operator's steps.
