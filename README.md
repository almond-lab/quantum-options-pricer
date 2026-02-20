# Quantum Options Pricer

A REST microservice that prices **European Vanilla Options** (Calls & Puts) using
**Quantum Amplitude Estimation (IAE)** running on NVIDIA GPU (T4 / L4) via
Qiskit-Aer + cuTensorNet.

- **PUT /price** — prices a call or put using a quantum circuit; puts use put-call parity
- **GET /health** — backend status (GPU/CPU, sampler type)
- **GET /docs** — interactive Scalar API reference

---

## Table of Contents

1. [Run locally (Docker Compose)](#1-run-locally-docker-compose)
2. [Run locally (bare Python)](#2-run-locally-bare-python)
3. [Deploy to Kubernetes (GKE)](#3-deploy-to-kubernetes-gke)
4. [Open the Scalar API docs](#4-open-the-scalar-api-docs)
5. [Pricing examples (curl & Python)](#5-pricing-examples)
6. [Request reference](#6-request-reference)
7. [Response reference](#7-response-reference)

---

## 1. Run locally (Docker Compose)

### Prerequisites
- Docker ≥ 24 + Docker Compose v2
- NVIDIA GPU + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (skip for CPU-only mode)
- A [Databento](https://databento.com) API key (needed only when `volatility` is omitted)

### Steps

```bash
# 1. Clone and enter the repo
git clone git@github.com:almond-lab/quantum-options-pricer.git
cd quantum-options-pricer

# 2. Set your API key
cp .env.example .env
#    edit .env → set DATABENTO_API_KEY=your_key_here

# 3. Build and start (GPU, T4 by default)
docker compose up --build

# For L4 (sm_89):
GPU_ARCH=89 docker compose up --build

# CPU-only (no GPU required, slower — use num_qubits=4 for speed):
BACKEND=cpu docker compose up --build
```

The service is ready when you see:
```
INFO:     Application startup complete.
```

API is now live at **http://localhost:8000**

---

## 2. Run locally (bare Python)

Requires Python 3.11 and a CUDA 12.4 environment for GPU mode.

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure env
cp .env.example .env
# edit .env → set DATABENTO_API_KEY

# Start the server
PYTHONPATH=. uvicorn api.main:app \
  --host 0.0.0.0 --port 8000 \
  --log-config config/logging.yaml

# CPU mode (no GPU needed):
BACKEND=cpu PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000 \
  --log-config config/logging.yaml
```

> **Tip:** On CPU, use `num_qubits: 4` in requests — 8 qubits takes ~87 s on CPU vs ~100–500 ms on a T4.

---

## 3. Deploy to Kubernetes (GKE)

### One-time cluster setup

```bash
# 1. Create a GPU node pool (T4)
gcloud container node-pools create gpu-pool \
  --cluster YOUR_CLUSTER \
  --zone us-central1-a \
  --machine-type n1-standard-4 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --num-nodes 1

# 2. Install the NVIDIA device plugin (once per cluster)
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-device-plugin.yaml

# 3. Build and push image to Artifact Registry
export PROJECT=your-gcp-project-id
gcloud auth configure-docker us-docker.pkg.dev
docker build --build-arg CUDA_ARCH=75 \
  -t us-docker.pkg.dev/${PROJECT}/quantum-pricer/api:latest .
docker push us-docker.pkg.dev/${PROJECT}/quantum-pricer/api:latest
```

### Edit the deployment manifest

In `k8s/deployment.yaml`, replace the placeholder:
```yaml
image: us-docker.pkg.dev/YOUR_PROJECT/quantum-pricer/api:latest
#                         ^^^^^^^^^^^^
```

### Deploy

```bash
# Namespace
kubectl apply -f k8s/namespace.yaml

# Secret from your .env file
kubectl -n quantum-pricer create secret generic quantum-pricer-secret \
  --from-env-file=.env \
  --dry-run=client -o yaml | kubectl apply -f -

# Deploy everything
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Watch rollout
kubectl -n quantum-pricer rollout status deployment/quantum-pricer
```

### Verify

```bash
# Check the pod picked up a GPU
kubectl -n quantum-pricer describe pod -l app=quantum-pricer | grep -A4 Limits

# Port-forward and hit health
kubectl -n quantum-pricer port-forward svc/quantum-pricer 8000:80
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "backend": "gpu",
  "sampler_type": "AerSamplerV2",
  "gpu_available": true,
  "num_qubits_default": 8
}
```

> For L4 GPUs swap `CUDA_ARCH=89` in the build and `nodeSelector: nvidia-l4` in the Deployment.
> Full GKE guide: [`k8s/README.md`](k8s/README.md)

---

## 4. Open the Scalar API docs

Once the service is running, open **http://localhost:8000/docs** in your browser.

You'll see the full interactive Scalar API reference with:
- All request fields with descriptions and examples
- A **Try it out** panel to run live requests directly from the browser
- Full response schema including circuit metadata

> On Kubernetes with port-forward active (`kubectl port-forward svc/quantum-pricer 8000:80`), the same URL works: **http://localhost:8000/docs**

---

## 5. Pricing examples

### Health check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "backend": "gpu",
  "sampler_type": "AerSamplerV2",
  "gpu_available": true,
  "num_qubits_default": 8
}
```

---

### Example 1 — Fully manual (no market data fetch)

All parameters supplied explicitly. No Databento key or internet access required.

```bash
curl -s -X POST http://localhost:8000/price \
  -H "Content-Type: application/json" \
  -d '{
    "spot":           260.58,
    "strike":         260.00,
    "days_to_expiry": 30,
    "volatility":     0.30,
    "risk_free_rate": 0.036,
    "option_type":    "call",
    "num_qubits":     8,
    "epsilon_target": 0.01,
    "alpha":          0.05
  }' | python3 -m json.tool
```

**Response:**
```json
{
  "ticker": null,
  "option_type": "call",
  "spot": 260.58,
  "strike": 260.0,
  "days_to_expiry": 30,
  "time_to_expiry_years": 0.08219178,
  "volatility": 0.3,
  "risk_free_rate": 0.036,
  "market_data_used": false,
  "market_data_source": null,
  "price": 9.1842,
  "confidence_interval": { "lower": 9.0912, "upper": 9.2772 },
  "calculated_via_parity": false,
  "num_qubits": 8,
  "price_bins": 256,
  "total_circuit_qubits": 17,
  "grover_depth": 6,
  "oracle_queries": 157,
  "iae_rounds": 4,
  "epsilon_target": 0.01,
  "alpha": 0.05,
  "backend": "gpu"
}
```

---

### Example 2 — Auto-fetch spot + vol from live market data (AAPL call)

Omit `spot` and `volatility` — they are fetched automatically from yfinance and
Databento OPRA respectively. Only `ticker`, `strike`, and `days_to_expiry` are required.

```bash
curl -s -X POST http://localhost:8000/price \
  -H "Content-Type: application/json" \
  -d '{
    "ticker":         "AAPL",
    "strike":         230.00,
    "days_to_expiry": 21,
    "option_type":    "call"
  }' | python3 -m json.tool
```

**Response** (values will reflect live market data at call time):
```json
{
  "ticker": "AAPL",
  "option_type": "call",
  "spot": 227.52,
  "strike": 230.0,
  "days_to_expiry": 21,
  "time_to_expiry_years": 0.05753425,
  "volatility": 0.2814,
  "risk_free_rate": 0.03595,
  "market_data_used": true,
  "market_data_source": "rfr=yfinance(^IRX,3.595%) | spot=yfinance(AAPL,227.5200) | vol=databento(AAPL240221C00230000,bid=2.45,ask=2.55,mid=2.50,T=21d)",
  "price": 2.3871,
  "confidence_interval": { "lower": 2.2931, "upper": 2.4811 },
  "calculated_via_parity": false,
  "num_qubits": 8,
  "price_bins": 256,
  "total_circuit_qubits": 17,
  "grover_depth": 6,
  "oracle_queries": 157,
  "iae_rounds": 4,
  "epsilon_target": 0.01,
  "alpha": 0.05,
  "backend": "gpu"
}
```

---

### Example 3 — Put via put-call parity (auto-fetch all market data)

```bash
curl -s -X POST http://localhost:8000/price \
  -H "Content-Type: application/json" \
  -d '{
    "ticker":         "AAPL",
    "strike":         230.00,
    "days_to_expiry": 21,
    "option_type":    "put"
  }' | python3 -m json.tool
```

Note `"calculated_via_parity": true` in the response — the put price is derived
from the call circuit result using P = C − S₀ + K·e^(−rT).

---

### Example 4 — Fast CPU test (4 qubits, all manual)

Use `num_qubits: 4` for quick iteration on a machine without a GPU.

```bash
curl -s -X POST http://localhost:8000/price \
  -H "Content-Type: application/json" \
  -d '{
    "spot":           100.00,
    "strike":         100.00,
    "days_to_expiry": 30,
    "volatility":     0.20,
    "risk_free_rate": 0.05,
    "option_type":    "call",
    "num_qubits":     4
  }' | python3 -m json.tool
```

---

### Python requests equivalent

```python
import requests

resp = requests.post(
    "http://localhost:8000/price",
    json={
        "ticker":         "AAPL",
        "strike":         230.00,
        "days_to_expiry": 21,
        "option_type":    "call",
    },
    timeout=30,
)
resp.raise_for_status()
data = resp.json()

print(f"Price : ${data['price']:.4f}")
print(f"95% CI: [{data['confidence_interval']['lower']:.4f}, {data['confidence_interval']['upper']:.4f}]")
print(f"Source: {data['market_data_source']}")
print(f"Backend: {data['backend']}  |  {data['iae_rounds']} IAE rounds  |  {data['oracle_queries']} oracle queries")
```

---

## 6. Request reference

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `ticker` | string | When `spot` or `volatility` omitted | — | Equity root ticker, e.g. `"AAPL"` |
| `spot` | float | No | auto-fetch | Underlying price (USD). Fetched from yfinance if omitted |
| `strike` | float | **Yes** | — | Strike price (USD) |
| `days_to_expiry` | int | **Yes** | — | Calendar days until expiry |
| `volatility` | float | No | auto-fetch | Annualised IV, e.g. `0.30` = 30%. Fetched via Databento OPRA if omitted |
| `risk_free_rate` | float | No | auto-fetch | Annualised rate, e.g. `0.036`. Fetched from yfinance `^IRX` if omitted |
| `option_type` | `"call"` \| `"put"` | No | `"call"` | Put is priced via put-call parity |
| `num_qubits` | int (3–16) | No | `8` | Uncertainty qubits — 8 = 256 bins (GPU prod default); use 4 for CPU dev |
| `epsilon_target` | float (0.005–0.1) | No | `0.01` | IAE precision target |
| `alpha` | float (0.01–0.1) | No | `0.05` | Significance level — CI is `1 - alpha` (95% default) |

---

## 7. Response reference

| Field | Description |
|-------|-------------|
| `price` | Quantum option price in USD |
| `confidence_interval.lower/upper` | IAE confidence interval (USD) |
| `calculated_via_parity` | `true` if put price derived via parity, not a separate circuit |
| `market_data_used` | `true` if any field was auto-fetched |
| `market_data_source` | Human-readable provenance string (sources + resolved values) |
| `num_qubits` / `price_bins` | Grid resolution: `price_bins = 2^num_qubits` |
| `total_circuit_qubits` | Total qubits in the Grover circuit |
| `grover_depth` | Grover operator applications per IAE round |
| `oracle_queries` | Total oracle calls across all IAE rounds |
| `iae_rounds` | Number of IAE iterations performed |
| `backend` | `"gpu"` (AerSamplerV2 + cuTensorNet) or `"cpu"` (StatevectorSampler) |
