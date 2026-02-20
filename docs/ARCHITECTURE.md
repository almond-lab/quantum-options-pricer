# Quantum Options Pricer — Architecture

## Overview

This service prices **European Vanilla Options (Calls and Puts)** using quantum amplitude estimation running on NVIDIA GPUs via CUDA-accelerated tensor networks. It replaces the closed-form Black-Scholes equation with a quantum circuit that directly simulates the probability distribution of future stock prices and estimates the expected option payoff.

The service is exposed as a **FastAPI microservice** — callable from any frontend, backtesting script, or Excel plugin via HTTP POST.

---

## Why Quantum?

Black-Scholes solves a differential equation analytically. Quantum Amplitude Estimation (QAE) instead:

1. **Encodes** the probability distribution of all possible future stock prices directly into qubit amplitudes — a superposition of outcomes simultaneously
2. **Applies** the payoff function as a quantum circuit — marking which states are in-the-money
3. **Measures** the expected payoff using Grover-like iterations with a provable quadratic speedup over classical Monte Carlo

Classical Monte Carlo needs O(1/ε²) samples for precision ε. QAE needs O(1/ε). On large circuits, the GPU tensor network backend makes the simulation tractable.

---

## Quantum Strategy (3 Steps)

### Step 1 — Load Uncertainty
A `LogNormalDistribution` circuit encodes the terminal stock price `S_T` into `2^n` amplitude bins across a price range `[low, high]`:

```
ln(S_T) ~ N(mu, sigma²)

mu    = ln(S₀) + (r - 0.5·vol²)·T
sigma = vol·√T
low   = E[S_T] - 3·Std[S_T]
high  = E[S_T] + 3·Std[S_T]
```

With `n` qubits, the distribution is discretised into `2^n` price levels. More qubits = finer resolution = closer to the continuous Black-Scholes price.

### Step 2 — Compute Payoff
`EuropeanCallPricing` appends a comparator circuit that:
- Marks states where `S_T > K` (in-the-money)
- Rotates an ancilla qubit by an angle proportional to `max(S_T - K, 0)`
- The ancilla encodes the payoff magnitude in its amplitude

### Step 3 — Amplitude Estimation
`IterativeAmplitudeEstimation` (IAE) applies Grover iterations to amplify the ancilla signal and extract:
- The expected payoff `E[max(S_T - K, 0)]`
- A confidence interval at the requested significance level `alpha`

The raw result is discounted: **Price = E[Payoff] × e^(-rT)**

---

## Put Options — Put-Call Parity

Puts are computed classically from the Call circuit result:

```
P = C - S₀ + K·e^(-rT)
```

**Why**: A dedicated Put circuit doubles gate count and GPU time for a result that pure math gives for free. All API responses include `calculated_via_parity: true` when a Put is returned for full provenance.

---

## System Architecture

```
HTTP POST /price
        │
        ▼
  api/routes.py          — request validation (Pydantic), response shaping
        │
        ▼
  pricer/circuit.py      — pure circuit construction (no GPU, no I/O)
  ┌─────────────────────────────────────────────────────────┐
  │  derive_lognormal_params()  → LogNormalParams            │
  │  build_uncertainty_model()  → LogNormalDistribution      │
  │  build_estimation_problem() → EstimationProblem          │
  └─────────────────────────────────────────────────────────┘
        │
        ▼
  pricer/engine.py       — GPU execution + financial interpretation
  ┌─────────────────────────────────────────────────────────┐
  │  build_sampler()     → SamplerV2 (cuTensorNet or CPU)   │
  │  run_iae()           → IAEResult                         │
  │  interpret_results() → PricingResult (price + CI)        │
  └─────────────────────────────────────────────────────────┘
        │
        ▼
  JSON Response
```

---

## Market Data — Databento OPRA.PILLAR

The service fetches real NBBO quotes for the full options chain via Databento's
Pay-As-You-Go Historical API. Two calls are made per pricing session:

```
Session init
        │
        ├─ fetch_options_definitions()
        │    schema=DEFINITION, stype_in=PARENT, symbol=AAPL.OPT
        │    → all strikes + expirations for the trading day
        │
        └─ fetch_nbbo_snapshot()
             schema=CBBO_1M, stype_in=PARENT, symbol=AAPL.OPT
             window: 15:59:00 → 16:00:00 ET (1-minute close snapshot)
             → last consolidated BBO per instrument (bid, ask, mid, spread)
```

**Cost strategy**: CBBO_1M (1-minute sampled BBO) gives exactly one row per
instrument per minute. AAPL has ~4,000 active contracts; requesting 1 minute of
CBBO_1M captures the close quote for all of them in a single API call at minimal
cost (fractions of a cent vs. tick-level CMBP_1 which would return millions of rows).

**OPRA.PILLAR specifics**:
- Symbol format: `AAPL.OPT` (parent symbology, not bare ticker)
- Supported NBBO schemas: `cbbo-1m`, `cbbo-1s`, `cmbp-1`, `tcbbo` — **not** `mbp-1`
- Pay-As-You-Go embargo: most recent 1–2 trading sessions require a live data license.
  `get_options_chain()` automatically steps back up to 5 trading days to find
  available historical data.

---

## Module Reference

### `config/settings.py`
Pydantic-settings class loaded from environment / `.env`. Single source of truth for all defaults. Retrieved via `get_settings()` (LRU-cached — loaded once per process).

### `config/logging.yaml`
Structured logging config: rotating JSON file at `/app/logs/pricer.log` + human-readable console output. JSON format is suitable for log aggregators (Datadog, CloudWatch, GCP Logging).

### `pricer/circuit.py`
**Pure functions. No side effects.** Only imports: numpy, qiskit-finance, qiskit-algorithms. Responsible for all quantum circuit construction and mathematical parameter derivation.

| Function | Input | Output |
|---|---|---|
| `derive_lognormal_params` | spot, r, vol, T | `LogNormalParams` dataclass |
| `build_uncertainty_model` | num_qubits, params | `LogNormalDistribution` |
| `build_estimation_problem` | num_qubits, strike, params, model | `(EuropeanCallPricing, EstimationProblem)` |

### `pricer/engine.py`
GPU execution and financial result interpretation. Imports qiskit-aer (SamplerV2).

| Function | Input | Output |
|---|---|---|
| `build_sampler` | backend, device_index | `SamplerV2` (GPU) or `StatevectorSampler` (CPU) |
| `run_iae` | problem, sampler, epsilon_target, alpha | raw `IAEResult` |
| `interpret_results` | call, problem, result, params | `PricingResult` dataclass |

### `data/market_data.py`
Databento ingestion pipeline. No quantum dependencies — produces clean DataFrames
for downstream IV inversion and pricing.

| Function | Output |
|---|---|
| `get_options_chain(ticker)` | `OptionsChainSnapshot` (calls + puts DataFrames) |
| `fetch_options_definitions(client, ticker, day)` | instrument metadata DataFrame |
| `fetch_nbbo_snapshot(client, ticker, day)` | bid/ask/mid/spread DataFrame |
| `build_options_chain(defs, nbbo, day, ticker)` | joined snapshot with DTE columns |

---

## GPU Backend Configuration

| GPU | Architecture | CUDA SM | `GPU_ARCH` build arg |
|---|---|---|---|
| NVIDIA T4 | Turing | sm_75 | `75` (default) |
| NVIDIA L4 | Ada Lovelace | sm_89 | `89` |

The simulation method is `tensor_network` via **NVIDIA cuTensorNet** (part of the cuQuantum SDK). This contracts the circuit tensor network on GPU, enabling circuits with 20–30 qubits that would be intractable with statevector simulation.

`cuStateVec_enable: False` is explicitly set — we use cuTensorNet, not cuStateVec,
because tensor contraction scales better with circuit width (large number of qubits
with moderate entanglement).

### Runtime GPU Detection

At startup, `build_sampler()` checks `AerSimulator().available_devices()`:

- **GPU available** (`'GPU'` in devices): uses `AerSamplerV2` with `method='tensor_network'`
- **No GPU** (CPU-only instance, dev environment): falls back to `qiskit.primitives.StatevectorSampler`

The current development environment has **no GPU** (`nvidia-smi` not found;
`available_devices()` returns `('CPU',)` only). The CPU fallback is functional
but orders of magnitude slower — see the performance table below.

---

## Performance Benchmarks

> Measured on CPU (no GPU). All quantum runs use `StatevectorSampler`.
> GPU estimates derived from typical T4 cuTensorNet speedups for circuits of this depth.

### Black-Scholes Reference

| Mode | Latency | Notes |
|---|---|---|
| Scalar (one call + put pair) | **~272 µs** | Per-request API path |
| Vectorised (10,000 strikes, NumPy) | **~1,850 µs total / ~0.19 µs per strike** | Batch chain pricing |

Black-Scholes is essentially free — 6 floating-point operations, two `norm.cdf` lookups.

### Quantum — CPU `StatevectorSampler`

Parameters: S=260.58, K=260, T=0.25yr, vol=30%, ε=0.01, α=0.05

| Uncertainty qubits | Price bins | Total circuit qubits | Build time | IAE time | **Total** | Error vs BS |
|---|---|---|---|---|---|---|
| 4 | 16 | 9 | 33 ms | 375 ms | **0.4 s** | ±2–5% |
| 6 | 64 | 11 | 71 ms | 1,696 ms | **1.8 s** | ±1–3% |
| 8 | 256 | 13 | 200 ms | ~87,000 ms | **~87 s** | **<1%** |
| 10 | 1,024 | 15 | — | — | **not practical** | statevector too large |

**Why 8 is the default**: 256 bins gives <1% error vs Black-Scholes. On CPU
that takes ~87s which is development-only. On GPU tensor network (T4) the same
circuit runs in ~100–500 ms — see below.

### Quantum — GPU `tensor_network` (T4, estimated)

The tensor network method contracts the circuit as a sequence of tensor
operations rather than simulating the full 2^N statevector. For circuits with
depth ~10–20 and moderate entanglement, the speedup over CPU statevector is
roughly:

| Uncertainty qubits | CPU time | GPU T4 estimate | Speedup | Practical? |
|---|---|---|---|---|
| 4 | 0.4 s | ~10–30 ms | ~20× | Yes |
| 6 | 1.8 s | ~20–80 ms | ~30× | Yes |
| **8 (default)** | ~87 s | **~100–500 ms** | **~100–500×** | **Yes — production target** |
| 10 | not practical | ~500 ms–2 s | — | Yes |
| 12 | not practical | ~1–5 s | — | Yes |
| 16 | not practical | ~5–30 s | — | Yes, for research |

### BS vs Quantum — Summary

```
                BS scalar   4q GPU    8q GPU    8q CPU
Latency:         ~272 µs   ~20 ms    ~200 ms    ~87 s
Error vs BS:       0%      ±2–5%     <1%        <1%
Scales to exotic: No        Yes       Yes        Yes (slow)
```

Black-Scholes wins on speed by 3–4 orders of magnitude for vanilla Europeans.
The quantum engine's advantage appears when:
- **Pricing exotic payoffs** (barriers, Asians, cliquets) where no closed-form exists
- **Running full volatility surface calibration** across thousands of strikes/expiries
  simultaneously on GPU — the same circuit infrastructure, different payoff function
- **Research into quantum-native Greeks** (circuit-based parameter differentiation)

---

## API Request Parameters

| Parameter | Type | Default | Constraint | Description |
|---|---|---|---|---|
| `spot` | float | required | > 0 | Current stock price |
| `strike` | float | required | > 0 | Option strike price |
| `days_to_expiry` | int | required | > 0 | Calendar days to expiry |
| `volatility` | float | required | > 0 | Annualised implied volatility (e.g. 0.2 = 20%) |
| `risk_free_rate` | float | required | ≥ 0 | Annualised risk-free rate |
| `option_type` | str | `"call"` | call/put | Option type |
| `num_qubits` | int | **`8`** | 3–16 | Uncertainty qubits. **8 is the production default** (256 bins, <1% vs BS on GPU). Use 4–6 for fast dev/test. |
| `epsilon_target` | float | `0.01` | 0.005–0.1 | IAE precision target |
| `alpha` | float | `0.05` | 0.01–0.1 | Confidence level (CI = 1-alpha) |

---

## Environment Variables

See `.env.example` for full reference. Key variables:

| Variable | Default | Description |
|---|---|---|
| `BACKEND` | `gpu` | `gpu` or `cpu`. CPU falls back to `StatevectorSampler` automatically if no CUDA devices found. |
| `GPU_DEVICE` | `0` | CUDA device index |
| `GPU_ARCH` | `75` | Build-time CUDA arch (75=T4, 89=L4) |
| `LOG_LEVEL` | `INFO` | DEBUG/INFO/WARNING/ERROR |
| `NUM_SHOTS` | `8192` | Default shots (overridable per-request) |
| `PRECISION` | `double` | `double` or `single` |
| `DATABENTO_API_KEY` | — | Pay-As-You-Go key from databento.com/portal/keys |

---

## Kubernetes GPU Deployment

### How GPU Detection Works at Runtime

`build_sampler()` in `pricer/engine.py` checks for GPU availability automatically
at every call — no code change needed between environments:

```python
devices = AerSimulator().available_devices()
if "GPU" in devices:
    # AerSamplerV2 with method='tensor_network' via NVIDIA cuTensorNet
else:
    # Fallback: qiskit.primitives.StatevectorSampler (CPU)
```

When the pod lands on a GPU node with the NVIDIA driver visible, it picks up
`tensor_network` automatically. The current dev environment has no GPU and runs
the CPU fallback.

### Expected Latency Improvement on GPU Cluster

| Environment | 8-qubit latency | Notes |
|---|---|---|
| CPU-only (dev, this instance) | ~87 s | StatevectorSampler, 2^13 amplitudes sequentially |
| **T4 GPU pod** | **~100–500 ms** | cuTensorNet tensor contraction, ~100–500× faster |
| **L4 GPU pod** | **~50–200 ms** | Higher memory bandwidth than T4 |

The gain is not linear — it's structural. CPU statevector simulation stores all
2^13 = 8,192 complex amplitudes and evolves them gate-by-gate sequentially.
cuTensorNet instead contracts the circuit as a tensor network on thousands of
CUDA cores in parallel, keeping everything in GPU VRAM across IAE iterations
without host round-trips.

### Pod Spec Requirements

The container must have GPU access declared explicitly. Minimum Kubernetes pod spec:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: quantum-pricer
spec:
  runtimeClassName: nvidia          # required — exposes NVIDIA driver to container
  containers:
    - name: pricer
      image: quantum-options-pricer:latest
      resources:
        limits:
          nvidia.com/gpu: 1         # request exactly 1 GPU
        requests:
          nvidia.com/gpu: 1
      env:
        - name: BACKEND
          value: "gpu"
        - name: GPU_DEVICE
          value: "0"
        - name: GPU_ARCH
          value: "75"               # 75 for T4, 89 for L4
      ports:
        - containerPort: 8000
```

Or as a Deployment with a GPU node selector:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quantum-pricer
spec:
  replicas: 1                       # keep at 1 — GPU state is not fork-safe
  selector:
    matchLabels:
      app: quantum-pricer
  template:
    metadata:
      labels:
        app: quantum-pricer
    spec:
      runtimeClassName: nvidia
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4   # GKE example
        # accelerator: nvidia-l4                            # swap for L4
      containers:
        - name: pricer
          image: quantum-options-pricer:latest
          resources:
            limits:
              nvidia.com/gpu: 1
            requests:
              nvidia.com/gpu: 1
          envFrom:
            - secretRef:
                name: quantum-pricer-env   # kubectl create secret generic from .env
          ports:
            - containerPort: 8000
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 5
```

### Common Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Still using CPU fallback on GPU node | `runtimeClassName: nvidia` missing | Add it to pod spec |
| `available_devices()` returns only `('CPU',)` | NVIDIA device plugin not installed in cluster | Install `nvidia-device-plugin` DaemonSet |
| `CUDA error: no kernel image` | Wrong `GPU_ARCH` build arg | Rebuild with correct arch (75=T4, 89=L4) |
| `cuTensorNet not found` at startup | `cuquantum-python-cu12` not installed in image | Verify Dockerfile stage copies it from builder |
| OOM on GPU | Too many qubits or concurrent requests | Reduce `num_qubits` or keep `replicas: 1` |

### Verifying GPU is Active

After deploying, check the startup log for the sampler line:

```
# GPU active (what you want):
INFO | pricer.engine | Sampler: AerSamplerV2 | method=tensor_network | device=GPU:0

# CPU fallback (dev only):
WARNING | pricer.engine | GPU backend not requested — using StatevectorSampler (CPU reference)
```

---

## Docker

```bash
# Build for T4 (default)
docker compose build

# Build for L4
GPU_ARCH=89 docker compose build

# Run
docker compose up

# Test endpoint
curl -X POST http://localhost:8000/price \
  -H "Content-Type: application/json" \
  -d '{"spot":260,"strike":260,"days_to_expiry":30,"volatility":0.30,"risk_free_rate":0.053}'
```
