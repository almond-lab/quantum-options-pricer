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
| `build_sampler` | backend, device_index | `SamplerV2` |
| `run_iae` | problem, sampler, epsilon, alpha | raw `IAEResult` |
| `interpret_results` | call, problem, result, params | `PricingResult` dataclass |

---

## GPU Backend Configuration

| GPU | Architecture | CUDA SM | `GPU_ARCH` build arg |
|---|---|---|---|
| NVIDIA T4 | Turing | sm_75 | `75` (default) |
| NVIDIA L4 | Ada Lovelace | sm_89 | `89` |

The simulation method is `tensor_network` via **NVIDIA cuTensorNet** (part of the cuQuantum SDK). This contracts the circuit tensor network on GPU, enabling circuits with 20–30 qubits that would be intractable with statevector simulation.

`cuStateVec_enable: False` is explicitly set — we use cuTensorNet, not cuStateVec, because tensor contraction scales better with circuit width (large number of qubits with moderate entanglement).

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
| `num_qubits` | int | `3` | 3–10 | Uncertainty qubits (precision vs. speed) |
| `epsilon_target` | float | `0.01` | 0.005–0.1 | IAE precision target |
| `alpha` | float | `0.05` | 0.01–0.1 | Confidence level (CI = 1-alpha) |

---

## Accuracy vs. Speed Trade-off

| `num_qubits` | Price levels | Typical circuit width | GPU time (T4) | vs. Black-Scholes |
|---|---|---|---|---|
| 3 | 8 | ~8 qubits | < 1s | ±5–15% |
| 5 | 32 | ~12 qubits | ~5s | ±1–3% |
| 8 | 256 | ~18 qubits | ~60s | ±0.1–0.5% |
| 10 | 1024 | ~22 qubits | minutes | ±0.01% |

---

## Environment Variables

See `.env.example` for full reference. Key variables:

| Variable | Default | Description |
|---|---|---|
| `BACKEND` | `gpu` | `gpu` or `cpu` (cpu = dev/test only) |
| `GPU_DEVICE` | `0` | CUDA device index |
| `GPU_ARCH` | `75` | Build-time CUDA arch (75=T4, 89=L4) |
| `LOG_LEVEL` | `INFO` | DEBUG/INFO/WARNING/ERROR |
| `NUM_SHOTS` | `8192` | Default shots (overridable per-request) |
| `PRECISION` | `double` | `double` or `single` |

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
  -d '{"spot":100,"strike":100,"days_to_expiry":365,"volatility":0.2,"risk_free_rate":0.05}'
```
