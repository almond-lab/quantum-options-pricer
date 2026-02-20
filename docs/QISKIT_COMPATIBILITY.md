# Qiskit Compatibility & Debugging Guide

> **Who this is for**: anyone adding features, increasing qubit counts, upgrading
> dependencies, or debugging a broken `/price` endpoint. These lessons were learned
> the hard way during the initial GPU deployment. Reading this first will save you
> hours.

---

## Contents

1. [The qiskit ecosystem is a minefield](#1-the-qiskit-ecosystem-is-a-minefield)
2. [How IAE circuits actually work — the full pipeline](#2-how-iae-circuits-actually-work-the-full-pipeline)
3. [Bug catalogue — every error we hit and why](#3-bug-catalogue)
4. [The two monkey-patches in engine.py and why they exist](#4-the-two-monkey-patches-in-enginepy)
5. [epsilon_target and IAE circuit growth](#5-epsilon_target-and-iae-circuit-growth)
6. [Scaling to more qubits — what breaks and when](#6-scaling-to-more-qubits)
7. [Debugging methodology — how to reproduce issues in the live pod](#7-debugging-methodology)
8. [Version matrix — what is pinned and why](#8-version-matrix)

---

## 1. The qiskit ecosystem is a minefield

Qiskit is split across several independent packages that are versioned separately and
have non-obvious compatibility constraints:

```
qiskit          — core circuits, transpiler, primitives
qiskit-aer      — simulators (CPU/GPU C++ backend)
qiskit-finance  — domain circuits (LogNormalDistribution, EuropeanCallPricing)
qiskit-algorithms — IAE, VQE, etc.
cuquantum       — NVIDIA tensor network bindings
```

The problem: **qiskit-finance and qiskit-algorithms lag behind qiskit core by 1–2
major versions**. Their circuits emit gate types that were valid in older qiskit
but have been renamed, restructured, or restricted in newer ones. Meanwhile
qiskit-aer's C++ assembler has its own gate allowlist that doesn't always match
what the Python layer accepts.

**Result**: circuits that construct successfully in Python can fail at simulation
time with cryptic C++ errors like `unknown instruction: P(X)` or
`unknown instruction: Q`.

The version set that works for this project is documented in [section 8](#8-version-matrix).
Do not upgrade any of these packages independently without re-running the full test
suite against a live GPU pod.

---

## 2. How IAE circuits actually work — the full pipeline

Understanding the full circuit lifecycle is essential for debugging. Here is every
step from API call to simulation result:

```
POST /price
    │
    ▼
pricer/circuit.py — build_uncertainty_model()
    │
    │  LogNormalDistribution(num_qubits)
    │  ├── encodes S_T ~ LogNormal into 2^n amplitude bins
    │  └── internally calls qiskit.extensions.isometry() to decompose
    │      the initial state into gates — this is where Patch 1 is needed
    │
    ▼
pricer/circuit.py — build_estimation_problem()
    │
    │  EuropeanCallPricing(num_qubits, strike, ...)
    │  ├── appends a comparator: marks states where S_T > K
    │  ├── rotates ancilla qubit proportional to max(S_T - K, 0)
    │  └── returns (EuropeanCallPricing, EstimationProblem)
    │       EstimationProblem has:
    │         .state_preparation  — the LogNormal circuit
    │         .grover_operator    — the Grover reflection operator
    │         .objective_qubits   — which qubit to measure
    │
    ▼
pricer/engine.py — run_iae()
    │
    │  IterativeAmplitudeEstimation.estimate(problem)
    │  │
    │  │  IAE runs multiple rounds. In each round k, it builds a circuit:
    │  │    [state_prep] [grover_op]^k [measurements]
    │  │
    │  │  grover_op.power(k) wraps grover_op k times.
    │  │  In qiskit 2.x, .power(k) returns a QuantumCircuit where each
    │  │  iteration is a Gate named "Q" whose .definition = grover_op.
    │  │  Gate Q is NOT a basis gate — the C++ assembler rejects it.
    │  │
    │  └── sampler.run([(circuit, shots)])
    │       ▼
    │       BackendSamplerV2._run_pubs()
    │         ▼
    │         _run_circuits(circuits, backend, **opts)   ← Patch 2 intercepts here
    │           ▼
    │           backend.run(circuits)  — qiskit-aer C++ assembler
    │             ▼
    │             GPU statevector simulation
    │
    ▼
pricer/engine.py — interpret_results()
    │
    └── EuropeanCallPricing.interpret(iae_result) → expected payoff
        discount by e^(-rT) → option price
```

The two patches in `engine.py` fix problems at steps marked above. Their placement
is intentional — they must fire at module import time, before any circuit is built.

---

## 3. Bug catalogue

Each error below was encountered in production. They are ordered chronologically as
they appeared, because some only surface after earlier ones are fixed.

---

### Bug 1 — `ZoneInfoNotFoundError: No time zone found with key America/New_York`

**Where**: startup, before any circuit is built.

**Why**: `yfinance` and `databento` both call `zoneinfo.ZoneInfo("America/New_York")`.
The NVIDIA CUDA *runtime* base image (`nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`)
strips the system `tzdata` package to save space. CUDA *devel* images (used in the
builder stage) include it. The Python `zoneinfo` stdlib has no fallback — it raises
immediately if the timezone database is missing.

**Fix**: add `tzdata>=2024.1` to `requirements.txt`. This installs the Python-bundled
timezone data that `zoneinfo` uses as a fallback when system tzdata is absent.

**Lesson**: the devel→runtime image split in multi-stage builds frequently drops
system libraries that worked in the build stage. Check your runtime base image's
apt package list explicitly, especially for anything time/locale/locale-related.

---

### Bug 2 — `AerError: 'Invalid option device_index'`

**Where**: `build_sampler()` during `AerSimulator(...)` construction.

**Why**: `device_index` was a valid constructor argument in qiskit-aer ≤ 0.15. It
was removed in 0.17.x. The option no longer exists and passing it raises immediately.

**Fix**: use `device="GPU"` (which takes a string, not an index). Multiple GPUs are
addressed by setting `CUDA_VISIBLE_DEVICES` in the environment, not in the
constructor.

**Lesson**: qiskit-aer 0.17.x has breaking API changes vs 0.15.x. The PyPI
`qiskit-aer-gpu` package is frozen at 0.15.1. This project builds 0.17.2 from
source (see `Dockerfile`). Release notes between major versions are essential
reading before upgrading.

---

### Bug 3 — `ImportError: libopenblas.so.0: cannot open shared object file`

**Where**: startup, during `import qiskit_aer`.

**Why**: qiskit-aer's Python extension (`controller_wrappers.so`) links against
OpenBLAS dynamically. In the multi-stage Dockerfile, `libopenblas-dev` is installed
in the **builder** stage for compilation. The compiled `.so` is then copied to the
**runtime** stage — but the runtime stage has no OpenBLAS shared library installed.
Linux's dynamic linker (`ld.so`) can't resolve the dependency at import time.

**Fix**: add `libopenblas0` to the `apt-get install` in the runtime stage of the
Dockerfile. This is the runtime-only library (no headers) containing the `.so` files.

**Lesson**: in multi-stage builds, every shared library needed at *runtime* must be
explicitly installed in the *runtime* stage. Running `ldd /path/to/your.so | grep
"not found"` on the compiled extension (from within the builder) will reveal all
missing runtime deps before you deploy.

---

### Bug 4 — `ValueError: Input matrix is not unitary` (during circuit construction)

**Where**: `build_uncertainty_model()`, inside `LogNormalDistribution.__init__()`.

**Why**: this is a floating-point accumulation problem specific to 8+ qubit circuits.

`LogNormalDistribution` encodes the initial quantum state using the *isometry
decomposition* algorithm. For an n-qubit circuit, this decomposes the initial state
vector into a sequence of two-qubit gates via repeated SVD. Each SVD step introduces
a small floating-point error O(machine epsilon). After ~256 such steps (needed for
an 8-qubit circuit), the accumulated error in the resulting gate matrix can reach
O(1e-7), exceeding the default unitarity tolerance of `1e-8` used by qiskit's
`UnitaryGate.__init__()`.

The check that fails is this line in qiskit (simplified):

```python
if not is_unitary_matrix(data, rtol=1e-5, atol=1e-8):
    raise ValueError("Input matrix is not unitary.")
```

For num_qubits ≤ 7 the accumulated error stays below 1e-8 and no exception is
raised. At num_qubits = 8 it starts to exceed the threshold.

**Fix** (Patch 1 in `engine.py`): monkey-patch `is_unitary_matrix` to use looser
tolerances — `rtol=1e-4, atol=1e-5`. The matrix is functionally unitary; the error
is a numerical artefact of the decomposition, not a real non-unitarity. The
simulation result is not affected.

```python
import qiskit.circuit.library.generalized_gates.unitary as _unitary_mod
_orig = _unitary_mod.is_unitary_matrix
_unitary_mod.is_unitary_matrix = (
    lambda mat, rtol=1e-4, atol=1e-5: _orig(mat, rtol=rtol, atol=atol)
)
```

**Why this placement?** The patch must be applied before any `LogNormalDistribution`
is instantiated. Placing it at module import time in `engine.py` guarantees it fires
before any pricing request is handled. The patch is additive — it does not change
any simulation logic, only the validation threshold.

**Lesson**: floating-point validation tolerances that work for small circuits often
break for larger ones due to accumulated decomposition error. If you see
`Input matrix is not unitary` for num_qubits ≥ 8, this is the reason. The fix is
mathematical (loosen tolerance) not numerical (the simulation is correct).

---

### Bug 5 — `AerError: unknown instruction: P(X)`

**Where**: `sampler.run()`, inside the C++ assembler.

**Why**: `EuropeanCallPricing` and its underlying comparator circuit use a Pauli
rotation gate that qiskit-finance 0.4.x emits as a gate named `P(X)`. This is a
`PauliGate` instance that wraps a Pauli-X evolution. In qiskit 2.x, this gate exists
in the Python layer but has no corresponding entry in qiskit-aer 0.17.2's C++
gate table. When the circuit reaches the C++ assembler, it encounters an instruction
name it does not recognise and raises.

The gate is valid — it just needs to be *transpiled* to primitive basis gates
(`cx`, `u`, etc.) before being submitted to qiskit-aer. At `optimization_level=1`
the transpiler's `BasisTranslator` pass handles this automatically.

**Fix**: see [Patch 2](#the-patch-2-fix) below.

---

### Bug 6 — `AerError: unknown instruction: Q`

**Where**: `sampler.run()`, inside the C++ assembler, during IAE rounds with k ≥ 2.

**Why**: this one is subtle. In qiskit 2.x, `QuantumCircuit.power(k)` does not
repeat the circuit's gates `k` times. Instead it wraps the entire circuit as a
single `Gate` named `Q` whose `.definition` is the original circuit, then places
`k` of those `Gate Q` instances in a new circuit.

So when IAE builds the round-k circuit:

```python
circuit.compose(grover_op.power(k), inplace=True)
```

The resulting circuit contains `k` instructions, each a `Gate Q` whose definition
is the full grover operator. This composes correctly in Python, but the C++
assembler sees an instruction named `Q` with no corresponding kernel and raises.

Crucially: if you pre-transpile `grover_op` before passing it to `EstimationProblem`
(e.g., converting it to basis gates), IAE then calls `.power(k)` on the *transpiled*
circuit — wrapping the transpiled basis-gate circuit inside `Gate Q`. The `Gate Q`
problem remains regardless of pre-transpilation.

**Fix**: see [Patch 2](#the-patch-2-fix) below — transpile must happen *after* IAE
builds the full round-k circuit, not before.

---

## 4. The two monkey-patches in engine.py

### Why monkey-patches at all?

The clean fix for bugs 5 and 6 would be to configure `BackendSamplerV2` to transpile
at `optimization_level=1`. No such configuration option exists in qiskit 2.3.0 /
qiskit-aer 0.17.2. The transpilation step must be injected into the call path.

### Patch 1 — UnitaryGate tolerance

```python
# In pricer/engine.py, at module level (before any circuit is built)
import qiskit.circuit.library.generalized_gates.unitary as _unitary_mod
_orig_is_unitary = _unitary_mod.is_unitary_matrix
_unitary_mod.is_unitary_matrix = (
    lambda mat, rtol=1e-4, atol=1e-5: _orig_is_unitary(mat, rtol=rtol, atol=atol)
)
```

**What it patches**: the module-level `is_unitary_matrix` function in qiskit's
`unitary.py`. Every `UnitaryGate.__init__()` call imports and calls this function
by name from the module's global namespace — so replacing the module attribute
affects all subsequent `UnitaryGate` constructions globally.

**Safety**: the tolerances `rtol=1e-4, atol=1e-5` were chosen by measuring the
actual accumulated error for 8-qubit isometry decompositions (~1e-7). The patch
accepts matrices that are unitary up to that error, which is numerically harmless
for quantum simulation.

### Patch 2 — BackendSamplerV2._run_circuits

This patch fixes bugs 5 and 6 by intercepting the circuit submission path and
applying `optimization_level=1` transpilation.

#### How BackendSamplerV2 submits circuits

```
BackendSamplerV2.run(pubs)
    └── _run(coerced_pubs)
          └── _run_pubs(pubs, shots)
                ├── bind parameters to circuits
                └── _run_circuits(circuits, self._backend, memory=True, shots=shots, ...)
                      └── backend.run(circuits, **opts)  ← no transpilation here
```

`_run_circuits` is defined in `qiskit/primitives/backend_estimator_v2.py` and
imported into `qiskit/primitives/backend_sampler_v2.py`:

```python
# backend_sampler_v2.py, line ~30
from qiskit.primitives.backend_estimator_v2 import _run_circuits
```

This import creates a binding `backend_sampler_v2._run_circuits` in that module's
global namespace. When `_run_pubs` later calls `_run_circuits(...)`, Python looks
up `_run_circuits` in `backend_sampler_v2`'s globals — so **replacing
`backend_sampler_v2._run_circuits` in that module's namespace intercepts every
circuit submission**.

#### The patch

```python
import qiskit.primitives.backend_sampler_v2 as _bsv2_mod
from qiskit.compiler import transpile
_orig_run_circuits = _bsv2_mod._run_circuits

def _run_circuits_opt1(circuits, backend, clear_metadata=True, **run_options):
    if not isinstance(circuits, list):
        circuits = [circuits]
    t = transpile(circuits, backend=backend, optimization_level=1)
    return _orig_run_circuits(t, backend, clear_metadata=clear_metadata, **run_options)

_bsv2_mod._run_circuits = _run_circuits_opt1
```

**Why `optimization_level=1` specifically?**

- `optimization_level=0`: runs `BasisTranslator` only. Decomposes non-basis gates
  to `cx`/`u` but does not run `UnitarySynthesis`. `UnitaryGate` instances that
  survived from the isometry decomposition may remain.
- `optimization_level=1` (default): runs `UnitarySynthesis` + `BasisTranslator` +
  `Optimize1qGatesDecomposition`. This guarantees that `Gate Q`, `P(X)`, and any
  remaining `UnitaryGate` instances are fully decomposed to `cx`/`u`/`rz`/etc.
- `optimization_level=2–3`: deeper rewrite passes (Clifford simplification, routing)
  — unnecessary overhead for our circuits.

**Why NOT `backend_estimator_v2._run_circuits`?**
An earlier attempt patched `qiskit.primitives.backend_estimator_v2._run_circuits`
instead of `backend_sampler_v2._run_circuits`. This had no effect. The reason:
`backend_sampler_v2.py` has already imported `_run_circuits` by the time our patch
runs. Patching the definition site (`backend_estimator_v2`) after the import-time
copy has been made in `backend_sampler_v2` does not affect the copy. The patch must
target the module that *uses* the function, not the module that *defines* it.

**Cost**: the `transpile()` call adds ~200–400 ms per circuit on first invocation
(due to plugin discovery). Subsequent calls to the same circuit shape are faster.
For an epsilon=0.05 run (1–2 IAE rounds), this is a one-time cost of ~200–800 ms
on top of the simulation time.

---

## 5. epsilon_target and IAE circuit growth

### How IAE rounds work

`IterativeAmplitudeEstimation` achieves precision `ε` using a sequence of circuit
evaluations, each with a different number of Grover operator applications `k`:

```
Round 1:  k = 1   →  [state_prep] [grover_op]^1  [measure]
Round 2:  k = 2   →  [state_prep] [grover_op]^2  [measure]
Round 3:  k = 4   →  [state_prep] [grover_op]^4  [measure]
Round 4:  k = 8   →  [state_prep] [grover_op]^8  [measure]
...
```

The k values roughly double each round. To achieve precision ε, IAE needs roughly
`O(1/ε)` total Grover queries, distributed across rounds.

### Circuit size growth

The transpiled gate count scales with k because `gate Q.definition = grover_op` and
`grover_op.power(k)` creates k copies of `Gate Q` in the circuit. After
`optimization_level=1` transpilation each `Gate Q` is fully unrolled to basis gates:

| IAE round | k | Transpiled gate count (8 qubits) |
|-----------|---|-----------------------------------|
| 1 | 1 | ~598 |
| 2 | 2 | ~4,870 |
| 3 | 4 | ~13,000+ |
| 4 | 8 | ~30,000+ |

Gate count grows roughly linearly with k, but qiskit-aer GPU simulation time grows
super-linearly (more statevector operations per gate application). For 8-qubit
circuits on a T4 GPU, the k=2 circuit (~4,870 gates) causes a CUDA runtime segfault
in qiskit-aer 0.17.2.

### What epsilon values are safe

| epsilon_target | Typical IAE rounds | Max k reached | GPU-safe (8 qubits)? |
|----------------|--------------------|---------------|----------------------|
| 0.10 | 1 | 1 | Yes |
| **0.05** (default) | **1–2** | **1** | **Yes** |
| 0.02 | 3–5 | 4 | Not reliably (k=2+ segfaults) |
| 0.01 | 5–8 | 8 | No — segfaults at k=2 round |

**The default epsilon_target is 0.05** for this reason. This is also a reasonable
practical precision for option pricing — at $260 spot, ε=0.05 on the amplitude
translates to a price confidence interval of roughly ±$7, which is acceptable for
a quantum estimate (compare to Black-Scholes for calibration).

### If you need higher precision

Options (in order of preference):
1. **Upgrade qiskit-aer** — newer versions may fix the GPU segfault for larger circuits
2. **Use more qubits** (see next section) — a finer price grid naturally reduces
   the estimation error at a given epsilon level
3. **Run CPU fallback** — `StatevectorSampler` has no segfault issue but is ~100×
   slower
4. **Reduce num_qubits temporarily** — a 4-qubit circuit runs many IAE rounds safely
   even at epsilon=0.01

---

## 6. Scaling to more qubits

### How qubits affect circuit size

Every additional uncertainty qubit doubles the size of the initial state preparation
(the isometry decomposition):

| num_qubits | Price bins | Total circuit qubits | state_prep gates (approx) | Grover op gates (approx) |
|------------|------------|----------------------|--------------------------|--------------------------|
| 4 | 16 | 9 | ~15 | ~40 |
| 6 | 64 | 11 | ~60 | ~200 |
| 8 (default) | 256 | 17 | ~200 | ~590 |
| 10 | 1,024 | 19 | ~800 | ~2,500 |
| 12 | 4,096 | 21 | ~3,200 | ~10,000 |

The "total circuit qubits" column grows by 2 per uncertainty qubit (one ancilla for
the comparator register, one for the payoff rotation).

### What breaks as you add qubits

**At num_qubits = 8**: the UnitaryGate tolerance bug first appears (Patch 1). Circuit
construction fails without the `is_unitary_matrix` patch.

**At num_qubits ≥ 9**: the transpiled Grover operator grows to thousands of gates.
Each IAE round with k ≥ 2 creates a circuit with 2× as many basis gates. GPU
simulation time and memory pressure increase significantly.

**At num_qubits ≥ 12**: the statevector holds 2^(total_qubits) complex amplitudes.
At 21 qubits that is 2^21 = 2M amplitudes × 16 bytes (double complex) = 32 MB, which
fits in T4 VRAM (16 GB). However the transpilation step becomes slow (30+ seconds).

**At num_qubits ≥ 16**: statevector simulation becomes impractical even on GPU.
Consider switching to `method='tensor_network'` (requires cuquantum) for circuits
this wide.

### Testing new qubit counts

Before changing `num_qubits` in production, test incrementally in the pod:

```python
kubectl exec <pod> -- python3 -c "
from pricer.circuit import derive_lognormal_params, build_uncertainty_model, build_estimation_problem

# Test circuit construction only (no simulation)
lnp = derive_lognormal_params(260, 0.036, 0.3, 30/365)
model = build_uncertainty_model(10, lnp)   # change num_qubits here
ec, problem = build_estimation_problem(10, 260, lnp, model)
print(f'Built OK: state_prep={problem.state_preparation.num_qubits}q')
print(f'Grover gates: {len(problem.grover_operator.decompose().data)}')
"
```

If construction succeeds, test a single IAE round (epsilon=0.1, alpha=0.1 for the
smallest possible circuit depth), then tighten.

---

## 7. Debugging methodology

### Principle: test in the running pod, not locally

Local environments differ from the pod in:
- GPU availability
- CUDA driver version
- Python package versions (especially native extensions)
- Environment variables (secrets, backend settings)
- tzdata availability

Always reproduce errors inside the pod with `kubectl exec` before assuming you
understand the root cause.

### Iterative diagnosis pattern

1. **Check startup logs first**:
   ```bash
   kubectl logs <pod> | head -50
   ```
   The two patch confirmation lines should appear:
   ```
   DEBUG | pricer.engine | UnitaryGate tolerance patched (rtol=1e-4, atol=1e-5)
   DEBUG | pricer.engine | BackendSamplerV2._run_circuits patched (optimization_level=1)
   ```
   If they don't appear, the ConfigMap isn't mounted or engine.py failed to import.

2. **Check health endpoint**:
   ```bash
   kubectl port-forward pod/<name> 8001:8000 &
   curl http://localhost:8001/health
   # Expect: {"backend":"gpu","sampler_type":"BackendSamplerV2","gpu_available":true}
   ```
   `gpu_available: false` means the CUDA driver isn't accessible to the pod.

3. **Isolate circuit construction from simulation**:
   ```python
   kubectl exec <pod> -- python3 -c "
   # Apply patches first (same as engine.py does at import time)
   import qiskit.circuit.library.generalized_gates.unitary as _um
   _orig = _um.is_unitary_matrix
   _um.is_unitary_matrix = lambda mat, rtol=1e-4, atol=1e-5: _orig(mat, rtol=rtol, atol=atol)

   from pricer.circuit import *
   lnp = derive_lognormal_params(260.58, 0.036, 0.3, 30/365)
   model = build_uncertainty_model(8, lnp)
   ec, problem = build_estimation_problem(8, 260, lnp, model)
   print('Circuit OK:', problem.state_preparation.num_qubits, 'qubits')
   "
   ```

4. **Test the transpiler step alone**:
   ```python
   from qiskit.compiler import transpile
   from qiskit_aer import AerSimulator
   sim = AerSimulator(method='statevector', device='GPU', precision='single')
   t = transpile(problem.state_preparation, backend=sim, optimization_level=1)
   print('Transpiled:', len(t.data), 'gates')
   # Check no unknown gates remain:
   gate_names = {inst.operation.name for inst in t.data}
   print('Gate types:', gate_names)
   # Should be only: {'cx', 'u', 'rz', 'sx', 'x', 'measure', 'barrier'}
   ```

5. **Test a single simulation call directly**:
   ```python
   t1 = transpile(problem.state_preparation, backend=sim, optimization_level=1)
   t1.measure_all()
   job = sim.run(t1, shots=1024)
   print(job.result().get_counts())
   ```

6. **Add verbose print statements to the patch functions** to trace circuit sizes
   across IAE rounds:
   ```python
   def _run_circuits_opt1(circuits, backend, clear_metadata=True, **run_options):
       t = transpile(circuits, backend=backend, optimization_level=1)
       print(f'IAE round: {len(circuits[0].data)} → {len(t[0].data)} gates')
       return _orig_run_circuits(t, backend, clear_metadata=clear_metadata, **run_options)
   ```

### Reading C++ error messages from qiskit-aer

qiskit-aer wraps its C++ exceptions in `AerError`. The error string is the only
diagnostic:

| Error string | Root cause | Fix |
|---|---|---|
| `unknown instruction: P(X)` | PauliGate from qiskit-finance not transpiled | Patch 2 / transpile with opt_level=1 |
| `unknown instruction: Q` | Gate Q from grover_op.power(k) not transpiled | Patch 2 / transpile with opt_level=1 |
| `unknown instruction: unitary` | UnitaryGate not synthesized away | Patch 2 at opt_level=1 (runs UnitarySynthesis) |
| `Input matrix is not unitary` | Float accumulation in isometry decomp | Patch 1 (loosen tolerance) |
| `Invalid option device_index` | Removed API in qiskit-aer 0.17+ | Remove device_index, use device='GPU' |

When `AlgorithmError: 'The job was not completed successfully.'` appears from
qiskit-algorithms, it is always a wrapper around one of the above AerErrors.
The actual root cause is in the traceback's `__context__` or `__cause__` chain —
look for `AerError` several frames down.

---

## 8. Version matrix

These versions are tested together and work. Do not upgrade any package individually
without re-testing the full GPU pipeline.

| Package | Version | Notes |
|---------|---------|-------|
| `qiskit` | 2.3.0 | Core circuits + transpiler + primitives |
| `qiskit-aer` | 0.17.2 | Must be **built from source** with CUDA for GPU support — PyPI `qiskit-aer-gpu` tops at 0.15.1 (incompatible with qiskit 2.x) |
| `qiskit-finance` | 0.4.1 | Emits P(X) gates — handled by Patch 2 |
| `qiskit-algorithms` | 0.4.0 | IAE V2-primitive aware; creates Gate Q via .power(k) — handled by Patch 2 |
| `cuquantum-python-cu12` | latest | NVIDIA tensor network bindings (needed for method='tensor_network' if used) |
| CUDA | 12.4.1 | Driver + runtime. Build arg `CUDA_ARCH=75` for T4, `89` for L4 |
| Python | 3.11 | 3.12 untested with this exact qiskit-aer source build |

### How to check what's installed in the pod

```bash
kubectl exec <pod> -- python3 -c "
import qiskit, qiskit_aer, qiskit_finance, qiskit_algorithms
print('qiskit:', qiskit.__version__)
print('qiskit-aer:', qiskit_aer.__version__)
print('qiskit-finance:', qiskit_finance.__version__)
print('qiskit-algorithms:', qiskit_algorithms.__version__)
from qiskit_aer import AerSimulator
print('GPU devices:', AerSimulator().available_devices())
"
```

Expected output on a healthy GPU pod:
```
qiskit: 2.3.0
qiskit-aer: 0.17.2
qiskit-finance: 0.4.1
qiskit-algorithms: 0.4.0
GPU devices: ('CPU', 'GPU')
```

### Upgrading safely

If you need to upgrade any of these packages:

1. Upgrade in the builder stage first, re-run the full GPU smoke test
2. Watch for: new gate names in circuits, changes to `.power()` behaviour,
   new/removed constructor arguments, changes to `_run_circuits` signature
3. Check if the `_bsv2_mod._run_circuits` patch still intercepts correctly after
   the upgrade (the function may have been renamed or the import structure changed)
4. Run `test_circuit.py` and `smoke_test.py` against a live GPU pod before merging
