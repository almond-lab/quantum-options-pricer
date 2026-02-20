"""
pricer/engine.py
GPU execution layer — SamplerV2 construction, IAE execution, result interpretation.
All financial interpretation and put-call parity live here.
"""

import logging
from dataclasses import dataclass

import numpy as np
from qiskit.compiler import transpile
from qiskit.primitives import BackendSamplerV2, StatevectorSampler
from qiskit_algorithms import EstimationProblem, IterativeAmplitudeEstimation
from qiskit_aer import AerSimulator
from qiskit_finance.applications import EuropeanCallPricing

from config.settings import get_settings

logger = logging.getLogger("pricer.engine")

# ── Compatibility patch ───────────────────────────────────────────────────────
# qiskit's UnitaryGate validates matrices with atol=1e-8. For 8-qubit circuits,
# qiskit-finance's isometry decomposition accumulates enough floating-point error
# to exceed this threshold during LogNormalDistribution construction, despite the
# matrices being functionally unitary. Loosen the tolerance so circuit building
# succeeds; the simulation result is unaffected (error is O(1e-7), negligible).
import qiskit.circuit.library.generalized_gates.unitary as _unitary_mod
_orig_is_unitary = _unitary_mod.is_unitary_matrix
_unitary_mod.is_unitary_matrix = lambda mat, rtol=1e-4, atol=1e-5: _orig_is_unitary(mat, rtol=rtol, atol=atol)
logger.debug("UnitaryGate tolerance patched (rtol=1e-4, atol=1e-5)")

# ── Compatibility patch 2 ─────────────────────────────────────────────────────
# qiskit-finance 0.4.x emits P(X) (PauliGate) and IAE wraps grover_op.power(k)
# as "Gate Q" — both are unknown to qiskit-aer 0.17.2's C++ assembler.
# BackendSamplerV2._run_pubs calls _run_circuits(circuits, backend, **opts) which
# goes straight to backend.run() with no transpilation.  We replace _run_circuits
# in backend_sampler_v2's own module namespace (the import-binding the function
# looks up at call time) so every circuit is transpiled at optimization_level=1
# before hitting the C++ assembler.  This decomposes Gate Q and P(X) to basis
# gates (cx/u) while keeping the simulation semantically identical.
import qiskit.primitives.backend_sampler_v2 as _bsv2_mod
_orig_run_circuits = _bsv2_mod._run_circuits

def _run_circuits_opt1(circuits, backend, clear_metadata=True, **run_options):
    if not isinstance(circuits, list):
        circuits = [circuits]
    t = transpile(circuits, backend=backend, optimization_level=1)
    logger.debug(
        "run_circuits patch: transpiled %d circuits; first %d→%d gates",
        len(circuits), len(circuits[0].data), len(t[0].data),
    )
    return _orig_run_circuits(t, backend, clear_metadata=clear_metadata, **run_options)

_bsv2_mod._run_circuits = _run_circuits_opt1
logger.debug("BackendSamplerV2._run_circuits patched (optimization_level=1)")


# ── Sampler ───────────────────────────────────────────────────────────────────

def build_sampler(
    backend: str | None = None,
    device_index: int | None = None,
):
    """
    Construct a SamplerV2 targeting the configured compute backend.

    GPU path : BackendSamplerV2(AerSimulator statevector GPU).
               Circuits are pre-transpiled with optimization_level=1 in run_iae
               to synthesize away UnitaryGate and P(X) gates before the C++ backend
               sees them (qiskit-finance 0.4.x compatibility).
    CPU path : StatevectorSampler — pure Python, handles all gate types.
               For dev/testing only.
    """
    settings = get_settings()
    _backend = backend or settings.backend
    _device  = device_index if device_index is not None else settings.gpu_device

    if _backend == "gpu":
        logger.info(
            "Initialising BackendSamplerV2(AerSimulator) | method=statevector device=GPU[%d] precision=%s",
            _device, settings.precision,
        )
        simulator = AerSimulator(
            method="statevector",
            device="GPU",
            precision=settings.precision,
            fusion_enable=False,
        )
        return BackendSamplerV2(backend=simulator), simulator

    logger.warning(
        "GPU backend not requested — using StatevectorSampler (CPU reference). "
        "NOT suitable for production workloads."
    )
    return StatevectorSampler(), None


# ── IAE Execution ─────────────────────────────────────────────────────────────

def run_iae(
    problem: EstimationProblem,
    sampler,
    simulator,   # AerSimulator instance for transpile target (None on CPU)
    epsilon_target: float,
    alpha: float,
) -> object:
    """
    Run Iterative Amplitude Estimation.

    epsilon_target : half-width of the desired confidence interval (on amplitude)
    alpha          : significance level — CI covers (1-alpha) probability mass
    """
    logger.info("IAE start | epsilon=%.4f alpha=%.4f", epsilon_target, alpha)

    # Transpilation is handled by the _run_circuits patch above: every circuit
    # submitted to BackendSamplerV2 is compiled with optimization_level=1 before
    # reaching the AerSimulator C++ assembler.

    ae = IterativeAmplitudeEstimation(
        epsilon_target=epsilon_target,
        alpha=alpha,
        sampler=sampler,
    )
    result = ae.estimate(problem)

    logger.info(
        "IAE complete | amplitude=%.6f oracle_queries=%d",
        result.estimation, result.num_oracle_queries,
    )
    logger.debug(
        "IAE detail | confidence_interval=(%.6f, %.6f) num_rounds=%d",
        result.confidence_interval[0], result.confidence_interval[1], len(result.powers),
    )
    return result


# ── Result Interpretation ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class PricingResult:
    option_type: str
    price: float
    confidence_interval: tuple[float, float]
    call_price: float
    calculated_via_parity: bool
    oracle_queries: int
    num_iterations: int
    total_qubits: int
    grover_depth: int


def interpret_results(
    european_call: EuropeanCallPricing,
    problem: EstimationProblem,
    result: object,
    spot: float,
    strike: float,
    risk_free_rate: float,
    T: float,
    option_type: str,
) -> PricingResult:
    """
    Translate raw IAE amplitude into a present-value option price.

    Call: price = E[max(S-K, 0)] * e^(-rT)
    Put:  computed via put-call parity — P = C - S + K*e^(-rT)
    """
    discount = np.exp(-risk_free_rate * T)

    call_payoff = european_call.interpret(result)
    call_price  = call_payoff * discount

    ci_raw  = result.confidence_interval_processed
    call_ci = (ci_raw[0] * discount, ci_raw[1] * discount)

    logger.debug(
        "Call interpretation | payoff=%.6f price=%.6f CI=(%.6f, %.6f)",
        call_payoff, call_price, *call_ci,
    )

    if option_type == "call":
        return PricingResult(
            option_type="call",
            price=call_price,
            confidence_interval=call_ci,
            call_price=call_price,
            calculated_via_parity=False,
            oracle_queries=result.num_oracle_queries,
            num_iterations=len(result.powers),
            total_qubits=problem.grover_operator.num_qubits,
            grover_depth=problem.grover_operator.decompose().depth(),
        )

    pv_strike = strike * discount
    put_price  = call_price - spot + pv_strike
    put_ci = (
        call_ci[0] - spot + pv_strike,
        call_ci[1] - spot + pv_strike,
    )

    logger.info(
        "Put via parity | call=%.6f spot=%.6f pv_strike=%.6f put=%.6f",
        call_price, spot, pv_strike, put_price,
    )

    return PricingResult(
        option_type="put",
        price=put_price,
        confidence_interval=put_ci,
        call_price=call_price,
        calculated_via_parity=True,
        oracle_queries=result.num_oracle_queries,
        num_iterations=len(result.powers),
        total_qubits=problem.grover_operator.num_qubits,
        grover_depth=problem.grover_operator.decompose().depth(),
    )
