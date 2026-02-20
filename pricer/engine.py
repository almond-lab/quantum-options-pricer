"""
pricer/engine.py
GPU execution layer — SamplerV2 construction, IAE execution, result interpretation.
All financial interpretation and put-call parity live here.
"""

import logging
from dataclasses import dataclass

import numpy as np
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import EstimationProblem, IterativeAmplitudeEstimation
from qiskit_finance.applications import EuropeanCallPricing

from config.settings import get_settings

logger = logging.getLogger("pricer.engine")


# ── Sampler ──────────────────────────────────────────────────────────────────

def build_sampler(
    backend: str | None = None,
    device_index: int | None = None,
):
    """
    Construct a SamplerV2 targeting the configured compute backend.

    GPU path  : AerSamplerV2 with tensor_network via NVIDIA cuTensorNet (T4/L4)
    CPU path  : qiskit.primitives.StatevectorSampler — reference implementation,
                handles all circuit types including deprecated finance circuits.
                For dev/testing only.
    """
    settings = get_settings()
    _backend = backend or settings.backend
    _device  = device_index if device_index is not None else settings.gpu_device

    if _backend == "gpu":
        logger.info(
            "Initialising AerSamplerV2 | method=tensor_network device=GPU[%d] precision=%s",
            _device, settings.precision,
        )
        sampler = AerSamplerV2(
            options={
                "backend_options": {
                    "method": "tensor_network",
                    "device": "GPU",
                    "device_index": _device,
                    "precision": settings.precision,
                    "cuStateVec_enable": False,   # use cuTensorNet, not cuStateVec
                }
            }
        )
    else:
        logger.warning(
            "GPU backend not requested — using StatevectorSampler (CPU reference). "
            "NOT suitable for production workloads."
        )
        # Use qiskit.primitives.StatevectorSampler for CPU fallback.
        # AerSamplerV2 on CPU rejects deprecated gate types (P(X)) emitted by
        # qiskit-finance 0.4.x circuits. The reference sampler handles all types.
        sampler = StatevectorSampler()

    return sampler


# ── IAE Execution ────────────────────────────────────────────────────────────

def run_iae(
    problem: EstimationProblem,
    sampler,   # AerSamplerV2 (GPU) or StatevectorSampler (CPU)
    epsilon_target: float,
    alpha: float,
) -> object:
    """
    Run Iterative Amplitude Estimation.

    epsilon_target : half-width of the desired confidence interval (on amplitude)
    alpha          : significance level — CI covers (1-alpha) probability mass

    Tighter epsilon → more Grover iterations → longer runtime.
    Guard rails are enforced upstream at the Pydantic request model.
    """
    logger.info(
        "IAE start | epsilon=%.4f alpha=%.4f",
        epsilon_target, alpha,
    )

    ae = IterativeAmplitudeEstimation(
        epsilon_target=epsilon_target,
        alpha=alpha,
        sampler=sampler,
    )

    result = ae.estimate(problem)

    logger.info(
        "IAE complete | amplitude=%.6f oracle_queries=%d",
        result.estimation,
        result.num_oracle_queries,
    )
    logger.debug(
        "IAE detail | confidence_interval=(%.6f, %.6f) num_rounds=%d",
        result.confidence_interval[0],
        result.confidence_interval[1],
        len(result.powers),   # IAEResult has no num_iterations; powers list length = rounds
    )

    return result


# ── Result Interpretation ────────────────────────────────────────────────────

@dataclass(frozen=True)
class PricingResult:
    option_type: str
    price: float
    confidence_interval: tuple[float, float]
    call_price: float                # always present — underlying Call circuit value
    calculated_via_parity: bool      # True when option_type="put"
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
          'calculated_via_parity' flag is set True for full provenance.

    Confidence intervals are post-processed through the payoff function
    then discounted — giving financially meaningful bounds, not raw amplitudes.
    """
    discount = np.exp(-risk_free_rate * T)

    # Expected payoff at maturity (undiscounted), interpreted via the pricing circuit
    call_payoff = european_call.interpret(result)
    call_price  = call_payoff * discount

    # Confidence interval — post-processed via the estimation problem's
    # post_processing function, then discounted
    ci_raw = result.confidence_interval_processed     # payoff-space CI
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
            num_iterations=len(result.powers),  # IAEResult has no num_iterations attr
            total_qubits=problem.grover_operator.num_qubits,
            grover_depth=problem.grover_operator.decompose().depth(),
        )

    # ── Put via parity ────────────────────────────────────────────────────
    # P = C - S + K * e^(-rT)
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
        num_iterations=len(result.powers),  # IAEResult has no num_iterations attr
        total_qubits=problem.grover_operator.num_qubits,
        grover_depth=problem.grover_operator.decompose().depth(),
    )
