"""
pricer/circuit.py
Pure quantum circuit construction — no GPU, no I/O, no side effects.
Responsible for: log-normal parameter derivation, uncertainty model,
and composing the EstimationProblem handed to the engine.
"""

import logging
from dataclasses import dataclass

import numpy as np
from qiskit_algorithms import EstimationProblem
from qiskit_finance.applications import EuropeanCallPricing
from qiskit_finance.circuit.library import LogNormalDistribution

logger = logging.getLogger("pricer.circuit")


@dataclass(frozen=True)
class LogNormalParams:
    """Derived log-normal parameters for quantum encoding."""
    mu: float          # mean of log(S_T)
    sigma: float       # std dev of log(S_T)
    mean_spot: float   # E[S_T] = exp(mu + sigma^2/2)
    std_spot: float    # Std[S_T] = sqrt((exp(sigma^2)-1) * exp(2mu + sigma^2))
    low: float         # distribution lower bound
    high: float        # distribution upper bound


def derive_lognormal_params(
    spot: float,
    risk_free_rate: float,
    volatility: float,
    T: float,
    num_std: float = 3.0,
) -> LogNormalParams:
    """
    Derive log-normal distribution parameters for quantum amplitude encoding.

    The terminal stock price S_T follows:
        ln(S_T) ~ N(mu, sigma^2)
    where:
        mu    = ln(S_0) + (r - 0.5*vol^2) * T
        sigma = vol * sqrt(T)

    Bounds are set at mean ± num_std standard deviations of S_T (not ln(S_T)).
    """
    mu = np.log(spot) + (risk_free_rate - 0.5 * volatility**2) * T
    sigma = volatility * np.sqrt(T)

    # Moments of S_T (log-normal)
    mean_spot = np.exp(mu + sigma**2 / 2.0)
    std_spot = np.sqrt((np.exp(sigma**2) - 1.0) * np.exp(2.0 * mu + sigma**2))

    low = max(0.0, mean_spot - num_std * std_spot)
    high = mean_spot + num_std * std_spot

    params = LogNormalParams(
        mu=mu,
        sigma=sigma,
        mean_spot=mean_spot,
        std_spot=std_spot,
        low=low,
        high=high,
    )

    logger.debug(
        "LogNormal params | spot=%.4f T=%.4f mu=%.4f sigma=%.4f "
        "mean=%.4f std=%.4f bounds=(%.4f, %.4f)",
        spot, T, mu, sigma, mean_spot, std_spot, low, high,
    )
    return params


def build_uncertainty_model(
    num_qubits: int,
    params: LogNormalParams,
) -> LogNormalDistribution:
    """
    Encode the log-normal distribution of S_T into qubit amplitudes.

    num_qubits determines the discretisation resolution:
      2^num_qubits bins across [low, high].
    Higher num_qubits → better accuracy, deeper circuit, more GPU memory.
    """
    model = LogNormalDistribution(
        num_qubits=[num_qubits],
        mu=[params.mu],
        sigma=[[params.sigma**2]],   # 1×1 covariance matrix — qiskit-finance 0.4.x API
        bounds=[(params.low, params.high)],
    )

    depth = model.decompose().depth()
    logger.info(
        "Uncertainty model built | num_qubits=%d bins=%d depth=%d "
        "bounds=(%.4f, %.4f)",
        num_qubits, 2**num_qubits, depth, params.low, params.high,
    )
    return model


def build_estimation_problem(
    num_qubits: int,
    strike: float,
    params: LogNormalParams,
    uncertainty_model: LogNormalDistribution,
    rescaling_factor: float = 0.25,
) -> tuple[EuropeanCallPricing, EstimationProblem]:
    """
    Compose the EuropeanCallPricing payoff circuit and wrap it into an
    EstimationProblem for IterativeAmplitudeEstimation.

    rescaling_factor maps max_payoff into the valid rotation range [0, pi/2].
    Default 0.25 works well for ATM options. For deep ITM, consider lowering.

    Returns:
        european_call : needed later for result interpretation
        problem       : passed directly to IAE
    """
    european_call = EuropeanCallPricing(
        num_state_qubits=num_qubits,
        strike_price=strike,
        rescaling_factor=rescaling_factor,
        bounds=(params.low, params.high),
        uncertainty_model=uncertainty_model,
    )

    problem = european_call.to_estimation_problem()
    grover = problem.grover_operator

    total_qubits = grover.num_qubits
    grover_depth = grover.decompose().depth()

    logger.info(
        "EstimationProblem assembled | strike=%.4f total_qubits=%d "
        "grover_depth=%d rescaling_factor=%.3f",
        strike, total_qubits, grover_depth, rescaling_factor,
    )

    if total_qubits > 20:
        logger.warning(
            "Circuit width=%d exceeds 20 qubits — tensor_network memory usage "
            "will grow exponentially. Consider reducing num_qubits.",
            total_qubits,
        )

    return european_call, problem
