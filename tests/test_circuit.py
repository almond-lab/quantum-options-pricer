"""
tests/test_circuit.py
Unit tests for pricer/circuit.py
Tests are pure math + circuit construction — no GPU, no network calls.
"""

import numpy as np
import pytest
from qiskit_finance.circuit.library import LogNormalDistribution
from qiskit_algorithms import EstimationProblem

from pricer.circuit import (
    LogNormalParams,
    derive_lognormal_params,
    build_uncertainty_model,
    build_estimation_problem,
)


def _lnparams(p):
    """Helper: call derive_lognormal_params from atm_params fixture dict."""
    return derive_lognormal_params(
        spot=p["spot"],
        risk_free_rate=p["risk_free_rate"],
        volatility=p["volatility"],
        T=p["T"],
    )


# ── derive_lognormal_params ───────────────────────────────────────────────────

class TestDeriveLogNormalParams:

    def test_returns_dataclass(self, atm_params):
        assert isinstance(_lnparams(atm_params), LogNormalParams)

    def test_mu_formula(self, atm_params):
        """mu = ln(S0) + (r - 0.5*vol^2)*T"""
        p = atm_params
        result = _lnparams(p)
        expected = np.log(p["spot"]) + (p["risk_free_rate"] - 0.5 * p["volatility"]**2) * p["T"]
        assert abs(result.mu - expected) < 1e-10

    def test_sigma_formula(self, atm_params):
        """sigma = vol * sqrt(T)"""
        p = atm_params
        result = _lnparams(p)
        assert abs(result.sigma - p["volatility"] * np.sqrt(p["T"])) < 1e-10

    def test_mean_spot_equals_forward_price(self, atm_params):
        """E[S_T] = S0 * exp(r*T) under the risk-neutral measure."""
        p = atm_params
        result = _lnparams(p)
        expected = p["spot"] * np.exp(p["risk_free_rate"] * p["T"])
        assert abs(result.mean_spot - expected) / expected < 1e-6

    def test_bounds_positive(self, atm_params):
        """Lower bound must never be negative (stock price floor = 0)."""
        assert _lnparams(atm_params).low >= 0.0

    def test_bounds_ordering(self, atm_params):
        r = _lnparams(atm_params)
        assert r.low < r.mean_spot < r.high

    def test_std_spot_formula(self, atm_params):
        """Std[S_T] = sqrt((exp(sigma^2)-1) * exp(2mu+sigma^2)) — correct log-normal formula."""
        r = _lnparams(atm_params)
        expected = np.sqrt((np.exp(r.sigma**2) - 1.0) * np.exp(2.0 * r.mu + r.sigma**2))
        assert abs(r.std_spot - expected) < 1e-10

    def test_high_vol_low_bound_clamps_at_zero(self):
        """Very high vol can push low below 0 — must clamp."""
        result = derive_lognormal_params(spot=10.0, risk_free_rate=0.0, volatility=3.0, T=0.01)
        assert result.low >= 0.0

    def test_short_maturity(self):
        """1-day option (T = 1/365)."""
        result = derive_lognormal_params(spot=100.0, risk_free_rate=0.05, volatility=0.2, T=1/365)
        assert result.low >= 0.0
        assert result.high > result.low

    def test_various_spot_prices(self):
        for spot in [50.0, 100.0, 250.0, 1000.0]:
            r = derive_lognormal_params(spot=spot, risk_free_rate=0.05, volatility=0.2, T=1.0)
            assert r.low >= 0.0
            assert r.high > 0.0


# ── build_uncertainty_model ───────────────────────────────────────────────────

class TestBuildUncertaintyModel:

    def test_returns_lognormal_distribution(self, atm_params):
        model = build_uncertainty_model(num_qubits=3, params=_lnparams(atm_params))
        assert isinstance(model, LogNormalDistribution)

    def test_qubit_count(self, atm_params):
        params = _lnparams(atm_params)
        for n in [3, 4, 5]:
            model = build_uncertainty_model(num_qubits=n, params=params)
            assert model.num_qubits == n

    def test_probabilities_sum_to_one(self, atm_params):
        """Amplitude-encoded distribution must be normalised."""
        model = build_uncertainty_model(num_qubits=3, params=_lnparams(atm_params))
        assert abs(np.sum(model.probabilities) - 1.0) < 1e-6

    def test_higher_qubits_increases_depth(self, atm_params):
        """More qubits → deeper circuit (finer discretisation)."""
        params = _lnparams(atm_params)
        depth_3 = build_uncertainty_model(3, params).decompose().depth()
        depth_5 = build_uncertainty_model(5, params).decompose().depth()
        assert depth_5 > depth_3


# ── build_estimation_problem ──────────────────────────────────────────────────

class TestBuildEstimationProblem:

    @pytest.fixture
    def built_circuit(self, atm_params):
        params = _lnparams(atm_params)
        model = build_uncertainty_model(num_qubits=3, params=params)
        european_call, problem = build_estimation_problem(
            num_qubits=3,
            strike=atm_params["strike"],
            params=params,
            uncertainty_model=model,
        )
        return european_call, problem, params

    def test_returns_tuple_of_two(self, atm_params):
        params = _lnparams(atm_params)
        model = build_uncertainty_model(3, params)
        result = build_estimation_problem(3, atm_params["strike"], params, model)
        assert isinstance(result, tuple) and len(result) == 2

    def test_estimation_problem_type(self, built_circuit):
        _, problem, _ = built_circuit
        assert isinstance(problem, EstimationProblem)

    def test_grover_operator_exists(self, built_circuit):
        _, problem, _ = built_circuit
        assert problem.grover_operator is not None

    def test_circuit_wider_than_uncertainty_qubits(self, built_circuit):
        """Total qubits > num_qubits — ancillas needed for comparator."""
        _, problem, _ = built_circuit
        assert problem.grover_operator.num_qubits > 3

    def test_deep_otm_circuit_still_builds(self, atm_params):
        """
        Strike near the high bound of the distribution — deep OTM but still within domain.
        Strike must stay inside [low, high]; outside raises ValueError by design.
        """
        params = _lnparams(atm_params)
        model = build_uncertainty_model(3, params)
        deep_otm_strike = params.high * 0.95   # 5% below ceiling — still in-domain
        _, problem = build_estimation_problem(
            num_qubits=3,
            strike=deep_otm_strike,
            params=params,
            uncertainty_model=model,
        )
        assert problem.grover_operator is not None

    def test_post_processing_callable(self, built_circuit):
        """EstimationProblem must carry a post_processing fn for CI interpretation."""
        _, problem, _ = built_circuit
        assert callable(problem.post_processing)
