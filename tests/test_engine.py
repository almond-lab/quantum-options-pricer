"""
tests/test_engine.py
Unit + integration tests for pricer/engine.py
All tests run on CPU (BACKEND=cpu set in conftest.py).
Integration test prices an ATM Call and validates against Black-Scholes.
"""

import os
import numpy as np
import pytest

from pricer.circuit import (
    derive_lognormal_params,
    build_uncertainty_model,
    build_estimation_problem,
)
from pricer.engine import (
    PricingResult,
    build_sampler,
    interpret_results,
    run_iae,
)


# ── build_sampler ─────────────────────────────────────────────────────────────

class TestBuildSampler:

    def test_cpu_sampler_builds(self):
        from qiskit.primitives import StatevectorSampler
        sampler = build_sampler(backend="cpu")
        assert isinstance(sampler, StatevectorSampler)

    def test_explicit_device_index(self):
        sampler = build_sampler(backend="cpu", device_index=0)
        assert sampler is not None

    def test_settings_default_used_when_no_args(self):
        """With BACKEND=cpu in env, no args should still return a valid sampler."""
        sampler = build_sampler()
        assert sampler is not None


# ── interpret_results (unit — no IAE run) ─────────────────────────────────────

class TestInterpretResults:
    """
    Tests for interpret_results() using a mock IAE result object.
    Validates put-call parity math and PricingResult fields without running IAE.
    """

    class _MockIAEResult:
        """
        Minimal IAE result stub for math validation.
        estimation_processed = post-processed amplitude (payoff value in financial units).
        This is what european_call.interpret() reads directly.
        """
        def __init__(self, estimation_processed, ci_processed, oracle_queries=50, num_iterations=5):
            self.estimation = estimation_processed        # raw amplitude (same value for mock)
            self.estimation_processed = estimation_processed  # what interpret() reads
            self.confidence_interval_processed = ci_processed
            self.confidence_interval = ci_processed
            self.num_oracle_queries = oracle_queries
            self.powers = list(range(num_iterations))  # IAEResult uses powers list; len = rounds

    @pytest.fixture
    def circuit_pieces(self, atm_params):
        params = derive_lognormal_params(
            spot=atm_params["spot"],
            risk_free_rate=atm_params["risk_free_rate"],
            volatility=atm_params["volatility"],
            T=atm_params["T"],
        )
        model = build_uncertainty_model(num_qubits=3, params=params)
        european_call, problem = build_estimation_problem(3, atm_params["strike"], params, model)
        return european_call, problem, params, atm_params

    def test_call_returns_pricing_result(self, circuit_pieces):
        european_call, problem, params, p = circuit_pieces
        mock_result = self._MockIAEResult(estimation_processed=5.0, ci_processed=(0.08, 0.12))
        result = interpret_results(
            european_call, problem, mock_result,
            spot=p["spot"], strike=p["strike"],
            risk_free_rate=p["risk_free_rate"], T=p["T"], option_type="call",
        )
        assert isinstance(result, PricingResult)

    def test_call_parity_flag_false(self, circuit_pieces):
        european_call, problem, params, p = circuit_pieces
        mock_result = self._MockIAEResult(estimation_processed=5.0, ci_processed=(0.08, 0.12))
        result = interpret_results(
            european_call, problem, mock_result,
            spot=p["spot"], strike=p["strike"],
            risk_free_rate=p["risk_free_rate"], T=p["T"], option_type="call",
        )
        assert result.calculated_via_parity is False
        assert result.option_type == "call"

    def test_put_parity_flag_true(self, circuit_pieces):
        european_call, problem, params, p = circuit_pieces
        mock_result = self._MockIAEResult(estimation_processed=5.0, ci_processed=(0.08, 0.12))
        result = interpret_results(
            european_call, problem, mock_result,
            spot=p["spot"], strike=p["strike"],
            risk_free_rate=p["risk_free_rate"], T=p["T"], option_type="put",
        )
        assert result.calculated_via_parity is True
        assert result.option_type == "put"

    def test_put_call_parity_math(self, circuit_pieces, bs):
        """
        P = C - S + K*e(-rT)
        Verify the put price in PricingResult matches this formula applied to call_price.
        """
        european_call, problem, params, p = circuit_pieces
        mock_result = self._MockIAEResult(estimation_processed=5.0, ci_processed=(0.08, 0.12))

        call_result = interpret_results(
            european_call, problem, mock_result,
            spot=p["spot"], strike=p["strike"],
            risk_free_rate=p["risk_free_rate"], T=p["T"], option_type="call",
        )
        put_result = interpret_results(
            european_call, problem, mock_result,
            spot=p["spot"], strike=p["strike"],
            risk_free_rate=p["risk_free_rate"], T=p["T"], option_type="put",
        )

        expected_put = call_result.price - p["spot"] + p["strike"] * np.exp(-p["risk_free_rate"] * p["T"])
        assert abs(put_result.price - expected_put) < 1e-10

    def test_ci_ordering(self, circuit_pieces):
        """Confidence interval lower bound must be <= upper bound."""
        european_call, problem, params, p = circuit_pieces
        mock_result = self._MockIAEResult(estimation_processed=5.0, ci_processed=(0.08, 0.12))
        result = interpret_results(
            european_call, problem, mock_result,
            spot=p["spot"], strike=p["strike"],
            risk_free_rate=p["risk_free_rate"], T=p["T"], option_type="call",
        )
        assert result.confidence_interval[0] <= result.confidence_interval[1]

    def test_call_price_positive(self, circuit_pieces):
        european_call, problem, params, p = circuit_pieces
        mock_result = self._MockIAEResult(estimation_processed=5.0, ci_processed=(0.08, 0.12))
        result = interpret_results(
            european_call, problem, mock_result,
            spot=p["spot"], strike=p["strike"],
            risk_free_rate=p["risk_free_rate"], T=p["T"], option_type="call",
        )
        assert result.price >= 0.0

    def test_circuit_metrics_populated(self, circuit_pieces):
        european_call, problem, params, p = circuit_pieces
        mock_result = self._MockIAEResult(estimation_processed=5.0, ci_processed=(0.08, 0.12), oracle_queries=42, num_iterations=4)
        result = interpret_results(
            european_call, problem, mock_result,
            spot=p["spot"], strike=p["strike"],
            risk_free_rate=p["risk_free_rate"], T=p["T"], option_type="call",
        )
        assert result.oracle_queries == 42
        assert result.num_iterations == 4
        assert result.total_qubits > 0
        assert result.grover_depth > 0


# ── End-to-end integration test (CPU, 3 qubits) ───────────────────────────────

class TestEndToEndPricing:
    """
    Runs the full pipeline on CPU: circuit → sampler → IAE → interpret.
    Uses 3 qubits (coarse) — validates result is in the right ballpark vs BS.
    Tolerance is wide (±40%) because 3-qubit discretisation is intentionally coarse.
    """

    @pytest.mark.slow
    def test_atm_call_price_in_bs_ballpark(self, atm_params, bs):
        p = atm_params
        bs_price = bs(p["spot"], p["strike"], p["T"], p["risk_free_rate"], p["volatility"], "call")

        params = derive_lognormal_params(p["spot"], p["risk_free_rate"], p["volatility"], p["T"])
        model = build_uncertainty_model(num_qubits=3, params=params)
        european_call, problem = build_estimation_problem(3, p["strike"], params, model)

        sampler = build_sampler(backend="cpu")
        iae_result = run_iae(
            problem, sampler,
            epsilon_target=0.05,   # loose precision — fast on CPU
            alpha=0.05,
        )
        result = interpret_results(
            european_call, problem, iae_result,
            spot=p["spot"], strike=p["strike"],
            risk_free_rate=p["risk_free_rate"], T=p["T"], option_type="call",
        )

        print(f"\n  BS call price   : {bs_price:.4f}")
        print(f"  Quantum call    : {result.price:.4f}")
        print(f"  CI              : ({result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f})")
        print(f"  Oracle queries  : {result.oracle_queries}")

        # 3-qubit coarse grid — allow ±40% relative error
        assert abs(result.price - bs_price) / bs_price < 0.40, (
            f"Quantum price {result.price:.4f} too far from BS {bs_price:.4f}"
        )

    @pytest.mark.slow
    def test_atm_put_via_parity(self, atm_params, bs):
        p = atm_params
        bs_put = bs(p["spot"], p["strike"], p["T"], p["risk_free_rate"], p["volatility"], "put")

        params = derive_lognormal_params(p["spot"], p["risk_free_rate"], p["volatility"], p["T"])
        model = build_uncertainty_model(num_qubits=3, params=params)
        european_call, problem = build_estimation_problem(3, p["strike"], params, model)

        sampler = build_sampler(backend="cpu")
        iae_result = run_iae(problem, sampler, epsilon_target=0.05, alpha=0.05)
        result = interpret_results(
            european_call, problem, iae_result,
            spot=p["spot"], strike=p["strike"],
            risk_free_rate=p["risk_free_rate"], T=p["T"], option_type="put",
        )

        print(f"\n  BS put price    : {bs_put:.4f}")
        print(f"  Quantum put     : {result.price:.4f}")
        print(f"  Via parity      : {result.calculated_via_parity}")

        assert result.calculated_via_parity is True
        assert abs(result.price - bs_put) / max(bs_put, 0.01) < 0.40
