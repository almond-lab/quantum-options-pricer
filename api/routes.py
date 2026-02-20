"""
api/routes.py
FastAPI router: POST /price  and  GET /health
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from pricer.circuit import derive_lognormal_params, build_uncertainty_model, build_estimation_problem
from pricer.engine import run_iae, interpret_results

from .models import ConfidenceInterval, HealthResponse, PricingRequest, PricingResponse

logger = logging.getLogger("api")

router = APIRouter()

# Single-worker executor: GPU state is not thread-safe; one IAE run at a time.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="iae-worker")


# ── helpers ───────────────────────────────────────────────────────────────────

def _price_sync(req: PricingRequest, sampler: Any, backend_label: str) -> PricingResponse:
    """
    Blocking pricing call — runs in the thread-pool executor so it doesn't
    stall the uvicorn event loop during the IAE computation.
    """
    T = req.days_to_expiry / 365.0

    lnp   = derive_lognormal_params(req.spot, req.risk_free_rate, req.volatility, T)
    model = build_uncertainty_model(req.num_qubits, lnp)
    ec, problem = build_estimation_problem(req.num_qubits, req.strike, lnp, model)

    iae_result = run_iae(problem, sampler, req.epsilon_target, req.alpha)
    result     = interpret_results(ec, problem, iae_result, req.spot, req.strike,
                                   req.risk_free_rate, T, req.option_type)

    return PricingResponse(
        option_type              = req.option_type,
        spot                     = req.spot,
        strike                   = req.strike,
        days_to_expiry           = req.days_to_expiry,
        time_to_expiry_years     = T,
        volatility               = req.volatility,
        risk_free_rate           = req.risk_free_rate,
        price                    = round(result.price, 6),
        confidence_interval      = ConfidenceInterval(
                                       lower=round(result.confidence_interval[0], 6),
                                       upper=round(result.confidence_interval[1], 6),
                                   ),
        calculated_via_parity    = result.calculated_via_parity,
        num_qubits               = req.num_qubits,
        price_bins               = 2 ** req.num_qubits,
        total_circuit_qubits     = result.total_qubits,
        grover_depth             = result.grover_depth,
        oracle_queries           = result.oracle_queries,
        iae_rounds               = result.num_iterations,
        epsilon_target           = req.epsilon_target,
        alpha                    = req.alpha,
        backend                  = backend_label,
    )


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/price",
    response_model=PricingResponse,
    summary="Price a European vanilla option",
    description=(
        "Prices a European Call or Put using Iterative Amplitude Estimation (IAE) "
        "on a quantum circuit. The call payoff is computed via the quantum circuit; "
        "puts are derived from the call price using put-call parity.\n\n"
        "**Latency guide** (8 qubits default):\n"
        "- GPU (T4 / L4): ~100–500 ms\n"
        "- CPU fallback (dev): ~87 s — use `num_qubits: 4` for fast local testing"
    ),
    tags=["Pricing"],
)
async def price_option(req: PricingRequest, request: Request) -> PricingResponse:
    sampler       = request.app.state.sampler
    backend_label = request.app.state.backend_label

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _executor, _price_sync, req, sampler, backend_label
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Pricing failed | req=%s", req.model_dump())
        raise HTTPException(status_code=500, detail=f"Pricing engine error: {exc}")

    logger.info(
        "Priced | type=%s spot=%.2f strike=%.2f dte=%d vol=%.3f "
        "qubits=%d price=%.4f backend=%s",
        req.option_type, req.spot, req.strike, req.days_to_expiry,
        req.volatility, req.num_qubits, result.price, backend_label,
    )
    return result


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health and backend status",
    tags=["Meta"],
)
async def health(request: Request) -> HealthResponse:
    return HealthResponse(
        backend          = request.app.state.backend_label,
        sampler_type     = type(request.app.state.sampler).__name__,
        gpu_available    = request.app.state.gpu_available,
        num_qubits_default = 8,
    )
