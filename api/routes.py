"""
api/routes.py
FastAPI router: POST /price  and  GET /health
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from data.resolver import fetch_spot, fetch_risk_free_rate, fetch_implied_vol
from pricer.circuit import derive_lognormal_params, build_uncertainty_model, build_estimation_problem
from pricer.engine import run_iae, interpret_results

from .models import ConfidenceInterval, HealthResponse, PricingRequest, PricingResponse

logger = logging.getLogger("api")

router = APIRouter()

# Single-worker executor: GPU state is not thread-safe; one IAE run at a time.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="iae-worker")


# ── helpers ───────────────────────────────────────────────────────────────────

def _resolve_params(req: PricingRequest) -> tuple[float, float, float, bool, str]:
    """
    Resolve spot, volatility, and risk_free_rate.
    Returns (spot, vol, rfr, market_data_used, market_data_source).
    Runs synchronously inside the thread-pool executor.
    """
    sources: list[str] = []
    market_data_used = False

    # ── Risk-free rate (resolve first — needed for vol inversion) ─────────────
    if req.risk_free_rate is not None:
        rfr = req.risk_free_rate
    else:
        rfr = fetch_risk_free_rate()
        sources.append(f"rfr=yfinance(^IRX,{rfr*100:.3f}%)")
        market_data_used = True

    # ── Spot price ────────────────────────────────────────────────────────────
    if req.spot is not None:
        spot = req.spot
    else:
        spot = fetch_spot(req.ticker)
        sources.append(f"spot=yfinance({req.ticker},{spot:.4f})")
        market_data_used = True

    # ── Implied volatility ────────────────────────────────────────────────────
    if req.volatility is not None:
        vol = req.volatility
    else:
        vol, vol_src = fetch_implied_vol(
            ticker        = req.ticker,
            strike        = req.strike,
            days_to_expiry= req.days_to_expiry,
            option_type   = req.option_type,
            spot          = spot,
            risk_free_rate= rfr,
        )
        sources.append(vol_src)
        market_data_used = True

    source_str = " | ".join(sources) if sources else None
    return spot, vol, rfr, market_data_used, source_str


def _price_sync(req: PricingRequest, sampler: Any, simulator: Any, backend_label: str) -> PricingResponse:
    """
    Full blocking pricing call — runs in the thread-pool executor.
    Resolves any auto-fetch params, then runs the quantum IAE circuit.
    """
    spot, vol, rfr, market_data_used, market_data_source = _resolve_params(req)

    T = req.days_to_expiry / 365.0

    lnp          = derive_lognormal_params(spot, rfr, vol, T)
    model        = build_uncertainty_model(req.num_qubits, lnp)
    ec, problem  = build_estimation_problem(req.num_qubits, req.strike, lnp, model)
    iae_result   = run_iae(problem, sampler, simulator, req.epsilon_target, req.alpha)
    result       = interpret_results(ec, problem, iae_result, spot, req.strike, rfr, T, req.option_type)

    return PricingResponse(
        ticker                   = req.ticker,
        option_type              = req.option_type,
        spot                     = round(spot, 4),
        strike                   = req.strike,
        days_to_expiry           = req.days_to_expiry,
        time_to_expiry_years     = round(T, 8),
        volatility               = round(vol, 6),
        risk_free_rate           = round(rfr, 6),
        market_data_used         = market_data_used,
        market_data_source       = market_data_source,
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
        "on a quantum circuit. Puts are derived from calls via put-call parity.\n\n"
        "**Auto-fetch**: omit `spot`, `volatility`, or `risk_free_rate` to have "
        "them resolved automatically from live market data. Provide `ticker` "
        "(e.g. `'AAPL'`) whenever any field is auto-fetched.\n\n"
        "| Omitted field | Source |\n"
        "|---|---|\n"
        "| `spot` | yfinance last price |\n"
        "| `volatility` | Databento OPRA NBBO mid → Black-Scholes IV inversion |\n"
        "| `risk_free_rate` | yfinance `^IRX` (13-week US T-bill) |\n\n"
        "**Latency guide** (`num_qubits=8` default):\n"
        "- GPU (T4 / L4): ~100–500 ms + ~1–2 s if auto-fetching market data\n"
        "- CPU fallback: ~87 s — use `num_qubits: 4` for fast local testing"
    ),
    tags=["Pricing"],
)
async def price_option(req: PricingRequest, request: Request) -> PricingResponse:
    sampler       = request.app.state.sampler
    simulator     = request.app.state.simulator
    backend_label = request.app.state.backend_label

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _executor, _price_sync, req, sampler, simulator, backend_label
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        logger.exception("Pricing failed | req=%s", req.model_dump())
        raise HTTPException(status_code=500, detail=f"Pricing engine error: {exc}")

    logger.info(
        "Priced | ticker=%s type=%s spot=%.2f strike=%.2f dte=%d vol=%.4f "
        "rfr=%.4f qubits=%d price=%.4f backend=%s market_data=%s",
        req.ticker, req.option_type, result.spot, req.strike, req.days_to_expiry,
        result.volatility, result.risk_free_rate, req.num_qubits,
        result.price, backend_label, result.market_data_used,
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
        backend            = request.app.state.backend_label,
        sampler_type       = type(request.app.state.sampler).__name__,
        gpu_available      = request.app.state.gpu_available,
        num_qubits_default = 8,
    )
