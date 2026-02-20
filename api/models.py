"""
api/models.py
Pydantic request / response models for the pricing API.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


class PricingRequest(BaseModel):
    """
    Input parameters for a single European vanilla option pricing request.

    **Auto-fetch behaviour**
    If `spot`, `volatility`, or `risk_free_rate` are omitted, they are
    resolved automatically from live market data:

    | Field            | Auto-source                                    |
    |------------------|------------------------------------------------|
    | `spot`           | yfinance — last traded price for `ticker`      |
    | `volatility`     | Databento OPRA NBBO mid → BS IV inversion      |
    | `risk_free_rate` | yfinance `^IRX` (13-week US T-bill yield)      |

    `ticker` is **required** whenever `spot` or `volatility` is omitted.
    """

    ticker: Optional[str] = Field(
        default=None,
        description=(
            "Equity root ticker (e.g. 'AAPL', 'SPY'). "
            "Required when spot or volatility is not provided. "
            "Do not append exchange suffixes — the service handles OPRA formatting internally."
        ),
        examples=["AAPL"],
    )
    spot: Optional[float] = Field(
        default=None,
        gt=0,
        description="Current underlying price (USD). Omit to fetch live from yfinance.",
        examples=[260.58],
    )
    strike: float = Field(
        ..., gt=0,
        description="Option strike price (USD)",
        examples=[260.0],
    )
    days_to_expiry: int = Field(
        ..., gt=0,
        description="Calendar days until expiration",
        examples=[30],
    )
    volatility: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "Annualised implied volatility, e.g. 0.30 = 30%. "
            "Omit to back-calculate from Databento OPRA NBBO mid via BS inversion."
        ),
        examples=[0.30],
    )
    risk_free_rate: Optional[float] = Field(
        default=None,
        ge=0,
        description=(
            "Annualised risk-free rate, e.g. 0.036 = 3.6%. "
            "Omit to fetch live from yfinance ^IRX (13-week US T-bill)."
        ),
        examples=[0.036],
    )
    option_type: Literal["call", "put"] = Field(
        default="call",
        description="Option type. Puts are priced via put-call parity.",
    )
    num_qubits: int = Field(
        default=8, ge=3, le=16,
        description=(
            "Uncertainty qubits controlling price-grid resolution. "
            "8 = 256 bins, <1% vs Black-Scholes on GPU (production default). "
            "Use 4–6 for fast dev/test on CPU."
        ),
        examples=[8],
    )
    epsilon_target: float = Field(
        default=0.05, ge=0.005, le=0.1,
        description=(
            "IAE precision target ε. Lower = more accurate, more oracle queries. "
            "Default 0.05 completes in 1–2 rounds on GPU; 0.01 requires ~8 rounds "
            "and may increase circuit depth significantly."
        ),
        examples=[0.05],
    )
    alpha: float = Field(
        default=0.05, ge=0.01, le=0.1,
        description="Confidence level: CI = 1 - alpha (default 95%)",
        examples=[0.05],
    )

    @model_validator(mode="after")
    def validate_auto_fetch_requirements(self) -> "PricingRequest":
        needs_ticker = self.spot is None or self.volatility is None
        if needs_ticker and not self.ticker:
            missing = []
            if self.spot is None:
                missing.append("spot")
            if self.volatility is None:
                missing.append("volatility")
            raise ValueError(
                f"`ticker` is required when {' and '.join(missing)} "
                f"{'are' if len(missing) > 1 else 'is'} not provided."
            )

        if self.ticker:
            self.ticker = self.ticker.upper().strip()

        if self.spot is not None and self.strike:
            ratio = self.strike / self.spot
            if ratio < 0.1 or ratio > 10.0:
                raise ValueError(
                    f"strike/spot ratio {ratio:.2f} is extreme (must be 0.1–10). "
                    "Deep OTM options fall outside the quantum circuit's price domain."
                )
        return self


class ConfidenceInterval(BaseModel):
    lower: float = Field(description="Lower bound of the IAE confidence interval (USD)")
    upper: float = Field(description="Upper bound of the IAE confidence interval (USD)")


class PricingResponse(BaseModel):
    """Quantum pricing result for a European vanilla option."""

    # ── Inputs echoed back (resolved values) ─────────────────────────────────
    ticker: Optional[str] = Field(default=None, description="Ticker used for market data fetch, if any")
    option_type: str
    spot: float = Field(description="Spot price used for pricing (resolved if auto-fetched)")
    strike: float
    days_to_expiry: int
    time_to_expiry_years: float = Field(description="T in years (days / 365)")
    volatility: float = Field(description="Implied volatility used (resolved if auto-fetched)")
    risk_free_rate: float = Field(description="Risk-free rate used (resolved if auto-fetched)")

    # ── Market data provenance ────────────────────────────────────────────────
    market_data_used: bool = Field(
        description="True if any parameter was auto-fetched from a live data source"
    )
    market_data_source: Optional[str] = Field(
        default=None,
        description="Human-readable description of which sources were used for auto-fetched fields",
    )

    # ── Quantum result ────────────────────────────────────────────────────────
    price: float = Field(description="Quantum option price (USD)")
    confidence_interval: ConfidenceInterval
    calculated_via_parity: bool = Field(
        description="True if this is a Put priced via put-call parity (not a separate circuit)"
    )

    # ── Circuit metadata ──────────────────────────────────────────────────────
    num_qubits: int
    price_bins: int = Field(description="2^num_qubits price levels in the grid")
    total_circuit_qubits: int
    grover_depth: int
    oracle_queries: int
    iae_rounds: int
    epsilon_target: float
    alpha: float
    backend: str = Field(description="Sampler backend used: 'gpu' or 'cpu'")


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    backend: str
    sampler_type: str
    gpu_available: bool
    num_qubits_default: int


class ErrorResponse(BaseModel):
    detail: str
