"""
api/models.py
Pydantic request / response models for the pricing API.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


class PricingRequest(BaseModel):
    """Input parameters for a single European vanilla option pricing request."""

    spot: float = Field(
        ..., gt=0,
        description="Current underlying price (USD)",
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
    volatility: float = Field(
        ..., gt=0,
        description="Annualised implied volatility, e.g. 0.30 = 30%",
        examples=[0.30],
    )
    risk_free_rate: float = Field(
        ..., ge=0,
        description="Annualised risk-free rate, e.g. 0.053 = 5.3%",
        examples=[0.053],
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
        default=0.01, ge=0.005, le=0.1,
        description="IAE precision target ε. Lower = more accurate, more iterations.",
        examples=[0.01],
    )
    alpha: float = Field(
        default=0.05, ge=0.01, le=0.1,
        description="Confidence level: CI = 1 - alpha (default 95%)",
        examples=[0.05],
    )

    @model_validator(mode="after")
    def strike_within_reason(self) -> "PricingRequest":
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

    # ── Inputs echoed back ────────────────────────────────────────────────────
    option_type: str
    spot: float
    strike: float
    days_to_expiry: int
    time_to_expiry_years: float = Field(description="T in years (days / 365)")
    volatility: float
    risk_free_rate: float

    # ── Quantum result ────────────────────────────────────────────────────────
    price: float = Field(description="Quantum option price (USD)")
    confidence_interval: ConfidenceInterval
    calculated_via_parity: bool = Field(
        description="True if this is a Put priced via put-call parity (not a separate circuit)"
    )

    # ── Circuit metadata ──────────────────────────────────────────────────────
    num_qubits: int = Field(description="Uncertainty qubits used")
    price_bins: int = Field(description="2^num_qubits price levels in the grid")
    total_circuit_qubits: int = Field(description="Uncertainty + ancilla qubits")
    grover_depth: int = Field(description="Decomposed Grover operator depth")
    oracle_queries: int = Field(description="Total oracle calls made by IAE")
    iae_rounds: int = Field(description="Number of IAE iterations")
    epsilon_target: float
    alpha: float
    backend: str = Field(description="Sampler backend used: 'gpu' or 'cpu'")


class HealthResponse(BaseModel):
    """Service health and backend status."""

    status: Literal["ok"] = "ok"
    backend: str = Field(description="Active backend: 'gpu' or 'cpu'")
    sampler_type: str = Field(description="Fully qualified sampler class name")
    gpu_available: bool
    num_qubits_default: int = Field(description="Default num_qubits for requests")


class ErrorResponse(BaseModel):
    detail: str
