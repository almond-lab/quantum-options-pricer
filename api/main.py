"""
api/main.py
FastAPI application entry point.

Scalar interactive API docs served at /docs  (replaces default Swagger UI).
OpenAPI JSON available at /openapi.json.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from scalar_fastapi import get_scalar_api_reference, Theme

from config.settings import get_settings, setup_logging
from pricer.engine import build_sampler

from .routes import router

setup_logging()
logger = logging.getLogger("api")


# ── Lifespan: initialise GPU sampler once at startup ─────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    sampler, simulator = build_sampler(settings.backend, settings.gpu_device)

    # Detect whether we actually got a GPU backend
    try:
        from qiskit_aer import AerSimulator
        gpu_available = "GPU" in AerSimulator().available_devices()
    except Exception:
        gpu_available = False

    sampler_name = type(sampler).__name__
    backend_label = "gpu" if gpu_available else "cpu"

    app.state.sampler       = sampler
    app.state.simulator     = simulator   # AerSimulator instance for transpile target
    app.state.backend_label = backend_label
    app.state.gpu_available = gpu_available

    logger.info(
        "Sampler ready | type=%s backend=%s gpu=%s",
        sampler_name, backend_label, gpu_available,
    )

    yield

    logger.info("Shutting down")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Quantum Options Pricer",
    version="0.1.0",
    summary="European vanilla option pricing via Iterative Amplitude Estimation (IAE) on GPU",
    description=(
        "Prices European **Call** and **Put** options using a quantum amplitude "
        "estimation circuit backed by NVIDIA cuTensorNet on GPU.\n\n"
        "- **Call**: priced directly by the quantum circuit\n"
        "- **Put**: derived via put-call parity (no extra circuit cost)\n"
        "- **Market data**: live NBBO quotes from Databento OPRA.PILLAR\n\n"
        "All responses include a confidence interval and full circuit metadata."
    ),
    contact={
        "name":  "almond-lab",
        "email": "dev@almond.org",
        "url":   "https://github.com/almond-lab/quantum-options-pricer",
    },
    license_info={
        "name": "MIT",
    },
    docs_url=None,      # disable default Swagger UI
    redoc_url=None,     # disable ReDoc
    lifespan=lifespan,
)

app.include_router(router)


# ── Scalar API reference ──────────────────────────────────────────────────────

@app.get("/docs", include_in_schema=False)
async def scalar_docs():
    return get_scalar_api_reference(
        openapi_url="/openapi.json",
        title="Quantum Options Pricer",
        theme=Theme.PURPLE,
        default_open_all_tags=True,
    )
