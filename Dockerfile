# =============================================================================
# Quantum Options Pricer — Dockerfile
# Base : NVIDIA CUDA 12.4.1 + cuDNN 9 (Ubuntu 22.04)
# GPU  : T4 (sm_75) default | swap ARG CUDA_ARCH=89 for L4 (Ada Lovelace)
# =============================================================================

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

ARG CUDA_ARCH=75
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    cmake \
    ninja-build \
    libopenblas-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Pin python3.11 as default
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/pip     pip     /usr/bin/pip3       1

WORKDIR /build

# Layer-cache: install deps before copying app code
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip scikit-build patchelf \
 && pip install --no-cache-dir -r requirements.txt

# Build qiskit-aer 0.17.2 with CUDA GPU support from source.
# qiskit-aer-gpu on PyPI tops at 0.15.1 (incompatible with qiskit 2.x).
# qiskit-aer 0.17.2 release explicitly fixes GPU builds for qiskit>=2.1.
# Uses old scikit-build so cmake args are passed with -- separator.
RUN git clone --depth 1 --branch 0.17.2 https://github.com/Qiskit/qiskit-aer.git /tmp/qiskit-aer-src \
 && cd /tmp/qiskit-aer-src \
 && CUDACXX=/usr/local/cuda/bin/nvcc \
    CMAKE_ARGS="-DAER_THRUST_BACKEND=CUDA -DAER_CUDA_ARCHITECTURES=${CUDA_ARCH}" \
    pip install --no-cache-dir . \
 && rm -rf /tmp/qiskit-aer-src


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy application source
COPY api/     ./api/
COPY pricer/  ./pricer/
COPY config/  ./config/
COPY data/    ./data/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--log-config", "/app/config/logging.yaml"]
