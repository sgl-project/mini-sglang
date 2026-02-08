ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04
ARG PYTHON_VERSION=3.12

# Build stage
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG PYTHON_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Copy all source files (editable install requires source to exist)
COPY pyproject.toml ./
COPY python/ ./python/

# Create venv and install dependencies
RUN uv venv --python=python${PYTHON_VERSION} /app/.venv \
    && . /app/.venv/bin/activate \
    && uv pip install -e . \
    && uv pip install torch-c-dlpack-ext

# Runtime stage
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS runtime

ARG PYTHON_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1001 minisgl

# Copy application from builder
COPY --from=builder --chown=minisgl:minisgl /app /app

# Create cache directories
RUN mkdir -p /app/.cache/huggingface /app/.cache/tvm-ffi /app/.cache/flashinfer \
    && chown -R minisgl:minisgl /app/.cache

WORKDIR /app

# Environment configuration
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:/app/.venv/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV PYTHONUNBUFFERED=1

# Set up cache directories
ENV HF_HOME=/app/.cache/huggingface
ENV TVM_FFI_CACHE_DIR=/app/.cache/tvm-ffi
ENV FLASHINFER_WORKSPACE_BASE=/app/.cache/flashinfer

USER minisgl

EXPOSE 1919

ENTRYPOINT ["python", "-m", "minisgl"]
CMD ["--help"]
