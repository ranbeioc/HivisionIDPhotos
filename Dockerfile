FROM python@sha256:18ce9b03d18802119f4f9270d10a1ceb45d0acb768c305b7fbbd7c6b5b5a020b AS compute

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENCV_FOR_THREADS_NUM=1
ENV HIVISION_MIN_AVAILABLE_MEMORY_MB=2048

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-app.txt ./
RUN pip install --no-cache-dir --requirement requirements.txt --requirement requirements-app.txt

COPY . .
ARG MODEL_SET=preview-default
RUN python scripts/download_verified_models.py --set "${MODEL_SET}" \
    && python scripts/download_verified_models.py --set "${MODEL_SET}" --verify-only \
    && python scripts/verify_model_provenance.py --set "${MODEL_SET}" \
    && python -m compileall api services scripts/benchmark_models.py scripts/probe_memory_pressure.py scripts/run_golden_matrix.py scripts/smoke_legacy_route.py deploy_api.py

RUN useradd --create-home --uid 10001 hivision \
    && chown -R hivision:hivision /app
USER 10001:10001

EXPOSE 8090
CMD ["python", "-u", "deploy_api.py"]

FROM compute AS legacy
USER root
COPY requirements-legacy.txt ./
RUN pip install --no-cache-dir --requirement requirements-legacy.txt
ENV ENABLE_LEGACY_GRADIO=true
USER 10001:10001

FROM compute AS production
