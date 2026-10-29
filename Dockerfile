FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PORT=8501

# libgomp1 is needed by LightGBM (a PyCaret dependency)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY insurance/ insurance/
COPY assets/ assets/
COPY app.py .
# The trained model is produced by `make train` / the CI pipeline before the build
COPY models/ models/

# run as an unprivileged user
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# Which commit this image was built from (shown in the app sidebar)
ARG GIT_SHA=dev
ENV GIT_SHA=${GIT_SHA}

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD curl -fs http://localhost:${PORT}/_stcore/health || exit 1

CMD streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true
