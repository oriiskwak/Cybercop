FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:$PATH \
    MODEL_CACHE_DIR=/models/huggingface/vlm \
    HF_HOME=/models/huggingface/hub_cache \
    CYBERCOP_WORK_DIR=/tmp/cybercop

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg libgl1 libglib2.0-0 python3 python3-pip python3-venv tini \
    && python3 -m venv /opt/venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY cybercop_pipeline_AdotX_v0_3.py ./
COPY pipeline ./pipeline
COPY rag ./rag
COPY app/__init__.py app/main_v0_3.py ./app/
COPY README.md LICENSE ./

# RapidOCR downloads its versioned ONNX detector/recognizer on first initialization.
# Bake those small assets while the virtual environment is still writable by root.
RUN python -c "from cybercop_pipeline_AdotX_v0_3 import load_ocr; load_ocr()"

RUN useradd --create-home --uid 10001 cybercop \
    && mkdir -p /models/huggingface /tmp/cybercop \
    && chown -R cybercop:cybercop /models /tmp/cybercop /app

USER cybercop

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)"

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "app.main_v0_3:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
