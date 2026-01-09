# ========================================
# Base Image (Python 3.10)
# ========================================
FROM python:3.10-slim

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    COQUI_TOS_AGREED=1 \
    TTS_HOME=/opt/tts_models \
    SPEECHBRAIN_CACHE=/opt/speechbrain_models \
    XDG_DATA_HOME=/opt \
    HUGGINGFACE_HUB_CACHE=/opt/huggingface_models \
    HF_HOME=/opt/huggingface_models \
    PATH="/opt/venv/bin:$PATH"

# ========================================
# System Dependencies
# ========================================
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    gcc \
    g++ \
    gnupg \
    make \
    espeak-ng \
    imagemagick \
    libsndfile1-dev \
    libgl1 \
    libglib2.0-0 \
    supervisor && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /opt/tts_models /opt/speechbrain_models /opt/huggingface_models && \
    chmod 777 /opt/tts_models /opt/speechbrain_models /opt/huggingface_models

# ========================================
# Python Virtual Environment
# ========================================
COPY requirements.txt .

RUN python3 -m venv /opt/venv && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm && \
    rm -rf /root/.cache/pip

# ========================================
# Pre-download models (XTTS, SpeechBrain)
# ========================================
COPY download_model.py .
RUN python download_model.py

# ========================================
# Downloads kokoro model
# ========================================
COPY download_kokoro.py .
RUN HF_HUB_OFFLINE=0 python download_kokoro.py

# ========================================
# Set offline mode for runtime
# ========================================
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# ========================================
# Supervisor Config + App
# ========================================
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY . .

EXPOSE 1603 1604 11434

ENTRYPOINT []

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]