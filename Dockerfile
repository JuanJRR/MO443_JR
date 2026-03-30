# 1. Base oficial de Python
FROM python:3.12-slim-bookworm

LABEL maintainer="Juan Jose Rodriguez <rodriguezjuan001@outlook.com>"

# 2. Configuración de entorno y zona horaria
ENV TZ=America/Sao_Paulo \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app
    
# 3. Instalación de dependencias del sistema en una sola capa
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    python3-tk \
    && pip install --no-cache-dir --upgrade pip \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 4. Optimización de Caché: Instalamos dependencias ANTES de copiar todo el código
COPY pyproject.toml .

# Instalamos solo las dependencias; esto no cambiará a menos que edites el pyproject.toml
RUN pip install --no-cache-dir --break-system-packages .[dev]

# 5. Copia del código fuente al final para evitar reconstrucciones pesadas
COPY . .

CMD ["/bin/bash"]