# 1. Base oficial de NVIDIA para que el contenedor "hable" con la GPU
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

LABEL maintainer="Juan Jose Rodriguez <rodriguezjuan001@outlook.com>"

# 2. Configuración de entorno y zona horaria [cite: 4]
ENV TZ=America/Sao_Paulo \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src 

# 3. Instalación de dependencias del sistema en una sola capa [cite: 5]
# Añadimos libgl1 (OpenCV), usbutils (ISAC/Hardware) y python3
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python3-setuptools \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    git \
    usbutils \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 4. Optimización de Caché: Instalamos dependencias ANTES de copiar todo el código [cite: 6]
COPY pyproject.toml .
# Instalamos solo las dependencias; esto no cambiará a menos que edites el pyproject.toml
RUN pip install --no-cache-dir --break-system-packages .[dev]

# 5. Copia del código fuente al final para evitar reconstrucciones pesadas [cite: 10]
COPY . .

EXPOSE 8888
CMD ["/bin/bash"]