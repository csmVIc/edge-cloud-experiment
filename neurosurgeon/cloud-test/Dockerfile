FROM arm64v8/python:3.7-slim

WORKDIR /app

COPY . .

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    vim \
    net-tools \
    iproute2 \
    iperf3 \
    iputils-ping \
    curl \ 
    libopenblas-dev \ 
    gcc \ 
    libjpeg-dev \
    zlib1g-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/KumaTea/pytorch-aarch64/releases/download/v1.6.0/torch-1.6.0-cp37-cp37m-linux_aarch64.whl \
    https://github.com/KumaTea/pytorch-aarch64/releases/download/v1.6.0/torchvision-0.7.0-cp37-cp37m-linux_aarch64.whl

RUN pip install  --no-cache-dir \ 
    xlrd \
    speedtest \
    xlwt \
    xlutils \
    apscheduler \
    torch-1.6.0-cp37-cp37m-linux_aarch64.whl \
    torchvision-0.7.0-cp37-cp37m-linux_aarch64.whl