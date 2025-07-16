FROM arm64v8/python:3.9-slim

WORKDIR /app

RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    iputils-ping \
    iproute2 \
    iperf3 \
    vim \
    curl \
    net-tools && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    

RUN pip install --no-cache-dir \
    tensorflow==2.13.0 \
    numpy==1.24.3

COPY . .