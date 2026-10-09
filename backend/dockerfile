FROM golang

ENV DEBIAN_FRONTEND=noninteractive

# Basic build tools + OpenCV development libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    cmake \
    git \
    wget \
    unzip \
    ca-certificates \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ONNX Runtime version
ENV ONNXRUNTIME_VERSION=1.29.0

# Download ONNX Runtime
RUN wget -q \
    https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz \
    -O /tmp/onnxruntime.tgz \
    && tar -xzf /tmp/onnxruntime.tgz -C /opt \
    && mv /opt/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION} /opt/onnxruntime \
    && rm /tmp/onnxruntime.tgz \
    && ln -s /opt/onnxruntime/lib/libonnxruntime.so \
           /opt/onnxruntime/lib/onnxruntime.so \
    && echo "/opt/onnxruntime/lib" > /etc/ld.so.conf.d/onnxruntime.conf \
    && ldconfig

ENV CGO_ENABLED=1

ENV CGO_CFLAGS="-I/opt/onnxruntime/include"
ENV CGO_LDFLAGS="-L/opt/onnxruntime/lib -lonnxruntime"
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib

COPY go.mod go.sum main.go postal_code_reader.go ./
COPY ./model ./model
RUN go mod download


CMD ["go", "run", "."]
