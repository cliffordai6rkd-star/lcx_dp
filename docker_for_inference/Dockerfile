ARG CUDA_VERSION=11.6.2
ARG UBUNTU_VERSION=20.04
ARG PYTHON_VERSION=3.9

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG PYTHON_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    curl \
    git \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev \
    libboost-all-dev \
    libeigen3-dev \
    libffi-dev \
    libgl1 \
    libglfw3 \
    libglib2.0-0 \
    libgtk-3-0 \
    libosmesa6 \
    libosmesa6-dev \
    libpoco-dev \
    libsm6 \
    libswresample-dev \
    libswscale-dev \
    libusb-1.0-0 \
    libxext6 \
    libxrender1 \
    libyaml-cpp-dev \
    ninja-build \
    patchelf \
    pkg-config \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-tk \
    python${PYTHON_VERSION}-venv \
    && python${PYTHON_VERSION} -m ensurepip --upgrade \
    && python${PYTHON_VERSION} -m pip install --no-cache-dir uv \
    && rm -rf /var/lib/apt/lists/*

RUN uv venv --python python${PYTHON_VERSION} ${VIRTUAL_ENV}

ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

WORKDIR /build

COPY dp_hirol-main /build/dp_hirol-main
COPY HirolPlatform /build/HirolPlatform
COPY docker_for_inference /build/docker_for_inference

WORKDIR /build/dp_hirol-main

# gym==0.21.0 ships legacy metadata rejected by pip>=24.1.
RUN uv pip install -r requirements/uv-pip-bootstrap.txt
RUN python -m pip install --no-build-isolation gym==0.21.0
RUN uv pip install --index-strategy unsafe-best-match --build-constraints requirements/uv-build-constraints.txt -r requirements/uv-common.txt
RUN uv pip install --index-strategy unsafe-best-match --build-constraints requirements/uv-build-constraints.txt -r /build/docker_for_inference/dp_runtime_requirements.txt
RUN uv pip install .

WORKDIR /build

# Install only the extra runtime pieces needed by the Hirol inference chain.
RUN uv pip install -r /build/docker_for_inference/hirol_inference_requirements.txt
RUN if [ -f /build/docker_for_inference/vendor/dm_robotics_panda/setup.py ]; then uv pip install -e /build/docker_for_inference/vendor/dm_robotics_panda; fi
RUN if [ -f /build/docker_for_inference/vendor/pika_sdk/setup.py ]; then uv pip install -e /build/docker_for_inference/vendor/pika_sdk; fi
RUN uv pip install -e /build/HirolPlatform

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ARG PYTHON_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    MUJOCO_GL=osmesa \
    PATH=/opt/venv/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ffmpeg \
    git \
    libgl1 \
    libglfw3 \
    libglib2.0-0 \
    libgomp1 \
    libgtk-3-0 \
    libosmesa6 \
    libpoco-dev \
    libsm6 \
    libusb-1.0-0 \
    libxext6 \
    libxrender1 \
    libyaml-cpp-dev \
    patchelf \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-tk \
    python${PYTHON_VERSION}-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build /workspace

ENV DP_HIROL_PATH=/workspace/dp_hirol-main \
    PYTHONPATH=/workspace/HirolPlatform:/workspace/dp_hirol-main:/workspace \
    WANDB_DIR=/workspace/wandb \
    MPLCONFIGDIR=/tmp/matplotlib

CMD ["/bin/bash"]
