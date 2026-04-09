# syntax=docker/dockerfile:1.4
ARG CUDA_VERSION=11.8.0
ARG UBUNTU_VERSION=20.04
ARG PYTHON_VERSION=3.9
ARG HOST_UID=1000
ARG HOST_GID=1000
ARG PANDA_PY_INSTALL_MODE=source
ARG PANDA_PY_VERSION=0.8.1
ARG LIBFRANKA_VERSION=0.15.3
ARG PANDA_PY_GIT_REF=v0.8.1

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG PYTHON_VERSION
ARG PANDA_PY_INSTALL_MODE
ARG PANDA_PY_VERSION
ARG LIBFRANKA_VERSION
ARG PANDA_PY_GIT_REF

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MAMBA_ROOT_PREFIX=/opt/micromamba \
    VIRTUAL_ENV=/opt/venv \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib \
    PATH=/opt/venv/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    bzip2 \
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
    libfmt-dev \
    libgl1 \
    libglfw3 \
    libglib2.0-0 \
    libgtk-3-0 \
    libjsoncpp-dev \
    libosmesa6 \
    libosmesa6-dev \
    libpoco-dev \
    libspdlog-dev \
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
    python${PYTHON_VERSION}-distutils \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba
RUN micromamba create -y --always-copy -p ${VIRTUAL_ENV} -c conda-forge \
    python=${PYTHON_VERSION} \
    pip \
    numpy=1.23.3 \
    "pinocchio=3.7.*" \
    "casadi=3.7.*" \
    && micromamba clean --all --yes
RUN python -m pip install --upgrade pip

WORKDIR /build

RUN mkdir -p /build/docker_for_inference
COPY docker_for_inference/dp_inference_requirements.txt /build/docker_for_inference/
COPY docker_for_inference/pip_build_constraints.txt /build/docker_for_inference/

ENV PIP_CONSTRAINT=/build/docker_for_inference/pip_build_constraints.txt

RUN --mount=type=cache,target=/root/.cache/pip python -m pip install -r /build/docker_for_inference/dp_inference_requirements.txt
COPY docker_for_inference/dp_runtime_requirements.txt /build/docker_for_inference/
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install -r /build/docker_for_inference/dp_runtime_requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install cmake==3.24.3
COPY docker_for_inference/hirol_inference_requirements.txt /build/docker_for_inference/
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install -r /build/docker_for_inference/hirol_inference_requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install "scikit-build-core<1" "pybind11<3"
RUN micromamba install -y -p ${VIRTUAL_ENV} -c conda-forge libboost-python-devel \
    && micromamba clean --all --yes
RUN python - <<'PY'
import pinocchio
print("pinocchio:", pinocchio.__version__)
PY

COPY docker_for_inference/install_fr3_stack.sh /build/docker_for_inference/
COPY docker_for_inference/vendor /build/docker_for_inference/vendor

RUN bash /build/docker_for_inference/install_fr3_stack.sh \
    "/build/docker_for_inference/vendor" \
    "${PANDA_PY_INSTALL_MODE}" \
    "${PANDA_PY_VERSION}" \
    "${LIBFRANKA_VERSION}" \
    "${PANDA_PY_GIT_REF}"
RUN --mount=type=cache,target=/root/.cache/pip if [ -f /build/docker_for_inference/vendor/dm_robotics_panda/setup.py ]; then python -m pip install -e /build/docker_for_inference/vendor/dm_robotics_panda; fi
RUN --mount=type=cache,target=/root/.cache/pip if [ -f /build/docker_for_inference/vendor/pika_sdk/setup.py ]; then python -m pip install -e /build/docker_for_inference/vendor/pika_sdk; fi
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install --force-reinstall --no-deps numpy==1.23.3 opencv-python==4.6.0.66
RUN python - <<'PY'
import numpy
import cv2
import pinocchio
import panda_py
from panda_py import Panda
print("numpy:", numpy.__version__)
print("cv2:", cv2.__version__)
print("pinocchio:", pinocchio.__version__)
print("panda_py:", panda_py.__version__)
print("Panda import: ok")
PY

RUN python -m pip uninstall -y torch torchvision torchaudio || true

RUN python -m pip install torch==2.1.0+cu118 torchvision \
    --index-url https://download.pytorch.org/whl/cu118

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu${UBUNTU_VERSION}

ARG PYTHON_VERSION
ARG HOST_UID
ARG HOST_GID

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    LD_LIBRARY_PATH=/usr/local/lib \
    MUJOCO_GL=osmesa \
    PATH=/opt/venv/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
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
    patchelf \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-distutils \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    if ! getent group "${HOST_GID}" >/dev/null; then \
        groupadd --gid "${HOST_GID}" teleophost; \
    fi; \
    for group_name in plugdev input dialout; do \
        if ! getent group "${group_name}" >/dev/null; then \
            groupadd "${group_name}"; \
        fi; \
    done; \
    useradd --uid "${HOST_UID}" --gid "${HOST_GID}" --create-home --shell /bin/bash teleophost; \
    usermod -aG plugdev,input,dialout teleophost; \
    mkdir -p /workspace /workspace/wandb /tmp/matplotlib; \
    chown teleophost:"${HOST_GID}" /workspace /workspace/wandb /tmp/matplotlib /home/teleophost

WORKDIR /workspace

COPY --from=builder --chown=teleophost:${HOST_GID} /opt/venv /opt/venv
COPY --from=builder /usr/local /usr/local
COPY --chown=teleophost:${HOST_GID} docker_for_inference /workspace/docker_for_inference

ENV DP_HIROL_PATH=/workspace/dp_hirol-main \
    PYTHONPATH=/workspace/HirolPlatform:/workspace/dp_hirol-main:/workspace \
    WANDB_DIR=/workspace/wandb \
    MPLCONFIGDIR=/tmp/matplotlib

CMD ["/bin/bash"]
