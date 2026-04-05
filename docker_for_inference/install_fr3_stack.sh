#!/usr/bin/env bash
set -euo pipefail

vendor_root="${1:-/build/docker_for_inference/vendor}"
install_mode="${2:-auto}"
panda_py_version="${3:-0.8.1}"

python_bin="${PYTHON_BIN:-python}"
libfranka_src=""
panda_py_src=""

for candidate in \
    "${vendor_root}/libfranka" \
    "${vendor_root}/franka/libfranka"; do
    if [[ -f "${candidate}/CMakeLists.txt" ]]; then
        libfranka_src="${candidate}"
        break
    fi
done

for candidate in \
    "${vendor_root}/panda_py" \
    "${vendor_root}/panda-python" \
    "${vendor_root}/pandapy"; do
    if [[ -f "${candidate}/setup.py" || -f "${candidate}/pyproject.toml" ]]; then
        panda_py_src="${candidate}"
        break
    fi
done

if [[ "${install_mode}" == "source" && -z "${panda_py_src}" ]]; then
    echo "PANDA_PY_INSTALL_MODE=source but no panda_py source was found under ${vendor_root}" >&2
    exit 1
fi

if [[ -n "${libfranka_src}" ]]; then
    cmake -S "${libfranka_src}" -B /tmp/libfranka-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTS=OFF
    cmake --build /tmp/libfranka-build --parallel "$(nproc)"
    cmake --install /tmp/libfranka-build
fi

if [[ -n "${panda_py_src}" ]]; then
    "${python_bin}" -m pip install "${panda_py_src}"
elif [[ "${install_mode}" == "pip" || "${install_mode}" == "auto" ]]; then
    "${python_bin}" -m pip install "panda-python==${panda_py_version}"
fi
