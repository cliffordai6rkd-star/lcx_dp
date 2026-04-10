#!/usr/bin/env bash
set -euo pipefail

vendor_root="${1:-/build/docker_for_inference/vendor}"
install_mode="${2:-auto}"
panda_py_version="${3:-0.8.1}"
libfranka_version="${4:-0.15.0}"
panda_py_ref="${5:-v0.8.1}"

python_bin="${PYTHON_BIN:-python}"
libfranka_src=""
panda_py_src=""

clone_source_if_needed() {
    local repo_url="$1"
    local ref="$2"
    local dest="$3"
    local alt_ref=""

    rm -rf "${dest}"
    if [[ "${ref}" == v* ]]; then
        alt_ref="${ref#v}"
    else
        alt_ref="v${ref}"
    fi

    if ! git clone --depth 1 --branch "${ref}" "${repo_url}" "${dest}"; then
        rm -rf "${dest}"
        git clone --depth 1 --branch "${alt_ref}" "${repo_url}" "${dest}"
    fi
    git -C "${dest}" submodule update --init --recursive || true
}

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

if [[ -z "${libfranka_src}" && "${install_mode}" != "pip" ]]; then
    libfranka_src="/tmp/libfranka-src"
    clone_source_if_needed \
        "https://github.com/frankarobotics/libfranka.git" \
        "${libfranka_version}" \
        "${libfranka_src}"
fi

if [[ -z "${panda_py_src}" && "${install_mode}" != "pip" ]]; then
    panda_py_src="/tmp/panda-py-src"
    clone_source_if_needed \
        "https://github.com/JeanElsner/panda-py.git" \
        "${panda_py_ref}" \
        "${panda_py_src}"
fi

if [[ -n "${panda_py_src}" ]]; then
    # libfranka 0.15.x introduces an overload for Robot::loadModel; v0.8.1
    # panda-py needs an explicit overload cast to compile against it.
    "${python_bin}" - <<PY
from pathlib import Path

path = Path("${panda_py_src}/src/libfranka.cpp")
source = path.read_text()
source = source.replace(
    '.def("load_model", &franka::Robot::loadModel)',
    '.def("load_model", py::overload_cast<>(&franka::Robot::loadModel))',
)
path.write_text(source)
PY
fi

if [[ -n "${libfranka_src}" ]]; then
    boost_root="${VIRTUAL_ENV:-/opt/venv}"
    cmake -S "${libfranka_src}" -B /tmp/libfranka-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="${boost_root};/usr/local;/usr/share/eigen3/cmake" \
        -DCMAKE_INCLUDE_PATH="/usr/include/eigen3" \
        -DBOOST_ROOT="${boost_root}" \
        -DBoost_NO_SYSTEM_PATHS=ON \
        -DEigen3_DIR="/usr/share/eigen3/cmake" \
        -DEIGEN3_INCLUDE_DIR="/usr/include/eigen3" \
        -DEIGEN3_INCLUDE_DIRS="/usr/include/eigen3" \
        -DBUILD_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF
    cmake --build /tmp/libfranka-build --parallel "$(nproc)"
    cmake --install /tmp/libfranka-build
    ldconfig
fi

if [[ -n "${panda_py_src}" ]]; then
    export CMAKE_PREFIX_PATH="${VIRTUAL_ENV:-/opt/venv};/usr/local${CMAKE_PREFIX_PATH:+;${CMAKE_PREFIX_PATH}}"
    export CMAKE_ARGS="-DBOOST_ROOT=${VIRTUAL_ENV:-/opt/venv} -DBoost_NO_SYSTEM_PATHS=ON -DCMAKE_INCLUDE_PATH=/usr/include/eigen3 -DEigen3_DIR=/usr/share/eigen3/cmake ${CMAKE_ARGS:-}"
    export CPLUS_INCLUDE_PATH="/usr/include/eigen3${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
    export C_INCLUDE_PATH="/usr/include/eigen3${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
    env -u PIP_CONSTRAINT -u PIP_BUILD_CONSTRAINT \
        "${python_bin}" -m pip install --no-deps --no-build-isolation "${panda_py_src}"
elif [[ "${install_mode}" == "pip" || "${install_mode}" == "auto" ]]; then
    "${python_bin}" -m pip install "panda-python==${panda_py_version}"
fi
