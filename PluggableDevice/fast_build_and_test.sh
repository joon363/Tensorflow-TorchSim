#!/usr/bin/env bash
set -Eeuo pipefail

cd /workspace/PyTorchSim/TensorFlow/PluggableDevice

quiet() {
  "$@" >/dev/null 2>&1
}

echo "=== BAZEL BUILD ==="
quiet bazel build -c opt //my_device/tools/pip_package:build_pip_package

echo "=== WHEEL BUILD ==="
quiet bazel-bin/my_device/tools/pip_package/build_pip_package .

echo "=== PIP INSTALL ==="
quiet pip install my_device-0.0.1-cp310-cp310-linux_x86_64.whl

echo "=== RUN TEST ==="
python test_device.py
