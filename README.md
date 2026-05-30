git lfs install
git clone https://github.com/PSAL-POSTECH/PyTorchSim.git
git clone https://github.com/joon363/Tensorflow-TorchSim Tensorflow
git clone https://github.com/joon363/StableHLO-compiler-pre-built-binary.git build


docker run -it ^
  --ipc=host ^
  --name torchsim ^
  -p 8888:8888 ^
  -v "%cd%:/workspace/PyTorchSim" ^
  -w /workspace/PyTorchSim ^
  ghcr.io/psal-postech/torchsim-test-2-8:9bf6f67cab0b410b637db9a64fd74bb0451988bd ^
  bash

sudo apt update
sudo apt install git-lfs

cd /workspace/PyTorchSim
git clone https://github.com/joon363/Tensorflow-TorchSim Tensorflow

cd Tensorflow

pip install mlir-python-bindings -f https://makslevental.github.io/wheels
pip install jupyter tensorflow

chmod +x /workspace/PyTorchSim/Tensorflow/build/bin/stablehlo-opt

source /opt/conda/etc/profile.d/conda.sh
conda activate
pkill -9 -f jupyter
pkill -9 -f ipykernel
pkill -9 -f python
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root