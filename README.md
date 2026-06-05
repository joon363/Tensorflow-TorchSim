## At Local (Windows + cmd)

Actually, we don't have to git clone pytorchsim but its ok..

```bash
git lfs install
git clone https://github.com/PSAL-POSTECH/PyTorchSim.git
cd PyTorchSim
git clone https://github.com/joon363/Tensorflow-TorchSim Tensorflow
```

Now mount tensorflow folder, frontend only

```bash
cd tensorflow
docker run -it --ipc=host --name torchsim_mine ^
  -v "%cd%:/workspace/PyTorchSim/Tensorflow" ^
  -v "%cd%\..\PyTorchSimFrontend:/workspace/PyTorchSimFrontend" ^
  -w /workspace/PyTorchSim ^
  ghcr.io/psal-postech/torchsim-ci:v1.1.0 bash
```

## Inside the docker
```bash
apt update
apt install git-lfs

cd Tensorflow
source /opt/conda/etc/profile.d/conda.sh
conda activate

pip install jupyter tensorflow
pip install mlir-python-bindings -f https://makslevental.github.io/wheels

chmod +x /workspace/PyTorchSim/Tensorflow/binaries/stablehlo-opt
chmod +x /workspace/PyTorchSim/Tensorflow/binaries/stablehlo-translate
chmod +x /workspace/PyTorchSim/Tensorflow/binaries/stablehlo-lsp-server
chmod +x /workspace/PyTorchSim/Tensorflow/binaries/mlir-opt
chmod +x /workspace/PyTorchSim/Tensorflow/binaries/mlir-translate
```
## Reproduce Results
python tests/test_correctness.py
