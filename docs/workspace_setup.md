# Overview
필요한 파일
```
/LLVM-22.1.0-rc1-Linux-X64/bin/mlir-opt
/LLVM-22.1.0-rc1-Linux-X64/bin/mlir-translate
/workspace/PyTorchSim/Tensorflow/build/bin/stablehlo-opt
```

PluggableDevice 세팅: PluggableDevice/README.md 참고

있으면 좋은 파일

- tensorflow 소스 코드
  ```
  cd /workspace
  git clone https://github.com/tensorflow/tensorflow
  ```
# 1. llvm-project 22.1.0 Binary download
- stablehlo1.0은 llvm 22.1.0을 써야 합니다. 왜냐면 https://github.com/llvm/llvm-project/commit/906295b8a31c8dac5aa845864c0bca9f02f86184 을 보면 `build`를 `create`로 바꿨기 때문입니다. 현재 PyTorchSim의 17로는 불가능합니다. ㅠㅠ
- https://github.com/llvm/llvm-project/releases/tag/llvmorg-22.1.0-rc1 다운로드

```bash
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-22.1.0-rc1/LLVM-22.1.0-rc1-Linux-X64.tar.xz

tar -xJf LLVM-22.1.0-rc1-Linux-X64.tar.xz
```
# 2. stablehlo-opt Build

### 빌드한 파일을 올려 놓았습니다. 
[build](https://github.com/joon363/StableHLO-compiler-pre-built-binary) 이거를 PyTorchSim/Tensorflow/build 안에 클론하면 됩니다.

직접 하는 법:
## 1. Clone repos

### StableHLO 1.0 Source Code

```bash
cd /workspace
git clone --depth=1 --single-branch https://github.com/openxla/stablehlo
```


## 2. StableHLO Build
### Modify CMakeLists.txt

- 받아온 바이너리에는 `FileCheck`, `not`이라는게 있는데 이 두 바이너리는 `-D LLVM_INSTALL_UTILS=ON` 을 해서 llvm-project를 직접 빌드 했을 때에만 생깁니다. 1번에서 받아온 바이너리에는 없어서 그냥 빌드하면 에러가 납니다.
- 그런데 이 둘은 test suite에만 사용되므로, 단순 stablehlo-opt 빌드용으로는 필요 없습니다.
- llvm 프젝트를 직접 빌드할 수 있는 환경이라면 스킵해도 되지만, 제 경우에는 빌드가 너무 오래 걸려서 (컴퓨터 켜놓고 4시간 다녀왔는데 10% 되어 있고 컴이 죽어 있었음) 아래처럼 하였습니다.

`/workspace/stablehlo/stablehlo/CMakeLists.txt`

```python
# add_subdirectory(api)
add_subdirectory(conversions)
add_subdirectory(dialect)
# add_subdirectory(integrations)
add_subdirectory(reference)
add_subdirectory(tests)
# add_subdirectory(testdata)
add_subdirectory(tools)
add_subdirectory(transforms)
```

`/workspace/stablehlo/stablehlo/conversions/linalg/CMakeLists.txt`

```python
# add_subdirectory(tests)
```

`/workspace/stablehlo/stablehlo/conversions/tosa/CMakeLists.txt`

```python
# add_subdirectory(tests)
```

`/workspace/stablehlo/stablehlo/tools/CMakeLists.txt`

```python
# add_dependencies(check-stablehlo-quick stablehlo-lsp-server)
```

`/workspace/stablehlo/stablehlo/tests/CMakeLists.txt`

```bash
# add_lit_testsuite(check-stablehlo-tests "Running the tests/ suite"
#   ${CMAKE_CURRENT_BINARY_DIR}
#   DEPENDS
#   FileCheck not
#   stablehlo-opt
#   stablehlo-translate
# )
# add_dependencies(check-stablehlo-quick check-stablehlo-tests)
```

### Build

```bash
rm -rf build

mkdir -p build

cd build

cmake .. -GNinja   -DLLVM_ENABLE_LLD="$LLVM_ENABLE_LLD" -DCMAKE_BUILD_TYPE='Release'   -DLLVM_ENABLE_ASSERTIONS='ON' -DSTABLEHLO_ENABLE_BINDINGS_PYTHON='OFF' -DMLIR_DIR=/LLVM-22.1.0-rc1-Linux-X64/lib/cmake/mlir # 아까 llvm 바이너리 압축 푼 경로

ninja -j 8 # 원하는 개수로 설정
```