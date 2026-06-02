# 고차원 연산(Conv2D 및 GEMM) 정합성 검증 보고서 (Correctness Report: Conv & GEMM)

## 1. 개요 (Overview)
본 보고서는 TensorFlow에서 PyTorchSim NPU 시뮬레이션 파이프라인으로 연동되는 고차원 텐서 연산(특히 GEMM 및 Conv2D)의 정합성 검증 방법과 그 결과, 그리고 직면했던 기술적 문제들의 해결 과정을 다룹니다. 

기존 벡터 연산 위주의 정합성 검증 단계에서 나아가, 비정방형 행렬 곱셈(GEMM) 및 공간 정보를 내포한 합성곱 연산(Conv2D)에 대하여 수치 정합성을 확보하고 발생 가능한 컴파일 및 시뮬레이션 예외들을 성공적으로 규명 및 격리했습니다.

---

## 2. 테스트 케이스 설계 (Test Case Design)

### 2.1 GEMM (Non-square Matrix Multiplication)
정방형 행렬 곱셈 이외의 비정방형 크기에서의 계산 정합성을 확인하기 위해 설계된 테스트 케이스입니다.
* **TF 연산**: `tf.matmul(x, y)`
* **입력 형상 (Shape)**:
  * `x_tf`: `[16, 32]` (정규분포 난수)
  * `y_tf`: `[32, 64]` (정규분포 난수)
* **출력 형상 (Shape)**: `[16, 64]`
* **목적**: 차원이 다른 입력 행렬에 대해 스트라이드 계산 및 평탄화 시 루프 한계치가 올바르게 할당되고 동작하는지 검증합니다.

### 2.2 Conv2D (NHWC / HWCF Layout)
TensorFlow의 표준 이미지/피처 연산 포맷인 NHWC 및 필터 포맷인 HWCF 형상을 처리하는 이차원 합성곱 테스트 케이스입니다.
* **TF 연산**: `tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')`
* **입력 및 필터 형상 (Shape)**:
  * `conv_x_tf`: `[1, 14, 14, 8]` (배치 크기 1, 가로/세로 14, 채널 8)
  * `conv_w_tf`: `[3, 3, 8, 16]` (필터 크기 3x3, 입력 채널 8, 출력 채널 16)
* **출력 형상 (Shape)**: `[1, 14, 14, 16]`
* **목적**: XLA에 의해 생성된 공간 패딩(Pad) 연산과 다차원 텐서 인덱싱이 NPU 백엔드에서 올바르게 수치 정합성을 지키는지 검증합니다.

---

## 3. 발생한 문제 및 해결 방안 (Issues & Resolutions)

고차원 연산 컴파일 과정에서 MLIR 하부 툴체인(`mlir-opt` 및 `mlir-translate`)의 한계로 인해 두 가지 주요 컴파일 및 링크 에러가 발생했으며, 이를 리라이터 코드 수정을 통해 극복했습니다.

### 3.1 linalg.map arity mismatch 오류 해결
XLA에서 합성곱을 낮출 때 생성된 패딩 연산이 Bufferization 패스를 거치면서 `linalg.map` 연산자로 번역되었습니다. 그러나 하부 RISC-V LLVM의 `mlir-opt` 버전 한계로 인해 다음과 같은 검증 실패 오류가 발생했습니다:
> `error: 'linalg.map' op expects number of operands to match the arity of mapper, but got: 0 and 1`

* **원인**: 입력 피드가 없는 상수 매핑 블록을 하부 옵티마이저가 처리하지 못했습니다.
* **해결**: `transform_tf_mlir` 단계에서 정규식을 활용하여, 단순 상수(패딩용 제로 값)를 채우는 `linalg.map` 블록을 동등한 의미의 `linalg.fill` 연산자로 변환했습니다.
  * *이전*: `linalg.map outs(%alloc_1) (%init) { linalg.yield %cst }`
  * *이후*: `linalg.fill ins(%cst) outs(%alloc_1)`

### 3.2 memref.subview 및 memref.copy 미지원 오류 해결
XLA의 패딩 연산 로어링 결과, 중간 영역을 채우기 위한 `memref.subview`와 원본 피처맵 복사를 위한 `memref.copy`가 잔존했습니다. 이로 인해 컴파일 단계에서 다음과 같은 오류가 야기되었습니다:
1. `mlir-translate` 시 `memref.subview` 다이얼렉트를 찾지 못하는 컴파일 오류.
2. 링크 시점에 베어메탈(Bare-metal) 크로스 링커가 `memrefCopy` 런타임 라이브러리 함수를 찾지 못하는 링크 오류 (`undefined reference to 'memrefCopy'`).

* **해결 (subview)**: 컴파일 패스 체인(`mlir_compile_command` 및 `mlir_gem5_compile_command`)의 처음에 `-expand-strided-metadata` 패스를 명시적으로 추가했습니다. 이 패스는 `memref.subview`를 하부 링커가 이해할 수 있는 `memref.reinterpret_cast` 구조로 해체해 줍니다.
* **해결 (copy)**: `transform_tf_mlir` 단계에 커스텀 C-like 루프 생성기를 연동하여, `memref.copy` 구문을 대상 텐서의 정적 형상에 맞춘 중첩 `scf.for`와 `memref.load/store` 루프 구조로 자동 치환했습니다. 이를 통해 외부 런타임 의존성인 `memrefCopy` 함수 호출을 원천 제거하고 순수 루프로 빌드되도록 보장했습니다.

---

## 4. 검증 결과 (Verification Results)

수정된 리라이터 및 컴파일러 옵션 하에 모든 테스트 케이스가 성공적으로 실행되었습니다:

```
Test [7/8] GEMM (non-square Matmul 16x32x64): PASSED
Test [8/8] Conv2D 14x14x8 to 16: PASSED

================ Summary: [8/8] Passed ================
```

* **비정방형 GEMM**: 차원 크기 축이 상이한 상태에서도 스트라이드 정보와 데이터 변환이 완벽히 인지되어 Spike 시뮬레이터에서 계산 정합성을 확인했습니다.
* **Conv2D**: 공간 패딩 처리가 루프로 완전히 치환되어 DRAM 상에서 합성곱 커널 연산이 오차 없이 시뮬레이션되었으며, 출력값은 TensorFlow CPU 예측 값과 `rtol=1e-4`, `atol=1e-4` 기준 정밀도로 완벽하게 일치했습니다.

---

## 5. 결론 및 백엔드 함수 명명 충돌 여부 분석

### 5.1 백엔드 함수 명명 구조 분석 (커널명 충돌 문제 여부)
여러 연산이 동시 컴파일될 때 `mlir_common.py:L837-L840` 및 `L854-L867` 부분의 함수 명명 규칙(예: `kernel` vs `main` 또는 `kernel_name`)으로 인한 충돌 여부를 분석했습니다.

* **동작 검토**:
  * PyTorchSim의 기능 시뮬레이션 모드에서 각 연산 커널은 해당 연산 소스 코드의 해시(Hash)에 의해 고유하게 지정되는 개별 임시 디렉토리(예: `/workspace/PyTorchSim/outputs/hn2p.../`) 내에 생성됩니다.
  * 컴파일러는 독립된 단일 빌드 경로 상에서 소스 파일(`{key}.mlir`)을 컴파일하여 각기 다른 바이너리(`validation_binary`)로 링크합니다.
  * 따라서 각 MLIR 파일 내부에 정의된 진입 함수명이 무조건 `func.func @kernel`로 통일되거나 `wrapper_kernel`로 링킹되더라도, 이들은 **서로 다른 시뮬레이터 프로세스 및 가상 주소 공간**에서 개별적으로 실행되므로 함수 이름 충돌이나 물리적인 링킹 충돌 현상은 전혀 발생하지 않습니다.
  * 만약 `V.graph.cpp_wrapper`가 활성화되어 여러 커널을 하나의 파일에 모아 컴파일 및 링킹하는 구조라면 이름 충돌이 나겠지만, 현재의 TensorFlow 직접 주입 컴파일 구조에서는 철저한 프로세스 격리가 유지되어 충돌이 없습니다.
