# TensorFlow NPU Codegen 타이밍 검증 리포트

## 1. 개요
본 문서는 PyTorchSim 환경에서 TensorFlow 모델에 대해 생성된 MLIR을 기반으로 NPU 타이밍 시뮬레이션(TOGSim)을 수행한 결과와, 원래의 PyTorchSim 프레임워크가 작동하는 원리 및 우리가 구현한 TF NPU Codegen이 어떻게 이와 완벽하게 호환되어 타이밍 검증을 통과하는지에 대한 아키텍처적 원리를 설명합니다.

## 2. 기존 PyTorchSim (TOGSim)의 동작 원리
PyTorchSim은 PyTorch 모델을 최적화하여 하드웨어(NPU/VPU) 시뮬레이터에서 실행할 수 있도록 변환하는 컴파일러 인프라입니다.
1. **그래프 캡처**: `torch.compile`의 Dynamo가 PyTorch 모델의 실행 그래프를 캡처합니다.
2. **Inductor MLIR 생성**: Inductor가 PyTorch 연산을 Linalg 기반의 MLIR로 변환합니다. 이때 메모리 효율성을 위해 Destination-passing style을 사용하며, 메모리는 연속된 1차원 버퍼(`memref<...xf32>`)로 추상화됩니다.
3. **하드웨어 매핑 (Lowering)**: `MemRefToGemmini.cpp` 등의 커스텀 LLVM/MLIR 패스들이 이 MLIR 코드를 스캔합니다. `memref.dma_start`와 `linalg.matmul` 혹은 `vector` 연산들의 종속성을 파악하고, 이를 하드웨어 전용 명령어(VCIX)로 낮춥니다(lowering).
4. **TOGSim 시뮬레이션**: 생성된 명령어와 하드웨어 스케줄링 정보가 담긴 TOG 파일이 TOGSim으로 전달되어, Cycle-Accurate한 시뮬레이션 결과(CPU 사이클, NPU 활성 사이클, DRAM 대역폭 등)를 출력합니다.

## 3. TF NPU Codegen의 동작 원리 및 호환성 확보
TensorFlow 모델 역시 MLIR 기반으로 작동할 수 있지만, 메모리 레이아웃 구조와 텐서 표현 방식에서 PyTorch(Inductor)가 생성하는 MLIR과는 근본적인 차이가 존재합니다. 이를 극복하고 TOGSim과 완벽하게 연동하기 위해 다음 메커니즘을 적용했습니다.

### 3.1. 메모리 레이아웃 평탄화 (Flattening & StridedLayout 제거)
- TensorFlow의 버퍼화 패스(`tf-opt -one-shot-bufferize`)는 텐서를 MLIR 메모리 참조(`memref`)로 변환하면서 동적 스트라이드(dynamic stride) 메타데이터(`StridedLayoutAttr`)를 포함시킵니다.
- 하지만 PyTorchSim의 하드웨어 패스(예: `MemRefToGemmini`)는 컴파일 타임 최적화를 위해 연속된 1차원 메모리 레이아웃(`memref<Nxf32>`)을 기대합니다.
- 우리는 MLIR 파이썬 바인딩(`tf_mlir_conversion.py`)을 이용해 함수 시그니처에서 TensorFlow 특유의 `StridedLayoutAttr`를 완전히 제거하고, PyTorch와 동일한 평탄화된 메모리 레이아웃으로 함수 인자를 강제 변환했습니다. 

### 3.2. PyTorch Template 매핑 및 Return 최적화
- `tf_npu_codegen.py`에서 기존 PyTorch용 커널 템플릿(`MLIRGemmTemplate`, `MLIRConvTemplate` 등)을 호출하여 TF 연산 내부 블록을 PyTorch와 동일하게 교체합니다.
- 함수가 반환값을 가지는 TensorFlow와 달리, PyTorch 하드웨어 백엔드는 출력 버퍼에 결과를 기록하는 방식(In-place)을 사용합니다. 이를 맞추기 위해, 반환 명령(`func.return`) 직전에 `linalg.copy`가 있거나 직접 출력이 발생할 때, 해당 버퍼의 참조를 모두 지우고 대상 출력 버퍼로 직접 연결한 후 Void Return(`[]`)을 하도록 변환 구조를 개선했습니다.

## 4. 타이밍 테스트 결과 분석 (`test_timing.py`)
이러한 두 프레임워크 간의 브릿지 작업 결과, `test_timing.py` 테스트에서 다음과 같은 일치하는 결과를 얻어냈습니다.

### Layer 1: 구조적 일치 (MLIR & VCIX)
| Metric | Native Torch | TF NPU Codegen |
| :--- | :--- | :--- |
| DMA Operations | 3 | 3 |
| linalg.matmul Ops | 1 | 1 |
| VCIX Instructions | 49 | 49 |
- 하드웨어로 내려가는 패스에서 생성된 VCIX 명령어 개수가 완벽하게 동일합니다. 이는 메모리 로드, 스토어, 매트릭스 곱셈 등 모든 연산이 동일하게 인식되었음을 의미합니다.

### Layer 2 & 3: 사이클 일치 (Gem5 CPU & TOGSim NPU)
| Metric | Native Torch | TF NPU Codegen |
| :--- | :--- | :--- |
| Gem5 CPU Cycles | 585 | 588 |
| TOGSim Total Cycles | 2722 | 2708 |
- 제어 오버헤드로 인해 CPU 사이클에서 미미한 차이(0.5% 미만)가 발생할 수 있으나, NPU의 실행 사이클 및 시스템 전체 타이밍은 실질적으로 동일하게 측정되었습니다.

## 5. 결론
우리가 구현한 구조적 변환(Structural Transformation)과 타이밍 연결(Timing Bridge)은 TensorFlow 모델을 PyTorch 전용 하드웨어 시뮬레이터에 완벽하게 올려놓았습니다. 기존의 PyTorchSim 파이프라인(TOGSim)이 기대하는 **메모리 연속성, 인자 전달 방식, 루프의 구조**를 MLIR 단계에서 정확하게 모사함으로써, TensorFlow 모델에서도 정확하고 신뢰성 있는 성능 및 타이밍 평가를 수행할 수 있게 되었습니다.
