# TensorFlow 파이프라인 확장 검증 보고서 v2

## 1. 개요 (Overview)

본 보고서는 PyTorchSim의 TensorFlow 파이프라인에 대해 수행된 다음 작업들을 문서화합니다:

1. **모듈 구조 리팩토링**: TensorFlow 관련 로직을 `extension_codecache.py`에서 분리하여 독립 패키지 `TensorFlowFrontend`로 추출
2. **NPU 최적화 코드 생성기 초안 작성**: StableHLO 출력으로부터 SRAM/DMA 기반 타일링 MLIR을 "brick-by-brick"으로 조립하는 `tf_npu_codegen.py` 구현
3. **복합 모델 정합성 테스트 추가**: Perceptron, Sigmoid Linear, Large Matmul 테스트를 포함한 11개 테스트 케이스로 확장
4. **Gem5 타이밍 시뮬레이션 검증**: TOGSim 타이밍 모드 동작 확인

---

## 2. 아키텍처 변경 (Architecture Changes)

### 2.1 모듈 분리 (Module Extraction)

기존에 [extension_codecache.py](file:///c:/Users/MAIN/Desktop/PyTorchSim/PyTorchSimFrontend/extension_codecache.py)에 인라인으로 작성되어 있던 TensorFlow 전용 로직을 `Tensorflow/TensorFlowFrontend/` 패키지로 추출했습니다.

| 파일 | 역할 |
| :--- | :--- |
| [tf_mlir_conversion.py](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/TensorFlowFrontend/tf_mlir_conversion.py) | `transform_tf_mlir()`: TF MLIR → PyTorchSim 호환 MLIR 변환 (함수명 변경, 평탄화, reinterpret_cast 삽입, 반환값→참조쓰기 변환) |
| | `handle_tensorflow_direct_test()`: TENSORFLOW_MLIR_DIRECT_TEST 활성 시 tf-mlir.mlir 파일 리디렉션 및 헤더 복사 |
| [tf_npu_codegen.py](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/TensorFlowFrontend/tf_npu_codegen.py) | NPU 최적화 MLIR 생성기 (brick-by-brick): MockNode/MockLayout, 엘리먼트별 DMA 루프, GEMM/Conv2D 템플릿 래퍼 |

`extension_codecache.py`의 TF 관련 수정은 최소화되었으며, 핵심 로직은 import 호출로 대체되었습니다:

```python
# extension_codecache.py (간소화된 형태)
def transform_tf_mlir(content, arg_attributes=None):
    from Tensorflow.TensorFlowFrontend.tf_mlir_conversion import transform_tf_mlir as impl
    return impl(content, arg_attributes)
```

> [!IMPORTANT]
> **심볼 충돌 주의**: `mlir.ir` 모듈을 `TensorFlowFrontend` 파일의 모듈 레벨에서 import하면 `tensorflow` 동적 라이브러리와 심볼 충돌이 발생하여 세그멘테이션 폴트가 납니다. 모든 `mlir` import는 별도 서브프로세스 내에서만 실행해야 합니다.

### 2.2 NPU 코드 생성기 (tf_npu_codegen.py)

StableHLO/Linalg MLIR을 파싱하여 NPU 최적화 MLIR을 "brick-by-brick"으로 조립하는 생성기입니다. 현재 지원하는 패턴:

| StableHLO/Linalg 패턴 | 생성되는 NPU MLIR |
| :--- | :--- |
| `linalg.add`, `linalg.generic(addf)` | SRAM 버퍼 + DMA 전송 + affine.vector_load/store 루프 |
| `linalg.generic(mulf)` | 동일 (곱셈) |
| `linalg.generic(maximumf)` (ReLU) | SRAM 버퍼 + DMA + max(x, 0) 벡터 연산 |
| `linalg.matmul` | PyTorchSim `MLIRGemmTemplate` 호출 |
| `linalg.conv_2d_nhwc_fhwc` | PyTorchSim `MLIRConvTemplate` 호출 |

> [!NOTE]
> 이 생성기는 현재 초안(draft) 상태이며, 실제 컴파일 파이프라인에 통합되지 않은 상태입니다. 통합 시 타이밍 모드(Gem5)에서의 사이클 정밀 시뮬레이션이 가능해집니다.

---

## 3. 정합성 검증 결과 (Correctness Results)

### 3.1 기능 시뮬레이션 (Spike Functional Mode)

[test_correctness.py](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/tests/test_correctness.py)를 Docker 환경에서 실행한 결과, **11개 전체 테스트가 통과**했습니다.

```
================ Summary: [11/11] Passed ================
```

| # | 테스트 케이스 | 입력 형상 | 검증 연산 | 상태 |
| :---: | :--- | :--- | :--- | :---: |
| 1 | Vector Add 1D | `[2]` × 3 | 벡터 덧셈 | ✅ Pass |
| 2 | Matrix Add 2D | `[2,2]` × 3 | 행렬 덧셈 | ✅ Pass |
| 3 | Vector Mul 2D | `[2,2]` × 2 | 원소별 곱셈 | ✅ Pass |
| 4 | Matmul 32×32 | `[32,32]` × 2 | 행렬 곱셈 | ✅ Pass |
| 5 | Linear Bias 32×32 | `[32,32]` + `[32]` | Matmul + Bias | ✅ Pass |
| 6 | ReLU 32×32 | `[32,32]` | 활성화 함수 | ✅ Pass |
| 7 | GEMM 16×32×64 | `[16,32]` × `[32,64]` | 비정방 행렬곱 | ✅ Pass |
| 8 | Conv2D 14×14 | `[1,14,14,8]` × `[3,3,8,16]` | 2D 합성곱 | ✅ Pass |
| 9 | **Perceptron 8×32→16** | `[8,32]` × `[32,16]` + `[16]` | Matmul + Bias + ReLU | ✅ Pass |
| 10 | **Sigmoid Linear 8×32→16** | `[8,32]` × `[32,16]` + `[16]` | Matmul + Bias + Sigmoid | ✅ Pass |
| 11 | **Large Matmul 64×128×32** | `[64,128]` × `[128,32]` | 대형 텐서 행렬곱 | ✅ Pass |

### 3.2 검증 파이프라인 흐름

```
TensorFlow 함수 (@tf.function)
    │
    ▼
StableHLO IR (XLA 컴파일)
    │  stablehlo-opt: --stablehlo-legalize-to-linalg
    ▼
Linalg/Memref IR
    │  mlir-opt: --one-shot-bufferize --convert-bufferization-to-memref
    ▼
tf-mlir.mlir (버퍼화된 Memref IR)
    │  transform_tf_mlir(): 함수명/시그니처/레이아웃 변환
    ▼
PyTorchSim 호환 MLIR (@kernel, 1D flat memref, void 반환)
    │  TENSORFLOW_MLIR_DIRECT_TEST=True → MLIRCodeCache.load
    ▼
RISC-V 바이너리
    │  Spike 시뮬레이터
    ▼
NPU 실행 결과 ←→ TF CPU 참조값 비교
```

### 3.3 타이밍 시뮬레이션 (TOGSim Timing Mode)

`TENSORFLOW_MLIR_DIRECT_TEST=False` (PyTorch 자체 MLIR 사용)로 타이밍 모드를 검증한 결과, TOGSim이 정상적으로 사이클 시뮬레이션을 수행합니다:

```
[TOGSim] TOGSim simulation started
[TOGSim] Simulation log is stored to ".../togsim_results/20260605_051742_36d9849c.log"
```

타이밍 트레이스 출력 (`*.trace`):
```
LAUNCH_KERNEL,0,0,0,/workspace/PyTorchSim/outputs/wchkx2ccxqs/tile_graph.onnx,...
```

> [!WARNING]
> TF MLIR을 직접 사용하는 경우(`TENSORFLOW_MLIR_DIRECT_TEST=True`), HW 특화 패스(`-dma-fine-grained`, `-test-pytorchsim-to-vcix`, `-test-tile-operation-graph`)가 바이패스되므로 타일 연산 그래프(TOG)가 생성되지 않습니다. 따라서 TF MLIR 기반 타이밍 시뮬레이션에는 `tf_npu_codegen.py`의 NPU 코드 생성기 통합이 필요합니다.

---

## 4. 발견된 제한 사항 (Limitations)

### 4.1 단일 커널 제약 (Single-Kernel Constraint)

현재 `TENSORFLOW_MLIR_DIRECT_TEST` 메커니즘은 하나의 `tf-mlir.mlir` 파일을 모든 컴파일 커널에 대입합니다. 그러나 `torch.compile`이 복합 모델(예: 2층 MLP)을 **여러 커널**로 분할하면, 각 커널이 동일한 전체 함수 MLIR을 받게 되어 인자 불일치 → 세그멘테이션 폴트가 발생합니다.

**영향받는 모델 유형:**
- 2개 이상의 `matmul` 연산을 포함하는 다층 네트워크 (MLP, Transformer 등)
- `torch.compile`이 Template Kernel로 분리하는 모든 복합 연산

**해결 방향:**
- 커널별 MLIR 파일 매핑 구현 (per-kernel MLIR substitution)
- 또는 `tf_npu_codegen.py` 통합으로 단일 모놀리식 MLIR → 단일 커널 컴파일

### 4.2 StableHLO 미지원 연산

| 미지원 패턴 | 원인 | 비고 |
| :--- | :--- | :--- |
| `tf.matmul(q, k, transpose_b=True)` | `--linalg-specialize-generic-ops`에서 transpose matmul 형상 검증 실패 | 명시적 전치 후 표준 matmul 사용으로 우회 가능 |
| `tf.nn.softmax` | `stablehlo.reduce` → linalg 로어링 시 복잡한 reduce 패턴 미지원 | 향후 커스텀 lowering pass 필요 |
| `tf.cast(tf.shape(...), ...)` | 동적 형상 연산 (`get_dimension_size`) 미지원 | 상수로 대체하여 우회 |

### 4.3 DRAM 전용 실행

기능 시뮬레이션에서 TF MLIR은 표준 `linalg` → `loops` → RISC-V 벡터/스칼라 경로로 컴파일되어 DRAM에서 직접 동작합니다. SRAM 타일링이 적용되지 않으므로:
- NPU 시스톨릭 어레이의 정확한 사이클 프로파일을 반영하지 못함
- `memref.alloc`으로 생성된 중간 버퍼에 대한 SRAM 할당이 없음

---

## 5. 이전 보고서 대비 변경점 (Changes from v1)

| 항목 | v1 | v2 |
| :--- | :--- | :--- |
| 테스트 수 | 6개 (기본 연산) | **11개** (기본 + Perceptron + Sigmoid + Large Matmul) |
| 코드 구조 | `extension_codecache.py` 인라인 | **`TensorFlowFrontend` 패키지로 분리** |
| NPU 코드 생성 | 없음 | **`tf_npu_codegen.py` 초안** (DMA/SRAM 타일링) |
| 타이밍 모드 | 미검증 | **TOGSim 동작 확인** (PyTorch MLIR 기준) |
| 복합 모델 | 미포함 | Perceptron ✅, Sigmoid Linear ✅, 다층 MLP ❌ (단일 커널 제약) |

---

## 6. 향후 과제 (Future Works)

1. **NPU 코드 생성기 통합**: `tf_npu_codegen.py`를 `transform_tf_mlir` 파이프라인에 연결하여 TF MLIR에서도 SRAM/DMA 타일링된 MLIR을 생성, Gem5 타이밍 시뮬레이션 활성화
2. **다중 커널 지원**: 커널별 MLIR 매핑 메커니즘을 구현하여 MLP, Transformer 등 다층 모델의 정합성 검증 지원
3. **StableHLO 확장**: Transpose matmul, Softmax, 동적 형상 연산에 대한 커스텀 lowering pass 구현
4. **자동 타일링 패스**: 표준 `linalg.matmul`을 DMA 기반 SRAM 타일링 루프로 자동 변환하는 MLIR pass 구현
5. **다중 코어/배치 지원**: 단일 코어 이외의 멀티코어 실행 및 배치 처리 검증
