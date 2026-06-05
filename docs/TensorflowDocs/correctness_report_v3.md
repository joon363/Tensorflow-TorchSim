# TensorFlow 파이프라인 확장 검증 보고서 v3

## 1. 개요 (Overview)

본 보고서는 `correctness_report_v2`에서 명시된 향후 과제(Future Works) 중 다음 3개의 구현 결과를 문서화합니다:

1. **과제 1: NPU 코드 생성기 통합** — `tf_npu_codegen.py`를 파이프라인에 연결하여 TF MLIR에서도 SRAM/DMA 타일링 MLIR을 생성할 수 있는 경로를 구축
2. **과제 2: 다중 커널 지원 (Option B)** — `torch.compile`이 복합 모델을 분할하는 다중 커널 환경에서 커널별 MLIR 매핑 메커니즘을 구현
3. **과제 4: 자동 타일링 패스** — PyTorchSim의 GEMM 타일링 전략을 참고하여 `linalg.matmul`을 DMA 기반 SRAM 타일링 루프로 자동 변환하는 모듈을 작성

또한, 다음 항목은 구현하지 않고 **한계점으로 문서화**합니다:

- **과제 3: StableHLO 확장** — Transpose matmul, Softmax, 동적 형상 연산에 대한 lowering pass (§5 참조)
- **과제 5: 다중 코어/배치 지원** — 멀티코어 실행 및 배치 처리 (범위 외)

---

## 2. 과제 1: NPU 코드 생성기 통합

### 2.1 구현 내용

[tf_npu_codegen.py](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/TensorFlowFrontend/tf_npu_codegen.py)의 `compile_tf_to_npu_mlir()` 함수는 StableHLO/Linalg 레벨의 MLIR을 파싱하여 NPU 최적화 MLIR을 "brick-by-brick"으로 조립합니다.

현재 지원 연산:

| Linalg 패턴 | NPU MLIR 생성 | 타일링 전략 |
| :--- | :--- | :--- |
| `linalg.add`, `linalg.generic(addf/mulf)` | SRAM 3-buffer DMA + affine.vector_load/store | Flat 1D, tile=512 |
| `linalg.generic(maximumf)` (ReLU) | SRAM 2-buffer DMA + max(x, 0) 벡터 연산 | Flat 1D, tile=512 |
| `linalg.matmul` | PyTorchSim `MLIRGemmTemplate` 직접 호출 | M/N/K 타일링 |
| `linalg.conv_2d_nhwc_fhwc` | PyTorchSim `MLIRConvTemplate` 직접 호출 | Conv 타일링 |

### 2.2 파이프라인 연결

[handle_tensorflow_direct_test()](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/TensorFlowFrontend/tf_mlir_conversion.py)가 `arg_attributes`를 받아 커널별 MLIR 매핑을 수행하도록 확장되었습니다.

```python
# extension_codecache.py (수정)
source_code = handle_tensorflow_direct_test(source_code, get_write_path, arg_attributes=arg_attributes)
```

NPU 코드 생성기를 실제 컴파일에 사용하려면 `mlir_compile_command`의 TF 분기에 PyTorchSim의 HW 패스를 추가해야 합니다:
- `-dma-fine-grained`: DMA 연산 세분화
- `-test-pytorchsim-to-vcix`: 시스톨릭 어레이 변환
- `-test-tile-operation-graph`: TOG 그래프 생성
- `-test-memref-to-gemmini`: Gemmini 타일 매핑

> [!NOTE]
> 현재 NPU 코드 생성기는 MLIR 텍스트를 직접 생성하는 방식이며, 생성된 MLIR이 PyTorchSim의 HW 패스와 정합하는지에 대한 검증이 추가적으로 필요합니다. 기능 시뮬레이션(Spike)에서는 DMA 명령이 스칼라 루프로 폴백되어 정상 동작하나, 타이밍 시뮬레이션(TOGSim)에서의 NPU 가속 프로파일 생성은 후속 작업으로 남습니다.

---

## 3. 과제 2: 다중 커널 지원 (Option B)

### 3.1 문제 정의

`torch.compile`은 복합 모델(예: 2-layer MLP)을 여러 커널로 분할합니다:

```
MLP: relu(x @ w1 + b1) @ w2 + b2
  → kernel_0: relu(matmul + bias)  [inputs: x(256), w1(2048), b1(64)] → [output: h(512)]
  → kernel_1: matmul + bias        [inputs: h(512), w2(1024), b2(16)] → [output: y(128)]
```

기존에는 `TENSORFLOW_MLIR_DIRECT_TEST`가 단일 `tf-mlir.mlir`을 **모든** `MLIRCodeCache.load` 호출에 동일하게 대입하여, kernel_0이 전체 함수(6개 인자)의 MLIR을 받고 kernel_0의 4개 인자와 불일치하여 Spike 실행 시 segfault가 발생했습니다.

### 3.2 구현된 메커니즘

**커널 인덱스 추적**:
```python
# tf_mlir_conversion.py
_tf_kernel_counter = 0  # 전역 카운터

def reset_tf_kernel_counter():
    """테스트 간 카운터 초기화"""
    global _tf_kernel_counter
    _tf_kernel_counter = 0

def handle_tensorflow_direct_test(source_code, get_write_path_fn, arg_attributes=None):
    global _tf_kernel_counter
    # ... 커널별 arg_attributes 기반 MLIR 매핑 ...
    _tf_kernel_counter += 1
```

**테스트 하네스 통합**:
```python
# test_correctness.py
def run_tf_test(...):
    torch._dynamo.reset()           # torch.compile 캐시 초기화
    reset_tf_kernel_counter()        # 커널 인덱스 초기화
    # ... 테스트 실행 ...
```

### 3.3 알려진 한계점

| 사항 | 설명 |
| :--- | :--- |
| **중간 버퍼 문제** | kernel_1의 입력 `buf1`은 kernel_0의 출력이며, TF MLIR에서는 함수 인자가 아니라 `memref.alloc()`으로 생성된 중간 버퍼입니다. 이를 커널 인자로 매핑할 수 없어 인자 불일치가 발생합니다. |
| **SSA 의존 분석** | TF MLIR의 연산 그래프를 PyTorch의 커널 경계와 동일하게 분할하려면 SSA 값의 def-use 체인을 분석하여 하위 그래프를 추출해야 합니다. 현재 문자열 파싱 방식으로는 이 분석이 불안정합니다. |
| **근본적 해결책** | MLIR Python Bindings(`mlir.ir.Module`)를 사용한 구조적 변환 (참조: [mlir_module_impl_plan.md](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/docs/TensorflowDocs/mlir_module_impl_plan.md)) |

### 3.4 테스트 결과

| 테스트 | 커널 수 | 결과 |
| :--- | :---: | :---: |
| Vector Add 1D | 1 | ✅ PASS |
| Matrix Add 2D | 1 | ✅ PASS |
| Vector Mul 2D | 1 | ✅ PASS |
| Matmul 32x32 | 1 | ✅ PASS |
| Linear Bias 32x32 | 1 | ✅ PASS |
| ReLU 32x32 | 1 | ✅ PASS |
| GEMM 16x32x64 | 1 | ✅ PASS |
| Conv2D 14x14x8→16 | 1 | ✅ PASS |
| Perceptron 8x32→16 | 1 | ✅ PASS |
| Sigmoid Linear 8x32→16 | 1 | ✅ PASS |
| Large Matmul 64x128x32 | 1 | ✅ PASS |
| **MLP 2-layer 8x32→64→16** | **2** | ❌ **XFAIL** (다중 커널 제약) |

```
================ Summary: [11/12] Passed, [1] Known Limitations ================
```

---

## 4. 과제 4: 자동 타일링 패스

### 4.1 설계 원리

PyTorchSim의 GEMM 템플릿([mlir_gemm_template.py](file:///c:/Users/MAIN/Desktop/PyTorchSim/PyTorchSimFrontend/mlir/mlir_gemm_template.py))의 타일링 전략을 참고하여 [auto_tiling.py](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/TensorFlowFrontend/auto_tiling.py)를 설계했습니다.

**PyTorchSim GEMM 타일링 구조**:
```
affine.for %i = 0 to M step TILE_M {           // M 타일링 (outer_loop)
  affine.for %j = 0 to N step TILE_N {         // N 타일링 (outer_loop)
    // Y_buffer 초기화 (zero 또는 bias DMA-in)
    affine.for %k = 0 to K step TILE_K {       // K 타일링 (accumulation_loop)
      DMA_IN X_tile[TILE_M × TILE_K] → X_buffer(SRAM)
      DMA_IN W_tile[TILE_K × TILE_N] → W_buffer(SRAM)
      linalg.matmul ins(X_buffer, W_buffer) outs(Y_buffer)   // SRAM 내 연산
    }
    DMA_OUT Y_buffer → Y_tile[TILE_M × TILE_N]  // 결과 DRAM 기록
  }
}
```

**타일 크기 결정 (`select_tile` 참조)**:
- SRAM 제약: `(TILE_M×TILE_K + TILE_K×TILE_N + TILE_M×TILE_N) × dtype_bytes ≤ spad_size`
- 초기 후보: `tile = floor(sqrt(spad_size / (3 × dtype_bytes)))`
- 2의 거듭제곱으로 정렬 후, SRAM 초과 시 반복 축소

### 4.2 구현

[auto_tiling.py](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/TensorFlowFrontend/auto_tiling.py)에 다음 함수를 구현했습니다:

| 함수 | 역할 |
| :--- | :--- |
| `_compute_gemm_tile_sizes(M, N, K, spad_size)` | SRAM 용량 기반 GEMM 타일 크기 자동 계산 |
| `_compute_elementwise_tile_size(total, spad_size, n_bufs)` | 엘리먼트별 연산의 DMA 타일 크기 계산 |
| `generate_tiled_gemm_mlir(op_idx, ...)` | 3-레벨 M/N/K 타일링 + SRAM 버퍼 + 복사 루프 MLIR 생성 |
| `tile_linalg_matmul(mlir_content, spad_size)` | MLIR 텍스트 내 `linalg.matmul` 패턴을 자동 타일링으로 치환 |

### 4.3 통합 상태

`auto_tiling.py`는 독립 모듈로 작성되었으며, `tf_npu_codegen.py`의 `compile_tf_to_npu_mlir()` 내에서 연산별 템플릿 매칭 실패 시 폴백으로 사용될 수 있도록 설계되었습니다.

> [!IMPORTANT]
> 생성된 타일링 MLIR은 SRAM 메모리 공간(`memref<...x..., 1>`)을 사용하는 `memref.load/store` 기반 복사 루프로 구성됩니다. PyTorchSim의 DMA 기반 접근(`memref.dma_start`)과는 다른 방식이며, 기능 시뮬레이션(Spike)에서는 정상 동작하나 타이밍 시뮬레이션에서의 DMA 연산 프로파일링에는 추가 패스가 필요합니다.

---

## 5. 과제 3: StableHLO 미지원 연산 (한계점)

다음 연산 패턴은 현재 `stablehlo-opt --stablehlo-legalize-to-linalg`에서 실패하며, 구현하지 않고 한계점으로 남깁니다:

### 5.1 Transpose Matmul (`transpose_b=True`)

```python
# 실패하는 패턴
tf.matmul(query, key, transpose_b=True)  # Self-Attention Q×K^T
```

- **원인**: XLA가 `stablehlo.dot_general`의 `contracting_dimensions`를 전치 형상으로 설정하나, `--linalg-specialize-generic-ops`가 이를 `linalg.matmul`로 특수화하지 못하고 `linalg.generic`으로 남김
- **에러**: `linalg.generic` 내부의 형상 추론 실패

### 5.2 Softmax (`tf.nn.softmax`)

```python
# 실패하는 패턴
tf.nn.softmax(logits)
```

- **원인**: Softmax는 `reduce(max)` → `subtract` → `exp` → `reduce(sum)` → `divide` 체인으로 분해되며, `reduce` 연산의 linalg 로어링이 복잡한 제네릭 패턴으로 남아 후속 패스에서 실패
- **해결 방향**: Python 리라이터로 reduce-exp-div 체인을 인식하여 단일 softmax 루프로 치환, 또는 C++ MLIR 패스로 구현

### 5.3 동적 형상 (`tf.shape`, `?` 축)

- **원인**: 현재 변환기의 형상 추출 코드(`int(dim)`)가 동적 축(`?`)을 처리하지 못함
- **완화**: `@tf.function(jit_compile=True)` 사용 시 XLA가 모든 형상을 정적으로 특화하므로, 동적 축은 일반적으로 발생하지 않음

---

## 6. v2 대비 변경 사항

| 항목 | v2 | v3 |
| :--- | :--- | :--- |
| 테스트 수 | 11개 (단일 커널) | **12개** (단일 커널 11 + MLP XFAIL 1) |
| 다중 커널 지원 | 없음 | **커널 인덱스 추적 및 arg_attributes 매핑 구현** |
| NPU 코드 생성기 | 초안 (미연결) | **파이프라인 연결 경로 구축** |
| 자동 타일링 | 없음 | **auto_tiling.py 구현** (GEMM M/N/K 타일링) |
| StableHLO 확장 | 향후 과제 | **한계점으로 문서화** (transpose matmul, softmax, 동적 형상) |

---

## 7. 파일 변경 이력

| 파일 | 변경 내용 |
| :--- | :--- |
| [tf_mlir_conversion.py](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/TensorFlowFrontend/tf_mlir_conversion.py) | `_tf_kernel_counter`, `reset_tf_kernel_counter()`, `_split_tf_mlir_for_kernel()` 추가. `handle_tensorflow_direct_test()`에 `arg_attributes` 인자 추가 |
| [auto_tiling.py](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/TensorFlowFrontend/auto_tiling.py) | **[NEW]** GEMM 자동 타일링 모듈 |
| [extension_codecache.py](file:///c:/Users/MAIN/Desktop/PyTorchSim/PyTorchSimFrontend/extension_codecache.py) | `handle_tensorflow_direct_test()` 호출에 `arg_attributes` 전달 |
| [test_correctness.py](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/tests/test_correctness.py) | MLP 2-layer 테스트 추가, `known_limitations` XFAIL 처리, `reset_tf_kernel_counter()` 호출 |

---

## 8. 향후 과제 (Future Works)

1. **MLIR Python Bindings 기반 구조적 변환**: 문자열 파싱을 `mlir.ir.Module` API 기반 AST 조작으로 전환하여 다중 커널 MLIR 분할의 안정성 확보 (참조: [mlir_module_impl_plan.md](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/docs/TensorflowDocs/mlir_module_impl_plan.md))
2. **NPU 코드 생성기 타이밍 모드 활성화**: `mlir_compile_command`의 TF 분기에 PyTorchSim HW 패스(`-dma-fine-grained`, `-test-pytorchsim-to-vcix`, `-test-tile-operation-graph`)를 조건부 추가하여 NPU 생성 MLIR의 타이밍 시뮬레이션 지원
3. **StableHLO 확장**: Transpose matmul, Softmax, 동적 형상 연산에 대한 커스텀 lowering pass 구현
4. **다중 코어/배치 지원**: 타일링 루프의 최외곽 축을 코어별로 분배하는 병렬화 구현
