You are a technical specialist tasked with understanding the existing codebase to answer a specific user query or prepare for a feature implementation.

# PyTorchSim: TensorFlow Integration Strategy Document

본 프로젝트는 기존 **PyTorch 기반 NPU 시뮬레이터(PyTorchSim)** 의 백엔드 자산을 최대한 재사용하면서, **TensorFlow(TFRT/XLA)** 모델을 지원하기 위한 아키텍처 분석 및 구현 로드맵을 기술한다. 

본 문서의 목적은 그 중에서 가장 첫번째 단계인 Tensorflow의 output과 PyTorchSim의 output을 비교하고 분석하는 데에 있다.


## 1. 현재 구현 상태 (As-Is)

현재 PyTorchSim은 **PyTorch 2.0 (Inductor)** 기반의 정적 컴파일 파이프라인을 구축한 상태이다.

* **Frontend:** `TorchDynamo`를 통해 파이썬 코드를 **FX Graph** 로 캡처 (Static Graph).
* **Intermediate Representation (IR):** `Aten Ops` (PyTorch 표준 연산자).
* **Backend Core (`mlir_codegen_backend.py`):**
* **Hand-written Kernel Generator:** 하드웨어 특성을 반영한 템플릿(`MLIRConvTemplate` 등)을 보유.
* **`ExtensionOverrides` (`mlir_ops.py`):** `ops.add`, `ops.mul` 등 추상 연산을 RISC-V 특화 MLIR 텍스트로 1:1 매핑.
* **Auto-tuning:** 타일링(Tiling) 사이즈 등을 결정하는 최적화 로직 포함.


* **Execution:** 생성된 MLIR을 RISC-V 바이너리로 컴파일하여 `TOGSimulator`에서 실행.

> **핵심 자산:** FX Graph 자체가 아니라, 고수준 연산(Ops)을 받아 하드웨어 최적화된 MLIR로 변환해주는 **Python Codegen Engine**이 프로젝트의 핵심이다.

---

## 2. TensorFlow Anatomy (Target Architecture)

TensorFlow를 통합하기 위해 이해해야 할 실행 구조와 확장 포인트는 다음과 같다.

* **Execution Flow:**
`Python Code` -> `AutoGraph` -> **`GraphDef`** -> `XLA` -> `HLO` -> **`StableHLO`** -> `TFRT/StreamExecutor` -> `Device`
* **StableHLO (The Bridge):**
    * TensorFlow/JAX의 표준 IR.
    * PyTorch의 `Aten Ops`와 유사한 추상화 레벨을 가짐 (예: `stablehlo.convolution`  `aten.conv2d`).

* **PluggableDevice (The Interface):**
    * Google이 제공하는 Modular TensorFlow 인터페이스. Cpp로 작성되어 있다.
    * **`src/device`:** 메모리 할당 및 실행 제어 (필수).
    * **`src/graph` (Optimizer):** 그래프 실행 전 전체 구조를 조작할 수 있는 훅(Hook). **여기가 JIT 컴파일의 진입점이다.**
    * **`src/kernels`:** 개별 연산 구현체. (JIT 컴파일 방식에서는 사용하지 않음).

---

## 3. 목표 및 전략 (To-Be)

- TensorFlow의 모델을 실행하되, **Eager Mode(인터프리터 방식)가 아닌 Graph Mode**로 실행하여 시뮬레이터의 성능(Fusion, Tiling)을 극대화한다.


## 4. 상세 분석: 3가지 MLIR 변환 경로 비교

### 4.1 Stage 1: TensorFlow→StableHLO (via XLA)

**입력 코드:**
```python
@tf.function(jit_compile=True)
def add_fn(x, y):
    return x + y

x = tf.constant([1.0, 2.0], tf.float32)
y = tf.constant([3.0, 4.0], tf.float32)
```

**Stablehlo Compiler Command**
```python
ir = add_fn.experimental_get_compiler_ir(x, y)(stage="stablehlo")
```

```bash
stablehlo-opt ir --stablehlo-target-independent-optimization 
```

**StableHLO 출력:**
```mlir
module @a_inference_add_fn_8__.9 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<2xf32>
    return %0 : tensor<2xf32>
  }
}
```

**특징 분석:**
1. **추상화 수준:** Very High - Torch와 **동일함**
2. **타입 정보:** `tensor<2xf32>` - **Shape 정보 완벽 보존**
3. **결론:** TensorFlow XLA도 동일한 StableHLO를 생성 (이는 StableHLO가 프레임워크 독립적 IR임을 보여줌)

---

### 4.3 Stage 2: PyTorchSim Hand-Written MLIR Kernel

**입력:** FX Graph에서 추출한 `aten.add` 연산 하나

**생성된 MLIR 코드:**
```mlir
memref.global @buf0_spad : memref<2xf32, 1>
memref.global @buf1_spad : memref<2xf32, 1>
memref.global @buf2_spad : memref<2xf32, 1>
func.func @kernel(%in_ptr0: memref<2xf32>,
                       %in_ptr1: memref<2xf32>,
                       %out_ptr0: memref<2xf32>)
{
    %const0 = arith.constant 0 : index
    %const1 = arith.constant 2 : index
    %const2 = arith.constant 3 : index
    %alloc0 = memref.alloc() : memref<1xi32>      // DMA token 0
    %alloc1 = memref.alloc() : memref<1xi32>      // DMA token 1
    %alloc2 = memref.alloc() : memref<1xi32>      // DMA token 2
    %spad0 = memref.get_global @buf0_spad : memref<2xf32, 1>
    %spad1 = memref.get_global @buf1_spad : memref<2xf32, 1>
    %spad2 = memref.get_global @buf2_spad : memref<2xf32, 1>
    
    // Outer loop: iterate over tiles
    affine.for %index0 = 0 to 2 step 2
    {
        // DMA LOAD: arg0 → spad0
        memref.dma_start %in_ptr0[%index0], %spad0[%const0], %const1, %alloc0[%const0], %const0, %const1 : 
          memref<2xf32>, memref<2xf32, 1>, memref<1xi32> {dram_stride=[1], sram_stride=[1], padding=0}
        
        // DMA LOAD: arg1 → spad1
        memref.dma_start %in_ptr1[%index0], %spad1[%const0], %const1, %alloc1[%const0], %const0, %const1 : 
          memref<2xf32>, memref<2xf32, 1>, memref<1xi32> {dram_stride=[1], sram_stride=[1], padding=0}
        
        // Inner loop: vectorized computation
        affine.for %compute_idx = 0 to 2 step 2
        {
            %tmp0 = affine.vector_load %spad0[%compute_idx] : memref<2xf32, 1>, vector<2xf32>
            %tmp1 = affine.vector_load %spad1[%compute_idx] : memref<2xf32, 1>, vector<2xf32>
            %tmp2 = arith.addf %tmp0, %tmp1 : vector<2xf32>
            affine.vector_store %tmp2, %spad2[%compute_idx] : memref<2xf32, 1>, vector<2xf32>
        } {inner_loop=false}
        
        // DMA STORE: spad2 → output
        memref.dma_start %spad2[%const0], %out_ptr0[%index0], %const2, %alloc2[%const0], %const0, %const1 : 
          memref<2xf32, 1>, memref<2xf32>, memref<1xi32> {dram_stride=[1], sram_stride=[1], padding=0}
    } {outer_loop=true}
    return
}
```

**특징 분석:**
1. **추상화 수준:** Very Low - 하드웨어 매우 구체적
2. **메모리 모델:** 명시적 계층
   - **DRAM** (HBM): `memref<2xf32>` (memory space 0)
   - **SPAD** (Scratch Pad): `memref<2xf32, 1>` (memory space 1)
3. **컴퓨팅 구조:**
   - **Outer Loop** (`affine.for`): 타일 단위 반복 (0 to 2 step 2)
   - **Inner Loop** (`affine.for`): 벡터화된 연산 (0 to 2 step 2)
4. **DMA 명시:** `memref.dma_start`로 명시적 데이터 이동
   - **Attributes:** `dram_stride`, `sram_stride`, `padding` 정보 포함
5. **벡터화:** `vector<2xf32>` (2-element vectors)
6. **메타데이터:** 루프 속성 (`{outer_loop=true}`, `{inner_loop=false}`)

---

## 5. 3단계 비교 분석표

| 항목 | **StableHLO (Torch/TF)** | **PyTorchSim Hand-Written** |
|------|--------------------------|------------------------------|
| **추상화 수준** | Very High (Pure Ops) | Very Low (HW-specific) |
| **타입** | `tensor<2xf32>` | `memref<2xf32, 1>` (memory space) |
| **메모리 계층** | 없음 (암시적) | DRAM, SPAD 명시 ✓ |
| **루프 구조** | 없음 | `affine.for` (명시적 타일링) |
| **DMA/데이터 이동** | 없음 | `memref.dma_start` ✓ |
| **벡터화** | 없음 | `vector<Nxf32>` ✓ |
| **Shape 정보** | `tensor<2xf32>` ✓ | `memref<2xf32, 1>` ✓ |
| **루프 메타데이터** | 없음 | `{outer_loop=true}` 등 ✓ |

---

## 6. 핵심 발견사항 (Key Findings)

### PyTorchSim의 추가 정보

PyTorchSim이 StableHLO로부터 새로 만들어야 하는 정보:
- ✓ 루프 구조 (Outer/Inner)
- ✓ 타일 크기 (step=2)
- ✓ SPAD 메모리 할당 및 매핑
- ✓ DMA 명령 및 스트라이드 정보
- ✓ 벡터화 전략
**이 내용은 mlir_codegen_backend.py의 codegen_loops에서 시행하고 있음**