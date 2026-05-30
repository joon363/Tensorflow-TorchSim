# PyTorchSim: 포괄적이고 정밀한 NPU 시뮬레이션 및 컴파일러 프레임워크

## 1. 프로젝트 개요 (Overview)

**PyTorchSim**은 자체 개발 중인 NPU(Neural Processing Unit) 하드웨어를 위해 설계된 전용 시뮬레이션 및 컴파일러 스택 프로젝트입니다. PyTorch 2.0의 최신 컴파일 엔진과 통합되어, 딥러닝 모델을 하드웨어 특화 명령어로 변환하고 성능을 사이클 단위로 정밀하게 검증합니다.

### 1.1 프로젝트 정체성

- **핵심 경로:** PyTorch → FX Graph -> TorchInductor IR → Custom MLIR Kernel → Custom NPU ISA (RISC-V 확장)
- **목적:** 딥러닝 컴파일 파이프라인 실험, 커스텀 하드웨어 가속기 성능 시뮬레이션, SRAM 관리 및 타일링 전략 검증.
- **차별점:** 기존 시뮬레이터 대비 압도적인 속도(TLS 기술)와 PyTorch 2.0과의 긴밀한 통합.

## 2. 시스템 아키텍처 (Architecture)

프로젝트는 프론트엔드, 미들엔드, 백엔드의 3단계 계층 구조로 설계되었습니다.

### Layer 1: Frontend & Graph Capture

- **PyTorch Inductor Backend:** `torch.compile` 인터페이스를 통해 FX Graph를 캡처하고 Aten IR로 정규화합니다.
- **FX Graph & Aten IR**: 상위 프레임워크의 연산을 PyTorch의 표준 연산 셋인 Aten IR로 정규화합니다.

### Layer 2: Codegen Engine (Middle-end)

* 이 엔진은 프로젝트의 핵심으로, 상위 IR을 하드웨어 가속기용 코드로 변환하는 역할을 수행합니다.
* **Torch FX Graph** 를 순회하면서 FX Node의 종류에 따라 `placeholder`, `get_attr`, `call_function`, `call_module`, `call_method`, `output`가 결정됩니다.
* 이후 `call_function`의 경우 Aten IR을 담고 있고, TorchInductor IR로 Lowering 됩니다 (여기까지는 PyTorchSim의 역할이 아님)
* 그 다음 Lowering된 `ops.mul`과 같은 형태의 노드에 따라 custom MLIR 코드를 생성합니다. 이는 `mlir_codegen_backend.py`에 정의되어 있습니다. 특히 `load`, `indirect_load`, `store_reduction`, `store`, `reduction`, `bucketize`는 `MLIRKernel`에 정의되어 있고 나머지 수학 연산 등에 대해서는 `ExtensionOverrides`에 정의되어 있습니다.
* **Tiling & Mapping:** 연산을 하드웨어 리소스에 맞게 매핑하고, 데이터 흐름을 최적화합니다.
- **Custom Kernels:** Convolution, GEMM 등 핵심 연산에 대해 하드웨어 최적화된 핸드라이팅(Hand-written) 커널을 제공합니다.

### Layer 3: Simulator Backend

- **ILS (Instruction-Level Simulation):** Gem5(타이밍)와 Spike(기능)를 결합하여 명령어 단위로 정밀 시뮬레이션합니다.
- **TLS (Tile-Level Simulation):** 반복되는 타일 연산의 지연 시간을 오프라인에서 측정하여 시뮬레이션 속도를 비약적으로 높입니다.
- **TOGSim:** 생성된 TOG(Tile Operation Graph)를 실행하며 DRAM 및 Interconnect의 경합을 모델링합니다.

## 3. 핵심 기술 세부사항

### 3.1 ISA 확장 및 하드웨어 모델

- **RISC-V 기반 확장:** 표준 RISC-V 벡터 확장(RVV)에 DMA 및 Systolic Array 제어를 위한 커스텀 명령어를 추가했습니다.
- **VCIX Interface:** SiFive의 Vector Coprocessor Interface를 채택하여 벡터 레지스터 파일(VRF)과 데이터플로우 유닛 간의 병목 현상을 제거했습니다.
- **SRAM 관리:** SW가 관리하는 Scratchpad Memory를 통해 캐시 미스에 의한 예측 불가능한 지연 시간을 방지합니다.

### 3.2 TOG (Tile Operation Graph)

시뮬레이션 속도 향상을 위한 핵심 데이터 구조입니다.

- **구성 노드:** `loopBegin`, `loopEnd`, `compute`, `loadDMA`, `storeDMA`, `waitDMA`.
- **특징:** 연산 간의 의존성을 DAG(Directed Acyclic Graph)로 표현하며, 연산(Compute)과 데이터 전송(DMA)의 오버랩을 정밀하게 제어합니다.


## 4. 주요 소프트웨어 컴포넌트

- **Scheduler (`PyTorchSimRunner`):** 연산을 시뮬레이션 디바이스로 라우팅하고 실행 순서를 통제하는 런타임 추상화 계층입니다.

사용 예:

```python
from Scheduler.scheduler import PyTorchSimRunner
device = PyTorchSimRunner.setup_device().custom_device()
```

역할:

* PyTorch에서 **custom device**를 등록
* 연산을 **실제 GPU/CPU가 아닌 시뮬레이션 디바이스**로 라우팅
* 추후:

  * 연산 스케줄링
  * 메모리 배치
  * 실행 순서 통제

즉, **Runtime abstraction layer**.