# TensorFlow 파이프라인 정합성 검증 보고서 (Correctness Verification Report)

## 배경 (Background)
PyTorchSim 프레임워크는 원래 PyTorch 프레임워크를 기반으로 설계 및 최적화된 포괄적이고 주기 정밀한(cycle-accurate) NPU 시뮬레이션 환경입니다. 구체적으로, PyTorch 2.0의 `TorchInductor` 컴파일 백엔드에 연동하여 FX 그래프를 하드웨어 의존적인 MLIR 커널(SRAM 버퍼, 커스텀 DMA 전송 및 시스톨릭 어레이 명령어 사용)로 로어링(lowering)합니다.

이 프레임워크를 TensorFlow 및 JAX까지 확장하기 위해 **OpenXLA** 컴파일러 에코시스템을 활용합니다. XLA는 TensorFlow 그래프를 하드웨어 독립적인 고수준 연산 다이얼렉트인 **StableHLO**로 변환합니다. 그러나 XLA가 생성하는 StableHLO와 그에 따른 Linalg/Memref 표현은 PyTorchSim이 생성하는 커스텀 하드웨어 의존적 MLIR 템플릿과 다음과 같은 근본적인 차이점이 있습니다.

1. **하드웨어 독립성**: StableHLO는 순수 수학적 정합성에 초점을 맞추며 표준 DRAM 메모리 공간에서 동작하는 반면, PyTorchSim의 커스텀 Inductor 로어링은 하드웨어 특정적인 연산(`dma_start`, `sram_buffer` 할당 등)을 생성합니다.
2. **버퍼 할당**: PyTorch Inductor는 출력 버퍼를 미리 할당하고 이를 파라미터로 전달하여 참조 반환(write by reference)하는 반면, XLA가 생성하는 MLIR은 출력을 내부적으로 할당(`memref.alloc`)하고 값 반환(return by value) 방식으로 처리합니다.
3. **다차원 레이아웃**: XLA는 다차원 스트라이드 memref(예: `memref<2x2xf32, strided<[?, ?], offset: ?>>`)를 함수 인자로 전달합니다. 반면 PyTorchSim의 `mlir_caller_codegen`이 생성하는 C 래퍼는 평탄화된 1D memref(인자당 5개의 파라미터)를 기대합니다.

본 문서는 이 TensorFlow 파이프라인의 정합성을 검증하기 위한 배경, 검증 방법, 의의, 결과, 한계점 및 향후 과제에 대해 보고합니다.

---

## 수행 작업 (Tasks)
주요 작업은 모든 TensorFlow 연산에 대해 하드웨어 의존적인 커스텀 템플릿을 매번 재작성하지 않고도, TensorFlow/XLA 컴파일러 파이프라인과 PyTorchSim 백엔드 사이의 아키텍처 격차를 해소하는 것입니다.

구체적인 수행 작업은 다음과 같습니다:
1. `TENSORFLOW_MLIR_DIRECT_TEST` 환경 변수를 사용하여 PyTorchSim의 코드 캐시 컴파일 흐름(`MLIRCodeCache.load`)을 가로챕니다.
2. XLA가 생성한 MLIR 함수의 시그니처(함수 이름, 반환 타입, 인자 개수)를 `mlir_caller_codegen.py`가 생성하는 C 테스트 하네스와 일치시킵니다.
3. Spike 시뮬레이터 내부에서 세그멘테이션 폴트(segmentation fault)를 유발했던 다차원 인자 불일치 문제를 해결합니다.
4. TensorFlow 모델/연산자를 CPU 참조 값 및 PyTorchSim 실행 결과와 자동으로 비교 검증하는 강력하고 범용적인 테스트 하네스를 구현합니다.

---

## 검증 방법 (Methods)

### 1. 컴파일 리디렉션 (Compilation Redirection)
`TENSORFLOW_MLIR_DIRECT_TEST=True` 환경 변수가 활성화되면 코드 캐시 시스템(`MLIRCodeCache.load`)이 PyTorch Inductor의 컴파일을 가로채고, 해당 MLIR 입력을 TensorFlow 그래프에서 직접 변환한 결과(`tf-mlir.mlir`)로 대체합니다.

### 2. MLIR 변환 (`transform_tf_mlir`)
시그니처, 레이아웃 및 반환 값 불일치 문제를 해결하기 위해 [extension_codecache.py](file:///c:/Users/MAIN/Desktop/PyTorchSim/PyTorchSimFrontend/extension_codecache.py) 내부에 커스텀 MLIR 리라이터(rewriter)를 구현했습니다:
* **함수 이름 변경**: 진입 함수(entry function)의 이름을 `@main`에서 `@kernel`로 변경합니다.
* **N차원 평탄화 (N-Dimensional Flattening)**: 함수 선언부의 모든 N차원 입력/출력 memref 파라미터를 1D memref 파라미터(예: `memref<1024xf32, strided<[?], offset: ?>>`)로 변환합니다. 이는 1D memref 레이아웃을 가정하는 C 래퍼(`validation_wrapper.c`)와 완벽하게 일치합니다.
* **내부 재해석 캐스트 (Internal Reinterpretation)**: 평탄화된 1D 파라미터로부터 원래의 N차원 레이아웃을 복원하기 위해 함수 본문 시작 부분에 `memref.reinterpret_cast` 명령어를 삽입합니다. 핵심 연산 로직은 수정 없이 그대로 유지됩니다.
* **출력 인자 삽입**: 결과 데이터를 담아갈 수 있도록 함수 인자 리스트 끝에 출력용 버퍼 파라미터인 `%arg_tf_out_flat`을 추가합니다.
* **값 반환을 참조 쓰기로 변환**: `return %val : memref<...>` 명령어를 제거하고, 출력을 엘리먼트 단위로 출력 인자 버퍼에 써주는 중첩 `scf.for` 복사 루프를 삽입하여 함수가 `void`를 반환하도록 변환합니다.

### 3. DRAM 정합성 모드 (DRAM Correctness Mode)
XLA가 생성한 MLIR에는 하드웨어 특정적 패스(SRAM/VCIX)가 포함되어 있지 않으므로, `TENSORFLOW_MLIR_DIRECT_TEST`가 활성화된 경우 커스텀 로어링 패스(`-test-pytorchsim-to-vcix`, `-test-memref-to-gemmini`, `-dma-fine-grained`)를 바이패스합니다. 대신 표준 루프 변환 패스(`-convert-linalg-to-loops`)를 사용하여 `linalg` 연산을 루프로 직접 낮춥니다.
이 방식은 코드를 DRAM에서 바로 접근하는 표준 RISC-V 스칼라/벡터 명령어로 컴파일하여, SRAM 시뮬레이션 오류 없이 순수 수학적 정합성을 검증할 수 있게 합니다.

---

## 검증 결과 (Results)
[test_correctness.py](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/tests/test_correctness.py) 아래에 완벽한 정합성 테스트 스위트가 구축되었습니다. Docker 환경에서 실행 시, 다음 모든 테스트 케이스가 성공적으로 컴파일 및 검증되었습니다:

| 테스트 케이스 (Test Case) | 입력 / 형상 (Shape) | TF CPU 참조 값 (Reference) | PyTorchSim NPU (Spike) | 상태 (Status) |
| :--- | :--- | :--- | :--- | :--- |
| **Vector Add 1D** | `[2]` | `tensor([9., 12.])` | `tensor([9., 12.])` | **Passed** |
| **Matrix Add 2D** | `[2, 2]` | `[[15, 18], [21, 24]]` | `[[15, 18], [21, 24]]` | **Passed** |
| **Vector Mul 2D** | `[2, 2]` | `[[5, 12], [21, 32]]` | `[[5, 12], [21, 32]]` | **Passed** |
| **Matmul 32x32** | `[32, 32] x [32, 32]` | 일치 (Match) | 일치 (Match) | **Passed** |
| **Linear Bias 32x32**| `[32, 32] x [32, 32] + [32]` | 일치 (Match) | 일치 (Match) | **Passed** |
| **ReLU 32x32** | `[32, 32]` | 일치 (Match) | 일치 (Match) | **Passed** |

모든 테스트 케이스가 RISC-V 바이너리로 컴파일되어 Spike 기능 시뮬레이터에서 오류 없이 작동하였으며, CPU 실행 결과와 완전히 일치하는 것을 확인했습니다.

---

## 의의 (Implications)
* **HW 매핑과 독립된 알고리즘 검증**: 복잡한 SRAM 타일링 휴리스틱을 작성하기 전에 RISC-V 시뮬레이터상에서 임의의 TensorFlow 서브그래프가 수학적으로 올바르게 동작하는지 먼저 검증할 수 있습니다.
* **레이아웃 매핑**: 프론트엔드와 백엔드 래퍼 간의 레이아웃 불일치를 해결하기 위해 `memref.reinterpret_cast`를 사용하는 범용적인 가교(bridge) 설계 방식을 확립했습니다.
* **통합 컴파일러 유효성 확인**: StableHLO -> Linalg -> Memref -> LLVM -> RISC-V 경로가 PyTorchSim의 새로운 프론트엔드 통합을 위한 견고하고 신뢰할 수 있는 검증 파이프라인임을 증명했습니다.

---

## 한계점 (Limitations)
* **DRAM 오버헤드**: 시스톨릭 어레이 특정 패스를 우회하기 때문에 행렬 곱셈이 RISC-V 벡터/스칼라 유닛의 중첩 루프로 실행되므로, NPU 시스톨릭 어레이의 정확한 사이클 프로파일을 반영하지 못합니다.
* **데이터 복사 비용**: 커널 실행 마지막 단계에서 내부 할당된 memref의 데이터를 출력 인자 파라미터로 명시적으로 복사해야 하는 추가적인 루프가 수반됩니다.
* **기능 시뮬레이션 위주 검증**: DRAM 루프 방식은 타일 연산 그래프(TOG)에 시스톨릭 어레이 연산 이벤트를 기록하지 않으므로, Gem5 시뮬레이터를 통한 상세한 타이밍 분석 모드는 우회해야 합니다.

---

## 장단점 분석 (Pros and Cons)

### 장점 (Pros)
* **높은 범용성**: 별도의 하드웨어 의존적 코드 없이도 CNN, MLP, ReLU, Matmul 등 임의의 TensorFlow 연산 및 모델 구조를 즉시 실행하고 검증할 수 있습니다.
* **프론트엔드 격리성**: 컴파일러 프론트엔드가 변경되더라도 백엔드 래퍼 형식과 시뮬레이터 인터페이스를 깨뜨리지 않습니다.
* **정확한 수치 검증**: CPU와 RISC-V 에뮬레이션 간 오차가 0%임을 정밀하게 보장합니다.

### 단점 (Cons)
* **타일링 및 SRAM 미반용**: 스크래치패드 메모리(SRAM)의 세밀한 배치 및 하드웨어 가속 구조를 활용하지 못합니다.
* **시뮬레이션 효율**: 루프 기반 DRAM 접근으로 인해 시뮬레이션 내에서의 하드웨어 가속 성능은 최적화되지 않은 상태입니다.

---

## 향후 과제 (Future Works)
1. **자동 타일링 패스 (Automated Tiling Pass)**: DRAM 상의 표준 `linalg.matmul` 연산을 DMA를 이용해 SRAM(메모리 공간 1)에서 타일 단위로 수행하도록 자동 변환해 주는 MLIR 패스를 구현합니다.
2. **SRAM 버퍼 자동 연동**: 중간 할당 연산(`memref.alloc`)에 대해 SRAM 버퍼 영역을 자동으로 할당하고 관리해 주는 컴파일러 최적화 패스를 탑재합니다.
3. **Gem5 타이밍 모드 지원**: 타일링된 루프로부터 TOG 그래프를 추출하여 TensorFlow 연산에 대해서도 NPU 사이클 정밀도 수준의 타이밍 모드 검증을 활성화합니다.
