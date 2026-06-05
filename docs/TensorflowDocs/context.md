# TensorFlow-PyTorchSim 통합 프로젝트 컨텍스트

## 1. 프로젝트 목표 (Project Goal)

PyTorchSim은 POSTECH PSAL Lab에서 개발한 **사이클 정밀(cycle-accurate) NPU 시뮬레이터**로, PyTorch 2.0의 TorchInductor 컴파일 백엔드에 연동하여 FX 그래프를 하드웨어 의존적인 MLIR 커널로 로어링합니다.

본 프로젝트의 목표는 **TensorFlow/JAX 생태계를 PyTorchSim에 통합**하는 것입니다. OpenXLA 컴파일러가 생성하는 **StableHLO** 중간 표현을 가교(bridge)로 삼아, 별도의 하드웨어 백엔드를 재구현하지 않고도 TensorFlow 모델을 PyTorchSim의 Spike(기능) 및 TOGSim(타이밍) 시뮬레이터에서 실행 및 검증할 수 있도록 합니다.

---

## 2. 디렉토리 구조 (Directory Structure)

```
Tensorflow/
├── TensorFlowFrontend/        # TF 전용 파이프라인 로직
├── PluggableDevice/           # TF PluggableDevice 커스텀 디바이스 (C++ bazel)
├── binaries/                  # MLIR 툴체인 바이너리
├── tests/                     # 정합성 테스트 스위트
│   ├── test_correctness.py    # 메인 테스트 하네스
│   ├── dev/                   # 개발/실험용 스크립트
│   ├── out/                   # 컴파일 중간 산출물 (gitignored)
│   └── togsim_results/        # 타이밍 시뮬레이션 로그 (gitignored)
├── docs/                      # 문서
│   ├── TensorflowDocs/        # TF 파이프라인 문서
│   ├── PyTorchSimDocs/        # 원본 PyTorchSim 분석 문서
│   ├── Reports/               # 연구 제안서/경과 보고서
│   └── workspace_setup.md     # 환경 구축 가이드
└── util/                      # LLM 프롬프트/출력 보관
```

---

## 3. 소스 파일 및 역할 (Files and Usages)

### 3.1 핵심 소스 (Core Source Files)

| 파일 | 역할 |
| :--- | :--- |
| [extension_codecache.py](file:///c:/Users/MAIN/Desktop/PyTorchSim/PyTorchSimFrontend/extension_codecache.py) | **PyTorchSim의 컴파일 캐시 진입점.** `MLIRCodeCache.load`에서 MLIR → LLVM IR → RISC-V 컴파일, Spike/TOGSim 실행을 담당. `TENSORFLOW_MLIR_DIRECT_TEST=True` 시 TF MLIR 파일로 대치하는 리디렉션 로직 포함. `mlir_compile_command` (기능용)와 `mlir_gem5_compile_command` (타이밍용) 두 가지 컴파일 파이프라인을 정의 |
| [tf_mlir_conversion.py](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/TensorFlowFrontend/tf_mlir_conversion.py) | **TF MLIR 변환기.** `transform_tf_mlir()`: TF/XLA가 생성한 Memref MLIR을 PyTorchSim 호환 형식으로 변환 (함수명 `@main`→`@kernel`, N차원→1D 평탄화, `memref.reinterpret_cast` 삽입, `return`→`scf.for` 복사 루프, `linalg.map`→`linalg.fill`, `memref.copy`→루프 치환). `handle_tensorflow_direct_test()`: MLIR 파일 경로 리디렉션 및 헤더 복사 |
| [tf_npu_codegen.py](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/TensorFlowFrontend/tf_npu_codegen.py) | **NPU 최적화 코드 생성기 (초안).** StableHLO/Linalg MLIR을 파싱하여 DMA/SRAM 기반 타일링 MLIR을 "brick-by-brick"으로 조립. `MockNode`/`MockLayout` (PyTorch Inductor 메타데이터 모킹), `generate_elementwise_npu()`, `generate_relu_npu()`, `generate_gemm_kernel()`, `generate_conv_kernel()`, `compile_tf_to_npu_mlir()` (통합 파서/컴파일러) 포함 |
| [auto_tiling.py](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/TensorFlowFrontend/auto_tiling.py) | **자동 타일링 모듈.** PyTorchSim GEMM 템플릿의 M/N/K 타일링 전략을 참고하여 `linalg.matmul`을 SRAM 기반 타일 루프로 자동 변환. `_compute_gemm_tile_sizes()`, `generate_tiled_gemm_mlir()`, `tile_linalg_matmul()` 포함 |
| [test_correctness.py](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/tests/test_correctness.py) | **정합성 테스트 하네스.** `tf_to_MLIR()`: TF 함수 → StableHLO → Linalg → Memref 변환. `run_tf_test()`: TF CPU 참조값 vs PyTorchSim NPU(Spike) 결과 자동 비교. 12개 테스트 케이스 (11 PASS + 1 XFAIL), `known_limitations` XFAIL 처리, JSON 로그 출력 |

### 3.2 환경/설정 파일

| 파일 | 역할 |
| :--- | :--- |
| [extension_config.py](file:///c:/Users/MAIN/Desktop/PyTorchSim/PyTorchSimFrontend/extension_config.py) | PyTorchSim 전역 설정. `pytorchsim_functional_mode`, `pytorchsim_timing_mode` 플래그를 TOGSIM_CONFIG YAML에서 로드 |
| `togsim_config_functional_only.yml` | 기능 시뮬레이션 전용 설정 (`pytorchsim_functional_mode: 1, pytorchsim_timing_mode: 0`) |
| `togsim_config_timing_only.yml` | 타이밍 시뮬레이션 전용 설정 (`pytorchsim_functional_mode: 0, pytorchsim_timing_mode: 1`) |

### 3.3 MLIR 툴체인 바이너리 (`binaries/`)

| 바이너리 | 역할 |
| :--- | :--- |
| `stablehlo-opt` | StableHLO → Linalg 변환 (`--stablehlo-legalize-to-linalg`, `--linalg-specialize-generic-ops`) |
| `mlir-opt` | Linalg → Memref 버퍼화 (`-one-shot-bufferize`), 루프 변환 (`-convert-linalg-to-loops`), LLVM 로어링 |
| `mlir-translate` | LLVM Dialect → LLVM IR 변환 (`-mlir-to-llvmir`) |
| `stablehlo-lsp-server` | StableHLO LSP 서버 (IDE 지원용) |

### 3.4 PluggableDevice (`PluggableDevice/`)

TensorFlow Modular PluggableDevice 인터페이스의 C++ 구현. Bazel 빌드 기반. 커스텀 NPU 디바이스(`my_device`)를 TF에 등록하여 `tf.device("MY_DEVICE")` 형태로 사용 가능. 현재 연구 초기 단계이며, 본 프로젝트에서는 XLA의 StableHLO 출력을 직접 가져오는 방식을 우선 사용.

---

## 4. 생성 파일 및 산출물 (Generated Files)

### 4.1 테스트 실행 시 생성되는 파일 (`tests/out/`)

| 파일 | 생성 주체 | 내용 |
| :--- | :--- | :--- |
| `0_input_stablehlo.mlir` | `test_correctness.py` → `tf_to_MLIR()` | TF `@tf.function`의 XLA 컴파일 결과 (StableHLO IR) |
| `1_stablehlo_clean.mlir` | `stablehlo-opt` | Linalg/Tensor로 변환된 MLIR |
| `tf-mlir.mlir` | `mlir-opt` | 최종 버퍼화된 Memref MLIR. `TENSORFLOW_MLIR_DIRECT_TEST` 시 이 파일이 PyTorchSim에 주입됨 |
| `compilation.log` | `test_correctness.py` | 모든 컴파일 명령어와 stdout/stderr 기록 |
| `test_results.json` | `test_correctness.py` | 테스트 결과 요약 JSON (테스트별 PASS/FAIL, 에러 정보) |

### 4.2 Wrapper Codegen (PyTorchSim이 생성하는 Python 파일)

`torch.compile`이 TorchInductor를 통해 생성하는 Python 래퍼 파일. 경로: `outputs/.torchinductor/<hash>/<hash>.py`

구성 요소:
1. **`arg_attributes` 리스트**: 각 인자의 `[이름, [입출력 방향, dtype, 원소 수, shape, stride]]`를 정의
2. **`extension_kernel_N`**: `custom_async_compile.mlir(...)` 호출로 MLIR 커널을 등록. MLIR 텍스트, vectorlane 크기, SRAM 설정 등을 인자로 전달
3. **`call(args)` 함수**: 입력 텐서의 크기/스트라이드 검증(`assert_size_stride`), 출력 버퍼 할당(`empty`), 커널 실행, SRAM 할당/해제 관리(`sram_plan_prefix/postfix`)

> [!NOTE]
> TF 파이프라인에서는 이 래퍼가 PyTorch Inductor에 의해 자동 생성되며, `TENSORFLOW_MLIR_DIRECT_TEST=True` 시 내부의 MLIR 텍스트만 TF MLIR로 대치됩니다.

### 4.3 Spike/TOGSim 실행 시 생성되는 파일 (`outputs/<hash>/`)

| 파일 | 생성 주체 | 내용 |
| :--- | :--- | :--- |
| `<hash>.mlir` | `MLIRCodeCache.load` | 컴파일 대상 MLIR 커널 |
| `validation_wrapper.c` | `mlir_caller_codegen.py` | C 테스트 하네스: raw 파일에서 텐서를 읽어 커널 호출 후 결과를 raw 파일로 출력 |
| `validation_binary` | `riscv-gcc` 크로스 컴파일 | RISC-V 실행 바이너리 |
| `runtime_NNNN/argN_M/0.raw` | PyTorchSim | 입력 텐서의 바이너리 데이터 |
| `runtime_NNNN/bufN/0.raw` | Spike 실행 후 | 출력 텐서의 바이너리 데이터 |
| `tile_graph.onnx` | `-test-tile-operation-graph` 패스 | 타이밍 모드에서 사용하는 타일 연산 그래프 |

### 4.4 TOGSim 결과 (`tests/togsim_results/`)

| 파일 | 내용 |
| :--- | :--- |
| `<timestamp>_<hash>.trace` | TOGSim 입력 트레이스: `LAUNCH_KERNEL,core,layer,batch,tile_graph_path,attribute_path,0` |
| `<timestamp>_<hash>.log` | TOGSim 시뮬레이션 로그: NPU 구성, DRAM 통계, 사이클 수 등 |

---

## 5. 문서 안내 (Documents Guide)

### 5.1 `TensorflowDocs/` — TF 파이프라인 문서

| 문서 | 내용 |
| :--- | :--- |
| [context.md](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/docs/TensorflowDocs/context.md) | (본 문서) 프로젝트 전체 컨텍스트 |
| [correctness_report.md](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/docs/TensorflowDocs/correctness_report.md) | v1 정합성 검증 보고서: 6개 기본 테스트 (Add, Mul, Matmul, Linear Bias, ReLU)의 배경, 방법, 결과, 의의, 한계, 향후 과제 |
| [correctness_report_conv_gemm.md](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/docs/TensorflowDocs/correctness_report_conv_gemm.md) | Conv2D 및 GEMM 고차원 연산 검증 보고서: `linalg.map` arity mismatch, `memref.subview/copy` 미지원 문제 해결 기록 |
| [correctness_report_v2.md](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/docs/TensorflowDocs/correctness_report_v2.md) | v2 정합성 보고서: 11개 테스트(Perceptron, Sigmoid Linear, Large Matmul 추가), 모듈 분리, NPU 코드 생성기 초안, 타이밍 모드 검증, 향후 과제 |
| [correctness_report_v3.md](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/docs/TensorflowDocs/correctness_report_v3.md) | v3 정합성 보고서: NPU 코드 생성기 통합, 다중 커널 지원(Option B + XFAIL), 자동 타일링 모듈, StableHLO 미지원 연산 한계점 문서화 |
| [limitation_tf.md](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/docs/TensorflowDocs/limitation_tf.md) | 문자열/정규식 기반 MLIR 파싱의 기술적 한계점 분석: 서식 취약성, 인자 파싱 불안정성, 이름 충돌, 동적 형상 미지원, 시맨틱 검증 결여 |
| [mlirInfo.md](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/docs/TensorflowDocs/mlirInfo.md) | MLIR Python Bindings(`mlir.ir`) 빠른 참조: Context 생성, `ir.Module.parse`, `module.walk()`, Def-Use 체인, 워크플로우 스니펫 |
| [mlir_module_impl_plan.md](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/docs/TensorflowDocs/mlir_module_impl_plan.md) | MLIR Python API 활용 구조적 변환 계획서: 문자열 파싱 → `mlir.ir.Module` 기반 AST 조작으로의 전환 설계. `ReinterpretCastOp`, `scf.ForOp` 빌드 코드 예시 포함 |
| [TensorFlowPluginTutorial.md](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/docs/TensorflowDocs/TensorFlowPluginTurorial.md) | TensorFlow PluggableDevice 플러그인 개발 튜토리얼: Device Runtime, Kernel/Op 등록, Graph Optimization, Profiler, 빌드/설치/실행 방법 |

### 5.2 `PyTorchSimDocs/` — 원본 PyTorchSim 분석

| 문서 | 내용 |
| :--- | :--- |
| [EntryPoints.md](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/docs/PyTorchSimDocs/EntryPoints.md) | `torch.compile` → FX Graph → TorchInductor IR → MLIR → RISC-V → Spike/TOGSim 전체 호출 체인의 단계별 진입점(EntryPoint) 기록 |
| [exampleTorchOutput.py](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/docs/PyTorchSimDocs/exampleTorchOutput.py) | PyTorchSim의 Wrapper Codegen이 생성하는 Python 파일의 실제 예시 (1024×1024 행렬 덧셈) |
| [torchtomlir.md](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/docs/PyTorchSimDocs/torchtomlir/torchtomlir.md) | PyTorch → MLIR 변환의 상세 분석: TorchInductor 코드 생성, CSEProxy, MLIR Kernel codegen, SRAM/DMA 템플릿 구조 |
| `pytorchsim_mlir_*.mlir` | PyTorchSim이 생성하는 MLIR 커널의 참조 예시 (add, matmul) |
| `tensorflow_stablehlo_add.mlir` | TF/XLA가 생성하는 StableHLO IR 참조 예시 (add) |

### 5.3 `Reports/` — 연구 보고서

| 문서 | 내용 |
| :--- | :--- |
| [proposal.md](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/docs/Reports/proposal.md) | 연구 제안서: HLO 중간 표현 기반 TensorFlow-PyTorchSim 통합 연구 목적, 배경, 방법론 |
| [progress.md](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/docs/Reports/progress.md) | 연구 경과 보고서 |

### 5.4 기타

| 문서 | 내용 |
| :--- | :--- |
| [workspace_setup.md](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/docs/workspace_setup.md) | 개발 환경 구축 가이드: LLVM 22.1.0 바이너리 다운로드, StableHLO 빌드 방법, 필수 바이너리 경로 |
| [README.md](file:///c:/Users/MAIN/Desktop/PyTorchSim/Tensorflow/README.md) | Docker 컨테이너 실행, 볼륨 마운트, 바이너리 권한 설정, 테스트 재현 명령어 |

---

## 6. 현재 구현 상태 (Current Implementation)

### 6.1 완료된 사항

1. **StableHLO 기반 MLIR 변환 파이프라인**: TF `@tf.function` → StableHLO → Linalg → Memref → PyTorchSim 호환 MLIR
2. **MLIR 시그니처 변환기** (`transform_tf_mlir`): N차원 평탄화, reinterpret_cast, 출력 인자 삽입, 반환→복사 루프 변환
3. **컴파일러 호환 패치**: `linalg.map` → `linalg.fill`, `memref.copy` → `scf.for` 루프, `-expand-strided-metadata` 패스 추가
4. **11개 정합성 테스트 통과** (Spike 기능 시뮬레이션): Vector Add, Matrix Add, Mul, Matmul, Linear Bias, ReLU, GEMM, Conv2D, Perceptron, Sigmoid Linear, Large Matmul
5. **TOGSim 타이밍 모드 검증**: PyTorch MLIR 기준 사이클 시뮬레이션 동작 확인
6. **모듈 분리**: TF 전용 로직을 `TensorFlowFrontend` 패키지로 추출, `extension_codecache.py` 수정 최소화

### 6.2 알려진 제한 사항

- **단일 커널 제약**: `torch.compile`이 2개 이상의 커널로 분할하는 복합 모델(MLP, Transformer)은 `tf-mlir.mlir` 대치 메커니즘과 호환 불가
- **StableHLO 미지원 연산**: `transpose_b` matmul, `softmax`, 동적 형상(`tf.shape`) 등이 `stablehlo-opt` 로어링에서 실패
- **DRAM 전용**: TF MLIR은 SRAM/DMA 패스를 우회하므로 타이밍 시뮬레이션에서 NPU 가속 프로파일 미반영

---

## 7. 향후 과제 (Future Works)

1. **NPU 코드 생성기 통합**: `tf_npu_codegen.py`를 컴파일 파이프라인에 연결 → TF MLIR에서도 SRAM/DMA 타일링 MLIR 생성
2. **다중 커널 지원**: 커널별 MLIR 매핑으로 MLP/Transformer 등 다층 모델 정합성 검증
3. **StableHLO 확장**: Transpose matmul, Softmax, 동적 형상 연산용 커스텀 lowering pass
4. **자동 타일링 패스**: `linalg.matmul` → DMA 기반 SRAM 타일링 루프 자동 변환
5. **다중 코어/배치 지원**: 멀티코어 실행 및 배치 처리 검증
