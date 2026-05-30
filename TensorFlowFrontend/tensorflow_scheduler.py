import logging
import sys
import importlib.util
import sys
from pathlib import Path

from torch.fx.experimental.sym_node import magic_methods, method_to_operator
from torch._inductor.graph import GraphLowering
DUMP_DIR = Path("/workspace/PyTorchSim/TensorFlow/tests/dump")

log = logging.getLogger(__name__)


def is_magic_method(op):
    magic_ops = {method_to_operator(m) for m in magic_methods}
    return op in magic_ops
class MyScheduler(GraphLowering):
    def compile_to_module(self):
        code, _ = self.codegen()

        # 1. 디렉토리 보장
        DUMP_DIR.mkdir(parents=True, exist_ok=True)

        # 2. 파일 경로
        module_path = DUMP_DIR / "generated_module.py"
        module_path.write_text(code)

        # 3. 모듈 로드
        spec = importlib.util.spec_from_file_location(
            "generated_module", module_path
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        log.debug("Output code written to: %s", mod.__file__)
        log.debug("Output code: \n%s", code)

        return mod
    
    def codegen(self):
        from torch._inductor.scheduler import Scheduler

        self.init_wrapper_code()

        self.scheduler = Scheduler(self.buffers)
        self.scheduler.codegen()
        return self.wrapper_code.generate(self.is_inference)