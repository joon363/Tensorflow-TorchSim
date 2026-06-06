import os
import sys
import contextlib

@contextlib.contextmanager
def suppress_c_logs():
    """
    Redirects C-level stdout and stderr (file descriptors 1 and 2) to /dev/null,
    but keeps Python's sys.stdout and sys.stderr working by mapping them to 
    the original file descriptors. This suppresses messy C++ logs (Spike, TOGSim, XLA)
    while allowing Python's print() to still output to the console.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    
    null_fd = os.open(os.devnull, os.O_RDWR)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    
    os.dup2(null_fd, 1)
    os.dup2(null_fd, 2)
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    # We duplicate again because fdopen takes ownership of the fd and closes it
    new_stdout_fd = os.dup(saved_stdout_fd)
    new_stderr_fd = os.dup(saved_stderr_fd)
    
    sys.stdout = os.fdopen(new_stdout_fd, 'w', buffering=1)
    sys.stderr = os.fdopen(new_stderr_fd, 'w', buffering=1)
    
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(null_fd)

import test_correctness
import test_timing
import plot_timing
from tests_common import get_all_tests

def main():
    print("================================================================================")
    print("         INTEGRATED VERIFICATION: Correctness & Timing")
    print("================================================================================")
    
    tests = get_all_tests()
    
    print("\n[Phase 1] Correctness Testing")
    with suppress_c_logs():
        success = test_correctness.run_all_correctness_tests(tests)
    
    if not success:
        print("\nCorrectness tests failed. Stopping integrated verification.")
        sys.exit(1)
        
    print("\n[Phase 2] Timing Simulation")
    with suppress_c_logs():
        test_timing.run_all_timing_tests(tests)
        
    print("\n[Phase 3] Generating Graphs")
    with suppress_c_logs():
        plot_timing.plot_results()
        
    print("\nAll tests passed successfully and artifacts generated in outputs/ directory.")

if __name__ == "__main__":
    main()
