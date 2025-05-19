import os
import signal
import subprocess
import resource
import faulthandler
from tempfile import TemporaryDirectory
import platform

from verl.utils.reward_score.rewards.code_utils.utils import BASE_IMPORTS

CLI_ARG_SIZE_LIMIT = 1024 * 3

_ERROR_MSG_PREFIX = "Failed to execute program: "
_DEFAULT_TIMEOUT_SECONDS = 30


def code_exec(code, test: str = None, timeout=_DEFAULT_TIMEOUT_SECONDS):
    env = os.environ.copy()
    
    # Create a preexec_fn function to set resource limits
    def preexec_fn():
        reliability_guard()

    if 'pytest' not in code:

        code_to_run = f"""
{BASE_IMPORTS}
import pytest

{code}

{test}

if __name__ == "__main__":
    pytest.main([__file__])
"""
    else:
        code_to_run = code
    
    # solution is in {tmpdir}/solution.py
    with TemporaryDirectory() as tmpdir:
        # Write the solution to a file

        solution_path = os.path.join(tmpdir, "solution.py")

        with open(solution_path, "w") as f:
            f.write(code_to_run)
            
        command = ["pytest", "--maxfail=1", solution_path]
        
        try:
            result = subprocess.run(
                command,
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                check=False,
                preexec_fn=preexec_fn,
                timeout=timeout
            )
            
            stderr = result.stderr.decode().strip()
            stdout = result.stdout.decode()
            if result.returncode == 0:
                return True, stdout
            return False, _ERROR_MSG_PREFIX + f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
            
        except subprocess.TimeoutExpired:
            return False, _ERROR_MSG_PREFIX + f"Execution timed out after {timeout} seconds."
        except Exception as e:
            return False, _ERROR_MSG_PREFIX + f"An Exception occurred in the code: {str(e)}"


def reliability_guard(maximum_memory_bytes=None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)
    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None
