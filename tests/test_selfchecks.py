#!/usr/bin/env python3
"""Quick self-checks for a NeuroVF checkout.

    python tests/test_selfchecks.py        (or: pytest tests)

1. Both applications compile.
2. NeuroVF's built-in consistency checks (run_td_validation_tests) pass in both applications.
3. The oracle self-test (oracle/selftest_oracle.py) passes.

These checks take a minute or two and need no display. They do not replace the full numerical validation workflow
(run_validation_imported.bat), which compares every exported number from the reviewed imported databases with the oracle.
"""
import importlib.util
import os
import py_compile
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APPS = [os.path.join(ROOT, "app", name) for name in ("NeuroVF.py", "NeuroVF_research.py")]


def _load(path):
    spec = importlib.util.spec_from_file_location("neurovf_under_test_" + os.path.basename(path)[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_applications_compile():
    for path in APPS:
        py_compile.compile(path, doraise=True)


def test_builtin_consistency_checks():
    for path in APPS:
        assert _load(path).run_td_validation_tests(verbose=False), f"built-in checks failed in {os.path.basename(path)}"


def test_oracle_selftest():
    result = subprocess.run([sys.executable, os.path.join(ROOT, "oracle", "selftest_oracle.py")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


if __name__ == "__main__":
    failed = 0
    for test in (test_applications_compile, test_builtin_consistency_checks, test_oracle_selftest):
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL  {test.__name__}: {exc}")
    sys.exit(1 if failed else 0)


