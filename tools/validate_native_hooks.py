"""Run native ComfyUI Hook regression tests on CPU, without starting a server."""
import argparse
import json
import os
from pathlib import Path
import sys
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comfy_root")
    parser.add_argument("--baseline", action="store_true",
                        help="Run against unmodified native Hooks (expected to fail)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(Path(args.comfy_root).resolve()), str(root)]
    os.environ["KREA2_HOOK_TEST_BASELINE"] = "1" if args.baseline else "0"
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    suite = unittest.defaultTestLoader.discover(str(root / "tests"), pattern="test_native_hooks.py")
    expected_tests = 11
    if suite.countTestCases() != expected_tests:
        print("Expected all 11 native Hook regression tests to be collected", file=sys.stderr)
        return 1
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    passed = result.wasSuccessful() and result.testsRun == expected_tests
    if not args.baseline:
        passed = passed and not result.skipped
    print(json.dumps({"status": "passed" if passed else "failed", "tests_run": result.testsRun,
                      "skipped": len(result.skipped), "baseline": args.baseline}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
