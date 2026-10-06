"""Run real ComfyUI Krea2 integration on CPU; missing dependencies/skips fail."""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('comfy_root', help='ComfyUI root; run using its Python environment')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    comfy = Path(args.comfy_root).resolve()
    if not (comfy / 'comfy/ldm/krea2/model.py').is_file():
        parser.error('This ComfyUI checkout does not contain the native Krea2 model')
    try:
        revision = subprocess.check_output(['git', '-C', str(comfy), 'rev-parse', 'HEAD'],
                                           text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = 'unknown'
    sys.path[:0] = [str(comfy), str(project)]
    sys.argv = [sys.argv[0], '--cpu']
    try:
        import comfy.options
        comfy.options.enable_args_parsing()
        importlib.import_module('comfy.ldm.krea2.model')
        suite = unittest.defaultTestLoader.discover(str(project / 'tests'), pattern='test_regional_comfy.py')
        if unittest.defaultTestLoader.errors:
            raise RuntimeError('\n'.join(unittest.defaultTestLoader.errors))
        expected = 8
        if suite.countTestCases() != expected:
            raise RuntimeError(f'Expected {expected} native integration tests; collected {suite.countTestCases()}')
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        passed = result.wasSuccessful() and result.testsRun == expected and not result.skipped
        report = {'status': 'passed' if passed else 'failed', 'comfy_commit': revision,
                  'tests_run': result.testsRun, 'skipped': len(result.skipped),
                  'phase': 'prompt_only', 'int8_validated': False}
    except Exception as error:
        passed = False
        report = {'status': 'failed', 'comfy_commit': revision,
                  'error_type': type(error).__name__, 'error': str(error)}
    print(json.dumps(report, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
