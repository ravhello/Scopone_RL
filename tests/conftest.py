# Keep this file minimal. GPU-dependent tests will self-skip via decorators.
import pytest  # noqa: F401
import os


@pytest.fixture(autouse=True, scope="session")
def force_cpu_for_tests():
    # Force CPU during test session unless explicitly overridden
    os.environ.setdefault('TESTS_FORCE_CPU', '1')
    os.environ.setdefault('SCOPONE_DEVICE', 'cpu')
    os.environ.setdefault('OBS_DEVICE', 'cpu')
    os.environ.setdefault('ACTIONS_DEVICE', 'cpu')
    os.environ.setdefault('ENV_DEVICE', 'cpu')
    os.environ.setdefault('REW_DEVICE', 'cpu')
    yield
