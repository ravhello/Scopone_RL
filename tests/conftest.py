# Keep this file minimal. GPU-dependent tests will self-skip via decorators.
import pytest  # noqa: F401
import os
import warnings

# Silence noisy third-party deprecation chatter that does not impact behavior.
warnings.filterwarnings(
    "ignore",
    message=r"Type google\.protobuf\.pyext\._message\..*PyType_Spec",
    category=DeprecationWarning,
)
# Extra guard: silence the same warning via module regex in case message changes slightly.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r".*google\.protobuf\.pyext\._message",
)


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
