import sys, os
import warnings

# Ensure project root is importable for tests (environment.py, actions.py, etc.)
ROOT = os.path.abspath(os.getcwd())
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Suppress DeprecationWarning from pygame.pkgdata (pytest noise)
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r".*pygame\.pkgdata")


