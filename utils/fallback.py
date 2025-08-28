import os
import sys
import traceback
from typing import Optional


class FallbackUsedError(RuntimeError):
    pass


def _should_raise() -> bool:
    # Always raise: strict mode enforced globally
    return True


def _should_trace() -> bool:
    # Keep stack traces enabled unless explicitly disabled
    return os.environ.get('SCOPONE_FALLBACK_TRACE', '1') != '0'


def notify_fallback(key: str, details: Optional[str] = None, *, raise_error: Optional[bool] = None) -> None:
    """Signal that a fallback path has been taken.

    key: short identifier for the fallback site (e.g., 'ppo.select_action.uniform_probs')
    details: optional human-readable extra info
    raise_error: override raising behavior for this call
    """
    # Enforce raising regardless of caller-provided override
    mode_raise = True
    msg = f"[FALLBACK] {key}"
    if details:
        msg = f"{msg}: {details}"
    if mode_raise:
        if _should_trace():
            # Include the stack in the exception message for easier debugging
            tb = ''.join(traceback.format_stack(limit=12))
            msg = f"{msg}\nStack (most recent call last):\n{tb}"
        raise FallbackUsedError(msg)
    # Print to stderr so it's visible in logs even without logging setup
    try:
        sys.stderr.write(msg + "\n")
        if _should_trace():
            traceback.print_stack(limit=12, file=sys.stderr)
    except Exception:
        # Last resort: silent failure of the notifier should not affect program flow
        pass


