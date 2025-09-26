import os
import sys
import argparse
import importlib
import inspect
from typing import List

# Ensure project root on sys.path for module imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Abilita di default feature dell'osservazione (dealer one-hot)
os.environ.setdefault('OBS_INCLUDE_DEALER', '1')


def _aggregate_and_print_line_profiler_results(global_profiler, project_root: str, extended_report: bool = False) -> None:
    """Replicates the per-file and top-lines summaries used in tools/profile_ppo.py."""
    global_profiler.print_stats(sort_by='cpu')
    if extended_report:
        print(global_profiler.generate_report(include_line_details=True))

    from collections import defaultdict

    def relpath(p):
        ap = os.path.abspath(p)
        if project_root in ap:
            return os.path.relpath(ap, project_root)
        return ap

    per_file = defaultdict(lambda: {'cpu_s': 0.0, 'gpu_s': 0.0, 'transfer_s': 0.0, 'hits': 0})
    per_site = defaultdict(lambda: {'hits': 0, 'cpu_s': 0.0, 'gpu_s': 0.0, 'transfer_s': 0.0})

    for func_name, lines in global_profiler.results.items():
        func_obj = global_profiler.functions.get(func_name)
        if func_obj is None:
            continue
        filename = inspect.getsourcefile(func_obj) or func_obj.__code__.co_filename
        rfile = relpath(filename)
        for line_no, (hits, cpu_time, gpu_time, transfer_time) in lines.items():
            pf = per_file[rfile]
            pf['cpu_s'] += cpu_time
            pf['gpu_s'] += gpu_time
            pf['transfer_s'] += transfer_time
            pf['hits'] += hits
            key = (rfile, int(line_no))
            site = per_site[key]
            site['hits'] += hits
            site['cpu_s'] += cpu_time
            site['gpu_s'] += gpu_time
            site['transfer_s'] += transfer_time

    print("\n===== Line-profiler — Time by file =====")
    ranked_files = sorted(per_file.items(), key=lambda kv: kv[1]['cpu_s'], reverse=True)
    for i, (f, s) in enumerate(ranked_files[:20], 1):
        total = s['cpu_s']
        print(f"{i:>2}. {f}\n    CPU: {total:8.4f}s  GPU: {s['gpu_s']:8.4f}s  Transfer: {s['transfer_s']:8.4f}s  Hits: {s['hits']}")

    print("\n===== Line-profiler — Top lines (by CPU time) =====")
    ranked_sites = sorted(per_site.items(), key=lambda kv: kv[1]['cpu_s'], reverse=True)
    for i, ((f, ln), agg) in enumerate(ranked_sites[:30], 1):
        print(f"{i:>2}. {f}:{ln}  CPU: {agg['cpu_s']:.6f}s  GPU: {agg['gpu_s']:.6f}s  Transfer: {agg['transfer_s']:.6f}s  Hits: {agg['hits']}")


def _resolve_attr(path: str):
    """Resolve a dotted path like 'pkg.mod:Class.method' or 'pkg.mod:function' to a Python callable/descriptor."""
    if ':' in path:
        mod_path, attr_path = path.split(':', 1)
    else:
        # also support full dotted attr 'pkg.mod.attr'
        parts = path.split('.')
        for i in range(len(parts), 0, -1):
            mod_path = '.'.join(parts[:i])
            importlib.import_module(mod_path)
            attr_path = '.'.join(parts[i:])
            break
        else:
            mod_path, attr_path = path, ''

    mod = importlib.import_module(mod_path)
    if not attr_path:
        return mod
    obj = mod
    for name in attr_path.split('.'):
        obj = getattr(obj, name)
    return obj


def _iter_module_functions(module) -> List:
    """Yield all functions defined in the given module (non-recursive)."""
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and getattr(obj, '__module__', None) == module.__name__:
            yield obj
        # class methods (unbound)
        if inspect.isclass(obj) and getattr(obj, '__module__', None) == module.__name__:
            for _, member in inspect.getmembers(obj):
                if inspect.isfunction(member) and getattr(member, '__qualname__', '').startswith(obj.__name__ + '.'):
                    yield member


def main():
    parser = argparse.ArgumentParser(description='Profile pytest tests with existing line profiler.')
    parser.add_argument('--line', action='store_true', help='Enable line-by-line profiler with per-line timings')
    parser.add_argument('--report', action='store_true', help='Print extended line-profiler report')
    parser.add_argument('-k', dest='keyword', default=None, help='Only run tests matching the expression')
    parser.add_argument('--maxfail', type=int, default=None, help='Exit after first N failures')
    parser.add_argument('--add-func', action='append', default=[], help='Qualified function or method to include, e.g. pkg.mod:Class.method')
    parser.add_argument('--add-module', action='append', default=[], help='Module to include all functions from, e.g. algorithms.ppo_ac')
    # Capture any extra args intended for pytest (e.g., file paths, -q, -x, etc.)
    args, remaining = parser.parse_known_args()

    # Build pytest arguments
    pytest_args: List[str] = []
    if args.keyword:
        pytest_args += ['-k', args.keyword]
    if args.maxfail is not None:
        pytest_args += ['--maxfail', str(args.maxfail)]
    # Append any remaining args (passed through to pytest)
    pytest_args += remaining
    # If no test selection provided, default to repo tests dir
    if not any(a for a in remaining if not a.startswith('-')):
        pytest_args.append(os.path.join(ROOT, 'tests'))

    if not args.line:
        # Simple passthrough to pytest if no line profiling requested
        import pytest  # type: ignore
        sys.exit(pytest.main(pytest_args))

    # Line profiling mode
    from profilers.line_profiler import profile as line_profile, global_profiler

    # Register user-requested functions/modules to be profiled
    def register_targets():
        # Functions
        for spec in args.add_func:
            func = _resolve_attr(spec)
            if inspect.isfunction(func):
                line_profile(func)  # registers into global_profiler
            elif inspect.ismethod(func):
                line_profile(func.__func__)
            else:
                print(f"[warn] Skipping non-function target: {spec}")
        # Modules
        for mod_spec in args.add_module:
            mod = _resolve_attr(mod_spec)
            if inspect.ismodule(mod):
                for fn in _iter_module_functions(mod):
                    line_profile(fn)
            else:
                print(f"[warn] Not a module: {mod_spec}")

    register_targets()

    # Pytest plugin to wrap test function execution inside our profiler
    class LineProfilerPytestPlugin:
        def pytest_pyfunc_call(self, pyfuncitem):  # noqa: N802 (pytest naming)
            # Ensure the test function itself is tracked
            line_profile(pyfuncitem.obj)

            # Execute the test under the wrapper, passing only declared fixture args
            func = pyfuncitem.obj
            wrapped = line_profile(func)
            # Only pass the arguments the function actually declares
            argnames = getattr(pyfuncitem, "_fixtureinfo", None)
            if argnames is not None:
                argnames = list(getattr(pyfuncitem._fixtureinfo, "argnames", []) or [])
            else:
                # Heuristic: introspect the function signature
                sig = inspect.signature(func)
                argnames = [p.name for p in sig.parameters.values() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
            all_kwargs = getattr(pyfuncitem, "funcargs", {}) or {}
            call_kwargs = {k: all_kwargs[k] for k in argnames if k in all_kwargs}
            wrapped(**call_kwargs)
            return True  # signal that we executed the call

    import pytest  # type: ignore
    exit_code = pytest.main(pytest_args, plugins=[LineProfilerPytestPlugin()])

    # Print the same summaries as the training profiler
    _aggregate_and_print_line_profiler_results(global_profiler, ROOT, extended_report=args.report)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()


