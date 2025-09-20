import functools
from collections import defaultdict
import pandas as pd
import contextlib
import sys
import time
import types
import os

# Try to import torch for GPU profiling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. GPU profiling will be disabled.")

class LineProfiler:
    """
    Enhanced profiler that tracks CPU, GPU, and transfer time at function level.
    """
    def __init__(self):
        self.functions = {}  # Functions to profile
        # Map code objects to unique function keys (module.qualname) to avoid name collisions
        self.code_to_key = {}
        self.allowed_codes = set()  # Code objects of functions we trace
        # Stats structure: {func_name: {line_no: [hits, cpu_time, gpu_time, transfer_time]}}
        self.results = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))
        self.current_stack = []  # Stack of currently executing functions
        self.line_timings = {}   # Track execution time per line
        self.last_line = {}      # Last line executed per frame
        self.start_times = {}    # Timing information for active frames
        self.import_times = {}   # Import statement execution times
        # Aggregate function-level GPU timings (seconds)
        self.func_gpu_time = defaultdict(float)
        # Control GPU timing granularity via env: none|function|line (default: function)
        self.gpu_granularity = os.environ.get('SCOPONE_LINE_GPU', 'function').strip().lower()
        
        # GPU monitoring
        self.use_cuda = TORCH_AVAILABLE and torch.cuda.is_available()
        if self.use_cuda:
            # Initialize GPU stats
            torch.cuda.reset_peak_memory_stats()
            self.last_memory_allocated = torch.cuda.memory_allocated()
            self.last_memory_reserved = torch.cuda.memory_reserved()
    
    def add_function(self, func):
        """Adds a function to profile with line-by-line instrumentation."""
        code = func.__code__
        # Use fully-qualified name for uniqueness across modules/classes
        try:
            qualified_key = f"{getattr(func, '__module__', '<unknown>')}.{getattr(func, '__qualname__', getattr(func, '__name__', '<lambda>'))}"
        except Exception:
            qualified_key = getattr(func, '__name__', '<unknown>')
        self.functions[qualified_key] = func
        self.allowed_codes.add(code)
        self.code_to_key[code] = qualified_key
        self.line_timings[id(code)] = {}
        
        # Create a tracing function for this code object
        def tracefunc(frame, event, arg):
            if event == 'call':
                # Initialize timing when entering a profiled function
                try:
                    codeobj = getattr(frame, 'f_code', None)
                    in_allowed = (isinstance(codeobj, types.CodeType) and (codeobj in self.allowed_codes))
                except Exception:
                    in_allowed = False
                if in_allowed:
                    func_start_time = time.perf_counter()
                    self.start_times[id(frame)] = {
                        'cpu_time': func_start_time,
                        'line_start_time': func_start_time,
                        'last_line': None
                    }
                    frame.f_trace = tracefunc
                return tracefunc
            if event == 'line':
                self._trace_line(frame)
            elif event == 'return':
                self._trace_return(frame)
            return tracefunc
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a copy of the original function to avoid modification
            frame = sys._getframe(0)
            if hasattr(sys, 'setprofile'):
                old_profile = sys.getprofile()
                sys.setprofile(None)
            
            old_trace = sys.gettrace()
            sys.settrace(tracefunc)
            frame.f_trace = tracefunc
            
            try:
                # Track start time for the entire function
                func_start_time = time.perf_counter()
                # Optional function-level GPU timing using CUDA events (low overhead)
                start_event = None
                end_event = None
                if self.use_cuda and self.gpu_granularity in ('function', 'func'):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                
                # Store timing info for the current frame
                self.start_times[id(frame)] = {
                    'cpu_time': func_start_time,
                    'line_start_time': func_start_time,
                    'last_line': None
                }
                
                # Execute the function
                result = func(*args, **kwargs)
                return result
            finally:
                # Restore original trace and profile functions
                sys.settrace(old_trace)
                if hasattr(sys, 'setprofile') and old_profile:
                    sys.setprofile(old_profile)
                
                # Handle timing for the function return
                frame_id = id(frame)
                if frame_id in self.start_times:
                    # Calculate final timing if not already done in trace_return
                    if self.start_times[frame_id]['last_line'] is not None:
                        # Clean up the timing info
                        del self.start_times[frame_id]
                # Finalize optional function-level GPU timing
                if self.use_cuda and self.gpu_granularity in ('function', 'func') and start_event is not None:
                    try:
                        end_event.record()
                        end_event.synchronize()
                        elapsed_s = start_event.elapsed_time(end_event) / 1000.0
                        try:
                            key_for_gpu = self.code_to_key.get(func.__code__)
                        except Exception:
                            key_for_gpu = None
                        if key_for_gpu is None:
                            # Fallback to simple name if mapping missing
                            key_for_gpu = getattr(func, '__name__', '<unknown>')
                        self.func_gpu_time[key_for_gpu] += float(elapsed_s)
                    except Exception:
                        pass
        
        return wrapper
    
    def _trace_line(self, frame):
        """Trace a line execution within a profiled function."""
        frame_id = id(frame)
        code = getattr(frame, 'f_code', None)
        if not isinstance(code, types.CodeType):
            return
        # Resolve unique key for this code object
        func_key = self.code_to_key.get(code)
        line_no = frame.f_lineno
        
        # Verify this is a function we're profiling
        if not func_key or (func_key not in self.functions):
            return
        
        # Get timing info for this frame
        if frame_id not in self.start_times:
            return
        
        timing_info = self.start_times[frame_id]
        last_line = timing_info['last_line']
        
        # If we have a previous line, record its execution time
        if last_line is not None:
            # Current time
            end_time = time.perf_counter()
            # Calculate elapsed time
            cpu_elapsed = end_time - timing_info['line_start_time']
            # No per-line GPU timing by default to avoid distortion
            gpu_elapsed = 0.0
            transfer_time = 0.0
            
            # Update results dictionary
            if last_line in self.results[func_key]:
                self.results[func_key][last_line][0] += 1  # Increment hits
                self.results[func_key][last_line][1] += cpu_elapsed  # Add CPU time
                self.results[func_key][last_line][2] += gpu_elapsed  # Add GPU time
                self.results[func_key][last_line][3] += transfer_time  # Add transfer time
            else:
                self.results[func_key][last_line] = [1, cpu_elapsed, gpu_elapsed, transfer_time]
        
        # Update timing info for the current line
        timing_info['last_line'] = line_no
        timing_info['line_start_time'] = time.perf_counter()
    
    def _trace_return(self, frame):
        """Handle function return while profiling."""
        frame_id = id(frame)
        code = getattr(frame, 'f_code', None)
        if not isinstance(code, types.CodeType):
            return
        func_key = self.code_to_key.get(code)
        
        # Verify this is a function we're profiling
        if not func_key or (func_key not in self.functions):
            return
        
        # Get timing info for this frame
        if frame_id not in self.start_times:
            return
        
        timing_info = self.start_times[frame_id]
        last_line = timing_info['last_line']
        
        # Process the last executed line
        if last_line is not None:
            # Current time
            end_time = time.perf_counter()
            # Calculate elapsed time
            cpu_elapsed = end_time - timing_info['line_start_time']
            gpu_elapsed = 0.0
            transfer_time = 0.0
            
            # Update results
            if last_line in self.results[func_key]:
                self.results[func_key][last_line][0] += 1
                self.results[func_key][last_line][1] += cpu_elapsed
                self.results[func_key][last_line][2] += gpu_elapsed
                self.results[func_key][last_line][3] += transfer_time
            else:
                self.results[func_key][last_line] = [1, cpu_elapsed, gpu_elapsed, transfer_time]
            
            # Clean up timing info (moved to wrapper for safety)
            # del self.start_times[frame_id]
    
    def print_stats(self, top_n=20, sort_by='total'):
        """
        Prints profile statistics with line-by-line details.
        
        Parameters:
        - top_n: Number of top functions to display
        - sort_by: Metric to sort by ('total', 'cpu', 'gpu', 'transfer')
        """
        print("=" * 100)
        print("ENHANCED LINE-BY-LINE PROFILER RESULTS")
        print("=" * 100)
        
        # Calculate total times
        total_cpu_time = 0
        total_gpu_time = 0
        total_transfer_time = 0
        
        # Function level statistics
        function_stats = {}
        
        # Collect statistics by function
        for func_name, lines in self.results.items():
            func_cpu_time = 0
            func_gpu_time = 0
            func_transfer_time = 0
            total_hits = 0
            
            # Calculate per-function totals
            for line_no, (hits, cpu_time, gpu_time, transfer_time) in lines.items():
                total_hits += hits
                func_cpu_time += cpu_time
                # Prefer function-level GPU timing if available
                func_gpu_time += gpu_time
                func_transfer_time += transfer_time
            if self.use_cuda and self.gpu_granularity in ('function', 'func'):
                func_gpu_time = max(func_gpu_time, self.func_gpu_time.get(func_name, 0.0))
            
            # Store function stats
            function_stats[func_name] = {
                'hits': total_hits,
                'cpu_time': func_cpu_time,
                'gpu_time': func_gpu_time,
                'transfer_time': func_transfer_time,
                'total_time': func_cpu_time  # Use CPU time as total
            }
            
            # Add to global totals
            total_cpu_time += func_cpu_time
            total_gpu_time += func_gpu_time
            total_transfer_time += func_transfer_time
        
        # Sort functions by the specified metric
        sort_metrics = {
            'total': 'total_time',
            'cpu': 'cpu_time',
            'gpu': 'gpu_time',
            'transfer': 'transfer_time'
        }
        
        sort_metric = sort_metrics.get(sort_by, 'total_time')
        sorted_functions = sorted(
            function_stats.items(), 
            key=lambda x: x[1][sort_metric], 
            reverse=True
        )
        
        # Display function summary table
        print(f"{'Function':<30} {'Hits':<10} {'CPU Time':<15} {'GPU Time':<15} {'Transfer':<15} {'% of Total':<10}")
        print("-" * 100)
        
        for func_name, stats in sorted_functions[:top_n]:
            cpu_time = stats['cpu_time']
            gpu_time = stats['gpu_time']
            transfer_time = stats['transfer_time']
            total_percent = (cpu_time / total_cpu_time * 100) if total_cpu_time > 0 else 0
            
            print(f"{func_name:<30} {stats['hits']:<10} {cpu_time:>12.6f}s {gpu_time:>12.6f}s {transfer_time:>12.6f}s {total_percent:>9.2f}%")
        
        # Display top 3 functions with line-by-line details
        print("\n" + "=" * 100)
        print("DETAILED LINE-BY-LINE ANALYSIS FOR TOP 3 FUNCTIONS")
        print("=" * 100)
        
        for i, (func_name, stats) in enumerate(sorted_functions[:3]):
            print(f"\n{i+1}. {func_name} ({stats['hits']} calls, {stats['cpu_time']:.6f}s total)")
            print("-" * 100)
            
            # Get source code for context if available
            source_lines = None
            if func_name in self.functions:
                try:
                    import inspect
                    source_lines = inspect.getsourcelines(self.functions[func_name])
                except Exception:
                    from utils.fallback import notify_fallback
                    notify_fallback('profiler.getsourcelines_failed', f'func={func_name}')
            
            # Get line numbers and timings
            func_lines = sorted(self.results[func_name].items())
            
            # Calculate percentages for this function
            func_cpu_time = stats['cpu_time']
            func_gpu_time = stats['gpu_time']
            
            # Print header
            print(f"{'Line':<10} {'Hits':<10} {'CPU Time':<15} {'%':<10} {'GPU Time':<15} {'Transfer':<15} {'Per Hit (CPU)':<15}")
            print("-" * 100)
            
            # Sort lines by CPU time
            sorted_lines = sorted(func_lines, key=lambda x: x[1][1], reverse=True)
            
            # Print each line's statistics
            for line_no, (hits, cpu_time, gpu_time, transfer_time) in sorted_lines:
                cpu_percent = (cpu_time / func_cpu_time * 100) if func_cpu_time > 0 else 0
                per_hit = cpu_time / hits if hits > 0 else 0
                
                # Highlight based on percentage
                if cpu_percent > 20:
                    highlight = "🔥🔥 CRITICAL"
                elif cpu_percent > 10:
                    highlight = "🔥 HOT"
                elif cpu_percent > 5:
                    highlight = "⚠️"
                else:
                    highlight = ""
                
                print(f"{line_no:<10} {hits:<10} {cpu_time:>12.6f}s {cpu_percent:>8.2f}% {gpu_time:>12.6f}s {transfer_time:>12.6f}s {per_hit:>12.6f}s {highlight}")
            
            # Show source code with hotspots if available
            if source_lines:
                print("\nSource code with hotspots:")
                print("-" * 100)
                
                code, start_line = source_lines
                for i, line in enumerate(code):
                    line_no = start_line + i
                    line_str = f"{line_no}: {line.rstrip()}"
                    
                    # Add performance annotation if this is a hotspot
                    if line_no in self.results[func_name]:
                        hits, cpu_time, gpu_time, transfer_time = self.results[func_name][line_no]
                        cpu_percent = (cpu_time / func_cpu_time * 100) if func_cpu_time > 0 else 0
                        
                        if cpu_percent > 5:  # Only annotate significant lines
                            line_str += f"  # {cpu_time:.6f}s ({cpu_percent:.1f}% of time)"
                            
                            if cpu_percent > 20:
                                line_str += " 🔥🔥 CRITICAL"
                            elif cpu_percent > 10:
                                line_str += " 🔥 HOT"
                            elif cpu_percent > 5:
                                line_str += " ⚠️"
                    
                    print(line_str)
        
        # Print summary
        print("\nSummary:")
        print(f"Total CPU Time: {total_cpu_time:.6f}s")
        if self.use_cuda:
            print(f"Total GPU Time: {total_gpu_time:.6f}s")
            print(f"Total Transfer Time: {total_transfer_time:.6f}s")
            print(f"CPU/GPU Ratio: {total_cpu_time/total_gpu_time if total_gpu_time > 0 else 0:.2f}")
        
        print("\n" + "=" * 100)

    def generate_report(self, output_file=None, include_line_details=True, top_n_detailed=3):
        """
        Generates a detailed report for each profiled function with GPU metrics.
        Provides extra detailed line-by-line analysis for top time-consuming functions.
        
        Parameters:
        - output_file: Path to save the report
        - include_line_details: Whether to include line-by-line timing details
        - top_n_detailed: Number of top functions to show detailed line-by-line analysis
        """
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("DETAILED GPU-AWARE PROFILER REPORT")
        report_lines.append("=" * 100)
        
        # Calculate total times
        total_cpu_time = 0
        total_gpu_time = 0
        total_transfer_time = 0
        
        for func_name, lines in self.results.items():
            for line_no, (hits, cpu_time, gpu_time, transfer_time) in lines.items():
                total_cpu_time += cpu_time
                total_transfer_time += transfer_time
                # Prefer function-level GPU totals if available
                if self.use_cuda and self.gpu_granularity in ('function', 'func'):
                    pass
                else:
                    total_gpu_time += gpu_time
        if self.use_cuda and self.gpu_granularity in ('function', 'func'):
            # Sum function-level GPU timings instead of per-line sums
            total_gpu_time = sum(self.func_gpu_time.values())
        
        total_time = total_cpu_time  # Use CPU time as total
        
        # Sort functions by total CPU time
        func_times = []
        for func_name, lines in self.results.items():
            func_cpu_time = sum(cpu_time for _, cpu_time, _, _ in lines.values())
            # Prefer function-level GPU time if available
            if self.use_cuda and self.gpu_granularity in ('function', 'func'):
                func_gpu_time = self.func_gpu_time.get(func_name, 0.0)
            else:
                func_gpu_time = sum(gpu_time for _, _, gpu_time, _ in lines.values())
            func_transfer_time = sum(transfer_time for _, _, _, transfer_time in lines.values())
            func_times.append((func_name, func_cpu_time, func_gpu_time, func_transfer_time))
        
        func_times.sort(key=lambda x: x[1], reverse=True)  # Sort by CPU time
        
        # Mark top N functions for detailed analysis
        top_funcs = set([name for name, _, _, _ in func_times[:top_n_detailed]])
        
        # First pass: Generate overall function summaries
        for func_idx, (func_name, func_cpu_time, func_gpu_time, func_transfer_time) in enumerate(func_times):
            lines = self.results[func_name]
            calls = next(iter(lines.values()))[0] if lines else 0
            
            cpu_percent = func_cpu_time / total_cpu_time * 100 if total_cpu_time > 0 else 0
            gpu_percent = func_gpu_time / total_gpu_time * 100 if total_gpu_time > 0 else 0
            transfer_percent = func_transfer_time / total_transfer_time * 100 if total_transfer_time > 0 else 0
            
            # Special formatting for top functions
            if func_name in top_funcs:
                report_lines.append(f"\n{'#' * 20} TOP {func_idx+1} HOTSPOT {'#' * 20}")
                report_lines.append(f"\n{func_name} ({calls} calls)")
                report_lines.append("=" * 100)
            else:
                report_lines.append(f"\n{func_name} ({calls} calls)")
                report_lines.append("-" * 100)
            
            report_lines.append(f"CPU Time: {func_cpu_time:.6f}s ({cpu_percent:.2f}% of total CPU time)")
            
            if self.use_cuda:
                report_lines.append(f"GPU Time: {func_gpu_time:.6f}s ({gpu_percent:.2f}% of total GPU time)")
                report_lines.append(f"Transfer Time: {func_transfer_time:.6f}s ({transfer_percent:.2f}% of total transfer time)")
                
                # Calculate CPU/GPU ratio
                cpu_gpu_ratio = func_cpu_time / func_gpu_time if func_gpu_time > 0 else float('inf')
                report_lines.append(f"CPU/GPU Ratio: {cpu_gpu_ratio:.2f}")
                
                # Estimate GPU utilization
                gpu_util = func_gpu_time / func_cpu_time * 100 if func_cpu_time > 0 else 0
                report_lines.append(f"GPU Utilization: {gpu_util:.2f}%")
                
                # Estimate transfer overhead
                transfer_overhead = func_transfer_time / (func_cpu_time + func_gpu_time) * 100 if (func_cpu_time + func_gpu_time) > 0 else 0
                report_lines.append(f"Transfer Overhead: {transfer_overhead:.2f}%")
            
                # Include line-by-line details based on conditions
                if include_line_details and (func_name in top_funcs or len(func_times) <= top_n_detailed * 2):
                    # Get line timings and sort by CPU time for this function
                    line_timings = []
                    for line_no, (hits, cpu_time, gpu_time, transfer_time) in lines.items():
                        line_timings.append((line_no, hits, cpu_time, gpu_time, transfer_time))
                    
                    # Sort lines by CPU time (descending)
                    line_timings.sort(key=lambda x: x[2], reverse=True)
                    
                    if func_name in top_funcs:
                        report_lines.append("\n🔍 DETAILED LINE-BY-LINE ANALYSIS:")
                        
                        # Add a table header for line analysis
                        report_lines.append("\n{:<10} {:<10} {:<15} {:<15} {:<15} {:<15} {:<15}".format(
                            "Line", "Hits", "CPU Time", "CPU %", "GPU Time", "GPU %", "Transfer Time"))
                        report_lines.append("-" * 100)
                    else:
                        report_lines.append("\nLine-by-line timing:")
                    
                    # Get a list of the top 5 hottest lines
                    top_lines = [lt[0] for lt in line_timings[:5]]
                    
                    # Process each line
                    for line_no, hits, cpu_time, gpu_time, transfer_time in line_timings:
                        line_cpu_percent = cpu_time / func_cpu_time * 100 if func_cpu_time > 0 else 0
                        per_call_cpu = cpu_time / hits if hits > 0 else 0
                        
                        if func_name in top_funcs:
                            # Detailed formatted table for top functions
                            line_gpu_percent = gpu_time / func_gpu_time * 100 if func_gpu_time > 0 else 0
                            transfer_percent = transfer_time / func_transfer_time * 100 if func_transfer_time > 0 else 0
                            
                            detail = "{:<10} {:<10} {:<15.6f}s {:<15.2f}%".format(
                                line_no, hits, cpu_time, line_cpu_percent)
                            
                            if self.use_cuda:
                                detail += " {:<15.6f}s {:<15.2f}% {:<15.6f}s".format(
                                    gpu_time, line_gpu_percent, transfer_time)
                            
                            # Special marker for critical hotspots
                            if line_no in top_lines:
                                if line_cpu_percent > 20:
                                    detail += " 🔥🔥 CRITICAL HOTSPOT 🔥🔥"
                                elif line_cpu_percent > 10:
                                    detail += " 🔥 HOTSPOT"
                                else:
                                    detail += " ⚠️ SIGNIFICANT"
                            
                            # Add per-call timing for detailed analysis
                            report_lines.append(detail)
                            report_lines.append(f"    └─ Per call: CPU {per_call_cpu:.6f}s" +
                                              (f", GPU {gpu_time/hits:.6f}s" if self.use_cuda else ""))
                            
                            # Add analysis of CPU/GPU efficiency for significant lines
                            if self.use_cuda and line_cpu_percent > 5:
                                cpu_gpu_ratio = cpu_time / gpu_time if gpu_time > 0 else float('inf')
                                if cpu_gpu_ratio > 5:
                                    report_lines.append(f"    └─ INEFFICIENT: CPU-bound (ratio: {cpu_gpu_ratio:.1f})")
                                elif transfer_time > gpu_time * 0.5 and transfer_time > 0.001:
                                    report_lines.append(f"    └─ BOTTLENECK: High transfer overhead ({(transfer_time/cpu_time*100):.1f}%)")
                                
                        else:
                            # Simple format for non-top functions
                            detail = f"Line {line_no}: {hits} hits, CPU: {cpu_time:.6f}s ({line_cpu_percent:.2f}%), {per_call_cpu:.6f}s per hit"
                            
                            if self.use_cuda:
                                line_gpu_percent = gpu_time / func_gpu_time * 100 if func_gpu_time > 0 else 0
                                per_call_gpu = gpu_time / hits if hits > 0 else 0
                                detail += f", GPU: {gpu_time:.6f}s ({line_gpu_percent:.2f}%), {per_call_gpu:.6f}s per hit"
                        
                            report_lines.append(detail)
                
                # Add source code context for top functions if available
                if func_name in top_funcs and func_name in self.functions:
                    try:
                        import inspect
                        import linecache
                        
                        # Get function source code
                        source_lines = inspect.getsourcelines(self.functions[func_name])
                        if source_lines:
                            code_lines, start_line = source_lines
                            
                            # Find hot lines (>5% of function time)
                            hot_lines = {}
                            for line_no, hits, cpu_time, gpu_time, transfer_time in line_timings:
                                if cpu_time / func_cpu_time * 100 > 5:
                                    hot_lines[line_no] = (cpu_time, hits)
                            
                            report_lines.append("\nSource code with hotspot analysis:")
                            report_lines.append("-" * 100)
                            
                            # Show source with annotations
                            for i, line in enumerate(code_lines):
                                line_number = start_line + i
                                line_str = f"{line_number}: {line.rstrip()}"
                                
                                # Add performance annotations for hot lines
                                if line_number in hot_lines:
                                    cpu_time, hits = hot_lines[line_number]
                                    percentage = (cpu_time / func_cpu_time * 100)
                                    
                                    if percentage > 20:
                                        line_str += f"  # 🔥🔥 CRITICAL: {percentage:.1f}% of time ({cpu_time:.6f}s, {hits} hits)"
                                    elif percentage > 10:
                                        line_str += f"  # 🔥 HOT: {percentage:.1f}% of time ({cpu_time:.6f}s)"
                                    else:
                                        line_str += f"  # ⚠️ {percentage:.1f}% of time ({cpu_time:.6f}s)"
                                
                                report_lines.append(line_str)
                            
                            # Add optimization suggestions based on profile data
                            if self.use_cuda:
                                report_lines.append("\nOptimization suggestions:")
                                report_lines.append("-" * 100)
                                
                                # Look for CPU/GPU imbalances
                                cpu_gpu_ratio = func_cpu_time / func_gpu_time if func_gpu_time > 0 else float('inf')
                                if cpu_gpu_ratio > 3:
                                    report_lines.append("- Function is CPU-bound. Consider moving more computation to GPU.")
                                
                                # Look for high transfer overhead
                                transfer_ratio = func_transfer_time / (func_cpu_time + func_gpu_time) if (func_cpu_time + func_gpu_time) > 0 else 0
                                if transfer_ratio > 0.2:  # >20% transfer overhead
                                    report_lines.append("- High data transfer overhead. Consider reducing CPU-GPU transfers.")
                                    
                                # Look for underutilized GPU
                                gpu_util = func_gpu_time / func_cpu_time * 100 if func_cpu_time > 0 else 0
                                if gpu_util < 30 and func_cpu_time > 0.1:
                                    report_lines.append("- Low GPU utilization. Consider batching operations or using CUDA streams.")
                    except Exception as e:
                        report_lines.append(f"\nCould not retrieve source code: {e}")
        
        # Add overall summary
        report_lines.append("\n" + "=" * 100)
        report_lines.append("SUMMARY")
        report_lines.append("=" * 100)
        report_lines.append(f"Total CPU Time: {total_cpu_time:.6f}s")
        
        if self.use_cuda:
            report_lines.append(f"Total GPU Time: {total_gpu_time:.6f}s")
            report_lines.append(f"Total Transfer Time: {total_transfer_time:.6f}s")
            report_lines.append(f"Overall CPU/GPU Ratio: {total_cpu_time/total_gpu_time if total_gpu_time > 0 else float('inf'):.2f}")
            report_lines.append(f"Overall GPU Utilization: {total_gpu_time/total_cpu_time*100 if total_cpu_time > 0 else 0:.2f}%")
            report_lines.append(f"Overall Transfer Overhead: {total_transfer_time/(total_cpu_time+total_gpu_time)*100 if (total_cpu_time+total_gpu_time) > 0 else 0:.2f}%")
        
        report = "\n".join(report_lines)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Report saved to {output_file}")
        
        return report

    def measure_gpu_kernel_time(self, func_name=None):
        """
        Context manager to precisely measure GPU kernel execution time.
        
        Usage:
            with profiler.measure_gpu_kernel_time('my_function'):
                # GPU operations here
        
        Returns time in seconds.
        """
        if not self.use_cuda:
            # No GPU available
            @contextlib.contextmanager
            def dummy_context():
                yield 0
            return dummy_context()
        
        @contextlib.contextmanager
        def gpu_timer():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            
            try:
                yield
            finally:
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event) / 1000  # ms to seconds
                
                if func_name and func_name in self.functions:
                    # Add to the function's stats
                    first_line = self.functions[func_name].__code__.co_firstlineno
                    self.results[func_name][first_line][2] += elapsed_time  # Add to GPU time
                
                return elapsed_time
        
        return gpu_timer()
    
    def _profile_call(self, func, *args, **kwargs):
        """Legacy profiling function - now delegates to the new implementation."""
        # Get the profiled version of the function
        profiled_func = self.add_function(func)
        # Call it directly without creating another wrapper
        return profiled_func(*args, **kwargs)

def profile(func=None, *, profiler=None):
    """
    Decorator for profiling a function with line-by-line CPU and GPU metrics.
    
    Usage:
        @profile
        def my_function():
            pass
    
    or:
        @profile(profiler=my_profiler)
        def my_function():
            pass
    """
    if func is None:
        return lambda f: profile(f, profiler=profiler)
    
    if profiler is None:
        # Use default global profiler
        profiler = global_profiler
    
    return profiler.add_function(func)

# Create a default global profiler
global_profiler = LineProfiler()

# Example usage
if __name__ == "__main__":
    @profile
    def slow_function(n):
        result = 0
        for i in range(n):
            result += i
            time.sleep(0.001)  # Simulate a slow operation
        return result
    
    slow_function(100)
    global_profiler.print_stats()
    print(global_profiler.generate_report())