import time
import functools
import numpy as np
from collections import defaultdict
import pandas as pd

class LineProfiler:
    """
    Profiler personalizzato che traccia il tempo di esecuzione a livello di riga.
    """
    def __init__(self):
        self.functions = {}  # Funzioni da profilare
        self.results = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # {func_name: {line_no: [hits, time]}}
        self.current_stack = []  # Stack delle funzioni attualmente in esecuzione
    
    def add_function(self, func):
        """Aggiunge una funzione da profilare."""
        self.functions[func.__name__] = func
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self._profile_call(func, *args, **kwargs)
        
        return wrapper
    
    def _profile_call(self, func, *args, **kwargs):
        """Profila una chiamata di funzione."""
        func_name = func.__name__
        frame = None
        
        # Salva il frame corrente
        try:
            frame = sys._getframe(1)
        except:
            pass
        
        self.current_stack.append((func_name, frame))
        start_time = time.time()
        
        # Esegui la funzione originale
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.time() - start_time
            # Aggiungi il tempo totale alla prima riga della funzione (entry point)
            first_line = func.__code__.co_firstlineno
            self.results[func_name][first_line][0] += 1
            self.results[func_name][first_line][1] += elapsed
            self.current_stack.pop()
    
    def print_stats(self, top_n=20):
        """Stampa le statistiche delle funzioni profilate."""
        print("=" * 80)
        print("LINE PROFILER RESULTS")
        print("=" * 80)
        
        # Calcola il tempo totale
        total_time = 0
        for func_name, lines in self.results.items():
            for line_no, (hits, time_spent) in lines.items():
                total_time += time_spent
        
        # Prepara i dati per stamparli in ordine
        all_stats = []
        for func_name, lines in self.results.items():
            func_time = sum(time_spent for _, time_spent in lines.values())
            func_percent = func_time / total_time * 100 if total_time > 0 else 0
            all_stats.append({
                'Name': func_name,
                'Calls': next(iter(lines.values()))[0] if lines else 0,
                'Total Time': func_time,
                'Percent': func_percent
            })
        
        # Ordina per tempo totale e stampa le prime N funzioni
        all_stats.sort(key=lambda x: x['Total Time'], reverse=True)
        
        # Crea un DataFrame per una visualizzazione più pulita
        df = pd.DataFrame(all_stats[:top_n])
        df['Total Time'] = df['Total Time'].apply(lambda x: f"{x:.6f}s")
        df['Percent'] = df['Percent'].apply(lambda x: f"{x:.2f}%")
        
        print(df.to_string(index=False))
        print("\n" + "=" * 80)

    def generate_report(self, output_file=None):
        """
        Genera un report dettagliato per ogni funzione profilata.
        Può salvare il report su file se specificato.
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("DETAILED LINE PROFILER REPORT")
        report_lines.append("=" * 80)
        
        # Calcola il tempo totale
        total_time = 0
        for func_name, lines in self.results.items():
            for line_no, (hits, time_spent) in lines.items():
                total_time += time_spent
        
        # Ordina le funzioni per tempo totale
        func_times = []
        for func_name, lines in self.results.items():
            func_time = sum(time_spent for _, time_spent in lines.values())
            func_times.append((func_name, func_time))
        
        func_times.sort(key=lambda x: x[1], reverse=True)
        
        # Genera il report per ogni funzione
        for func_name, func_time in func_times:
            lines = self.results[func_name]
            calls = next(iter(lines.values()))[0] if lines else 0
            percent = func_time / total_time * 100 if total_time > 0 else 0
            
            report_lines.append(f"\n{func_name} ({calls} calls, {func_time:.6f}s, {percent:.2f}% of total time)")
            report_lines.append("-" * 80)
            
            sorted_lines = sorted(lines.items())
            for line_no, (hits, time_spent) in sorted_lines:
                line_percent = time_spent / func_time * 100 if func_time > 0 else 0
                per_call = time_spent / hits if hits > 0 else 0
                report_lines.append(f"Line {line_no}: {hits} hits, {time_spent:.6f}s ({line_percent:.2f}%), {per_call:.6f}s per hit")
            
        
        report = "\n".join(report_lines)
        
        # Salva su file se richiesto
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Report salvato in {output_file}")
        
        return report

def profile(func=None, *, profiler=None):
    """
    Decoratore per profilare una funzione.
    
    Utilizzo:
        @profile
        def my_function():
            pass
    
    oppure:
        @profile(profiler=my_profiler)
        def my_function():
            pass
    """
    if func is None:
        return lambda f: profile(f, profiler=profiler)
    
    if profiler is None:
        # Usa il profiler globale di default
        profiler = global_profiler
    
    return profiler.add_function(func)

# Crea un profiler globale di default
global_profiler = LineProfiler()

# Esempio di utilizzo
if __name__ == "__main__":
    @profile
    def slow_function(n):
        result = 0
        for i in range(n):
            result += i
            time.sleep(0.001)  # Simula un'operazione lenta
        return result
    
    slow_function(100)
    global_profiler.print_stats()
    print(global_profiler.generate_report())