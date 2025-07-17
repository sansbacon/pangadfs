# pangadfs/pangadfs/profiler.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import time
import logging
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class OperationStats:
    """Statistics for a single operation type."""
    total_time: float = 0.0
    call_count: int = 0
    min_time: float = float('inf')
    max_time: float = 0.0
    times: List[float] = field(default_factory=list)
    
    @property
    def avg_time(self) -> float:
        """Calculate average time per call."""
        return self.total_time / self.call_count if self.call_count > 0 else 0.0
    
    def add_timing(self, duration: float) -> None:
        """Add a new timing measurement."""
        self.total_time += duration
        self.call_count += 1
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.times.append(duration)


class GAProfiler:
    """Genetic Algorithm profiler for timing and performance analysis."""
    
    def __init__(self, enabled: bool = True):
        """Initialize the profiler.
        
        Args:
            enabled: Whether profiling is active
        """
        self.enabled = enabled
        self.stats: Dict[str, OperationStats] = defaultdict(OperationStats)
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.best_solution_time: Optional[float] = None
        self.best_solution_generation: Optional[int] = None
        self.generation_times: List[float] = []
        self.current_generation: int = 0
        
        # Track setup phase separately
        self.setup_complete_time: Optional[float] = None
        
    def start_optimization(self) -> None:
        """Mark the start of optimization."""
        if self.enabled:
            self.start_time = time.time()
            logging.debug("GA Profiler: Optimization started")
    
    def end_optimization(self) -> None:
        """Mark the end of optimization."""
        if self.enabled:
            self.end_time = time.time()
            logging.debug("GA Profiler: Optimization completed")
    
    def mark_setup_complete(self) -> None:
        """Mark the completion of setup phase."""
        if self.enabled:
            self.setup_complete_time = time.time()
            logging.debug("GA Profiler: Setup phase completed")
    
    def mark_best_solution(self, generation: int) -> None:
        """Mark when the best solution was found.
        
        Args:
            generation: The generation number when best solution was found
        """
        if self.enabled:
            self.best_solution_time = time.time()
            self.best_solution_generation = generation
            logging.debug(f"GA Profiler: Best solution found at generation {generation}")
    
    def start_generation(self, generation: int) -> None:
        """Mark the start of a new generation.
        
        Args:
            generation: The generation number
        """
        if self.enabled:
            self.current_generation = generation
            self._generation_start_time = time.time()
    
    def end_generation(self) -> None:
        """Mark the end of current generation."""
        if self.enabled and hasattr(self, '_generation_start_time'):
            generation_time = time.time() - self._generation_start_time
            self.generation_times.append(generation_time)
            logging.debug(f"GA Profiler: Generation {self.current_generation} completed in {generation_time:.3f}s")
    
    @contextmanager
    def time_operation(self, operation_name: str):
        """Context manager for timing operations.
        
        Args:
            operation_name: Name of the operation being timed
        """
        if not self.enabled:
            yield
            return
            
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.stats[operation_name].add_timing(duration)
            logging.debug(f"GA Profiler: {operation_name} took {duration:.3f}s")
    
    def get_total_time(self) -> float:
        """Get total optimization time."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    def get_setup_time(self) -> float:
        """Get setup phase time."""
        if self.start_time and self.setup_complete_time:
            return self.setup_complete_time - self.start_time
        return 0.0
    
    def get_time_to_best_solution(self) -> float:
        """Get time from start to best solution."""
        if self.start_time and self.best_solution_time:
            return self.best_solution_time - self.start_time
        return 0.0
    
    def get_optimization_time(self) -> float:
        """Get time spent in optimization loop (excluding setup)."""
        if self.setup_complete_time and self.end_time:
            return self.end_time - self.setup_complete_time
        return 0.0
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get comprehensive profiling statistics."""
        total_time = self.get_total_time()
        setup_time = self.get_setup_time()
        optimization_time = self.get_optimization_time()
        time_to_best = self.get_time_to_best_solution()
        
        return {
            'total_time': total_time,
            'setup_time': setup_time,
            'optimization_time': optimization_time,
            'time_to_best_solution': time_to_best,
            'best_solution_generation': self.best_solution_generation,
            'generations_completed': len(self.generation_times),
            'avg_generation_time': sum(self.generation_times) / len(self.generation_times) if self.generation_times else 0.0,
            'operation_stats': dict(self.stats),
            'generation_times': self.generation_times.copy()
        }
    
    def print_profiling_results(self) -> None:
        """Print comprehensive profiling results in a formatted table."""
        if not self.enabled:
            print("Profiling was not enabled.")
            return
        
        stats = self.get_summary_stats()
        
        print("\n" + "=" * 60)
        print("GENETIC ALGORITHM PROFILING RESULTS")
        print("=" * 60)
        
        # Overall timing summary
        print(f"\nTotal Optimization Time: {stats['total_time']:.2f} seconds")
        print(f"Setup Time: {stats['setup_time']:.2f} seconds")
        print(f"Optimization Loop Time: {stats['optimization_time']:.2f} seconds")
        
        if stats['time_to_best_solution'] > 0:
            print(f"Time to Best Solution: {stats['time_to_best_solution']:.2f} seconds (Generation {stats['best_solution_generation']})")
            percentage = (stats['time_to_best_solution'] / stats['total_time']) * 100
            print(f"Best Solution Found at: {percentage:.1f}% of total runtime")
        
        print(f"Generations Completed: {stats['generations_completed']}")
        if stats['avg_generation_time'] > 0:
            print(f"Average Generation Time: {stats['avg_generation_time']:.3f} seconds")
        
        # Component timing table
        print(f"\nComponent Timing Summary:")
        print("┌─────────────────────┬───────────┬─────────┬─────────┬─────────┬───────────┐")
        print("│ Operation           │ Total (s) │ Avg (s) │ Min (s) │ Max (s) │ Calls     │")
        print("├─────────────────────┼───────────┼─────────┼─────────┼─────────┼───────────┤")
        
        # Sort operations by total time (descending)
        sorted_ops = sorted(stats['operation_stats'].items(), 
                          key=lambda x: x[1].total_time, reverse=True)
        
        for op_name, op_stats in sorted_ops:
            if op_stats.call_count > 0:
                min_time = op_stats.min_time if op_stats.min_time != float('inf') else 0.0
                print(f"│ {op_name:<19} │ {op_stats.total_time:>9.2f} │ {op_stats.avg_time:>7.3f} │ "
                      f"{min_time:>7.3f} │ {op_stats.max_time:>7.3f} │ {op_stats.call_count:>9} │")
        
        print("└─────────────────────┴───────────┴─────────┴─────────┴─────────┴───────────┘")
        
        # Performance insights
        print(f"\nPerformance Insights:")
        if stats['operation_stats']:
            total_component_time = sum(op.total_time for op in stats['operation_stats'].values())
            
            # Find top time consumers
            top_operations = sorted(stats['operation_stats'].items(), 
                                  key=lambda x: x[1].total_time, reverse=True)[:3]
            
            for i, (op_name, op_stats) in enumerate(top_operations, 1):
                if op_stats.total_time > 0:
                    percentage = (op_stats.total_time / total_component_time) * 100
                    print(f"• {op_name} consumed {percentage:.1f}% of component time")
        
        # Generation timing insights
        if len(stats['generation_times']) > 1:
            fastest_gen = min(stats['generation_times'])
            slowest_gen = max(stats['generation_times'])
            print(f"• Fastest generation: {fastest_gen:.3f}s, Slowest: {slowest_gen:.3f}s")
        
        print("=" * 60)
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export all profiling data to a dictionary for external use."""
        if not self.enabled:
            return {'profiling_enabled': False}
        
        stats = self.get_summary_stats()
        
        # Convert OperationStats objects to dictionaries
        operation_data = {}
        for op_name, op_stats in stats['operation_stats'].items():
            operation_data[op_name] = {
                'total_time': op_stats.total_time,
                'call_count': op_stats.call_count,
                'avg_time': op_stats.avg_time,
                'min_time': op_stats.min_time if op_stats.min_time != float('inf') else 0.0,
                'max_time': op_stats.max_time,
                'times': op_stats.times.copy()
            }
        
        return {
            'profiling_enabled': True,
            'total_time': stats['total_time'],
            'setup_time': stats['setup_time'],
            'optimization_time': stats['optimization_time'],
            'time_to_best_solution': stats['time_to_best_solution'],
            'best_solution_generation': stats['best_solution_generation'],
            'generations_completed': stats['generations_completed'],
            'avg_generation_time': stats['avg_generation_time'],
            'generation_times': stats['generation_times'],
            'operations': operation_data
        }
