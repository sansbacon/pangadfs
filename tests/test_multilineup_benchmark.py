# tests/test_multilineup_benchmark.py
# -*- coding: utf-8 -*-
# Comprehensive benchmark test for multilineup optimization approaches

import logging
import time
import tracemalloc
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
import sys
import os

# Add pangadfs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pangadfs.ga import GeneticAlgorithm
from pangadfs.optimize import OptimizeMultilineup, OptimizeMultilineupSets
from pangadfs.optimize_pool_based import OptimizePoolBasedSets
from pangadfs.optimize_multi_objective import OptimizeMultiObjective

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MultilineupBenchmark:
    """Comprehensive benchmark for multilineup optimization approaches"""
    
    def __init__(self, pool_path: str):
        self.pool_path = pool_path
        self.results = {}
        
        # Test configurations focused on large lineup sets
        self.test_configs = [
            # Primary focus: Large lineup sets
            {'lineups': 50, 'pop_size': 500, 'generations': 100, 'weight': 'high'},
            {'lineups': 100, 'pop_size': 1000, 'generations': 100, 'weight': 'high'},
            {'lineups': 150, 'pop_size': 1000, 'generations': 150, 'weight': 'high'},
            
            # Secondary: Baseline comparison
            {'lineups': 20, 'pop_size': 200, 'generations': 50, 'weight': 'low'},
        ]
        
        # Approaches to test
        self.approaches = {
            'OptimizeMultilineup': OptimizeMultilineup,
            'OptimizeMultilineupSets': OptimizeMultilineupSets,
            'OptimizePoolBasedSets': OptimizePoolBasedSets,
            'OptimizeMultiObjective': OptimizeMultiObjective
        }
        
        # Calculate theoretical optimal for 5% benchmark
        self.theoretical_optimal = None
        
    def calculate_theoretical_optimal(self) -> float:
        """Calculate theoretical optimal lineup score for 5% benchmark"""
        try:
            # Load pool data
            pool_df = pd.read_csv(self.pool_path)
            
            # Basic position requirements (adjust as needed)
            pos_requirements = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1}
            flex_count = 1  # FLEX position
            salary_cap = 50000
            
            # Sort by projection descending
            pool_df = pool_df.sort_values('proj', ascending=False)
            
            # Greedy selection for theoretical max
            selected_players = []
            total_salary = 0
            total_proj = 0
            pos_filled = {pos: 0 for pos in pos_requirements}
            
            # Fill required positions first
            for _, player in pool_df.iterrows():
                pos = player['pos']
                salary = player['salary']
                proj = player['proj']
                
                # Check if we can afford this player
                if total_salary + salary > salary_cap:
                    continue
                
                # Check if we need this position
                if pos in pos_requirements and pos_filled[pos] < pos_requirements[pos]:
                    selected_players.append(player)
                    total_salary += salary
                    total_proj += proj
                    pos_filled[pos] += 1
                    
                    # Check if all positions filled
                    if all(pos_filled[pos] >= pos_requirements[pos] for pos in pos_requirements):
                        break
            
            # Fill FLEX position with best remaining player
            if len(selected_players) < sum(pos_requirements.values()) + flex_count:
                for _, player in pool_df.iterrows():
                    if player['playerid'] not in [p['playerid'] for p in selected_players]:
                        pos = player['pos']
                        salary = player['salary']
                        proj = player['proj']
                        
                        # FLEX can be RB, WR, or TE
                        if pos in ['RB', 'WR', 'TE'] and total_salary + salary <= salary_cap:
                            selected_players.append(player)
                            total_salary += salary
                            total_proj += proj
                            break
            
            self.theoretical_optimal = total_proj
            logging.info(f'Theoretical optimal lineup: {total_proj:.1f} points (${total_salary:,})')
            return total_proj
            
        except Exception as e:
            logging.error(f'Error calculating theoretical optimal: {e}')
            # Fallback estimate
            self.theoretical_optimal = 200.0
            return 200.0
    
    def create_ga_context(self, config: Dict) -> Dict:
        """Create GA context for testing"""
        return {
            'ga_settings': {
                'csvpth': self.pool_path,
                'population_size': config['pop_size'],
                'n_generations': config['generations'],
                'stop_criteria': 20,
                'target_lineups': config['lineups'],
                'diversity_weight': 0.3,
                'diversity_method': 'jaccard',
                'min_overlap_threshold': 0.4,
                'mutation_rate': 0.1,
                'elite_divisor': 5,
                'points_column': 'proj',
                'position_column': 'pos',
                'salary_column': 'salary',
                'verbose': False,
                
                # Pool-based specific settings
                'initial_pool_size': 100000,
                'elite_pool_size': 10000,
                'initial_elite_ratio': 0.7,
                
                # Multi-objective specific settings
                'top_k_lineups': min(10, config['lineups']),
                'multi_objective_weights': (0.6, 0.3, 0.1),
            },
            'site_settings': {
                'salary_cap': 50000,
                'posfilter': {'QB': 12, 'RB': 8, 'WR': 6, 'TE': 5, 'DST': 4, 'FLEX': 6},
                'posmap': {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1, 'FLEX': 1},
                'flex_positions': ['RB', 'WR', 'TE']
            }
        }
    
    def run_single_test(self, approach_name: str, approach_class: Any, 
                       config: Dict, run_number: int) -> Dict[str, Any]:
        """Run a single test for one approach"""
        logging.info(f'Running {approach_name} - {config["lineups"]} lineups - Run {run_number + 1}')
        
        # Start memory tracking
        tracemalloc.start()
        start_time = time.time()
        
        try:
            # Create GA instance
            ga_ctx = self.create_ga_context(config)
            ga = GeneticAlgorithm(ctx=ga_ctx)
            
            # Create optimizer instance
            optimizer = approach_class()
            
            # Run optimization
            results = optimizer.optimize(ga)
            
            # Stop timing and memory tracking
            end_time = time.time()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Extract metrics
            lineups = results.get('lineups', [])
            scores = results.get('scores', [])
            diversity_metrics = results.get('diversity_metrics', {})
            
            if not scores:
                logging.warning(f'{approach_name} returned no scores')
                return None
            
            # Calculate key performance metrics
            best_score = max(scores)
            top_5_avg = np.mean(sorted(scores, reverse=True)[:min(5, len(scores))])
            total_score = sum(scores)
            avg_score = np.mean(scores)
            
            # Near-optimal coverage (within 5% of theoretical optimal)
            near_optimal_threshold = self.theoretical_optimal * 0.95
            near_optimal_count = sum(1 for score in scores if score >= near_optimal_threshold)
            near_optimal_pct = (near_optimal_count / len(scores)) * 100
            
            # Diversity metrics
            avg_overlap = diversity_metrics.get('avg_overlap', 0.0)
            min_overlap = diversity_metrics.get('min_overlap', 0.0)
            
            return {
                'approach': approach_name,
                'config': config,
                'run': run_number,
                'runtime_seconds': end_time - start_time,
                'peak_memory_mb': peak / 1024 / 1024,
                'best_score': best_score,
                'top_5_avg_score': top_5_avg,
                'total_score': total_score,
                'avg_score': avg_score,
                'near_optimal_count': near_optimal_count,
                'near_optimal_pct': near_optimal_pct,
                'avg_overlap': avg_overlap,
                'min_overlap': min_overlap,
                'lineup_count': len(scores),
                'success': True
            }
            
        except Exception as e:
            logging.error(f'Error in {approach_name}: {str(e)}')
            logging.error(traceback.format_exc())
            tracemalloc.stop()
            return {
                'approach': approach_name,
                'config': config,
                'run': run_number,
                'error': str(e),
                'success': False
            }
    
    def run_comprehensive_benchmark(self, runs_per_config: int = 3) -> Dict[str, Any]:
        """Run comprehensive benchmark across all approaches and configurations"""
        logging.info('Starting comprehensive multilineup benchmark')
        
        # Calculate theoretical optimal
        self.calculate_theoretical_optimal()
        
        all_results = []
        
        # Test each approach
        for approach_name, approach_class in self.approaches.items():
            logging.info(f'\n=== Testing {approach_name} ===')
            
            # Test each configuration
            for config in self.test_configs:
                logging.info(f'Configuration: {config["lineups"]} lineups, {config["pop_size"]} pop, {config["generations"]} gen')
                
                # Multiple runs for statistical significance
                config_results = []
                for run in range(runs_per_config):
                    result = self.run_single_test(approach_name, approach_class, config, run)
                    if result and result['success']:
                        config_results.append(result)
                        all_results.append(result)
                        
                        # Log key metrics
                        logging.info(f'  Run {run + 1}: Best={result["best_score"]:.1f}, '
                                   f'Near-optimal={result["near_optimal_pct"]:.1f}%, '
                                   f'Diversity={result["avg_overlap"]:.3f}, '
                                   f'Time={result["runtime_seconds"]:.1f}s')
                
                # Log configuration summary
                if config_results:
                    avg_best = np.mean([r['best_score'] for r in config_results])
                    avg_near_optimal = np.mean([r['near_optimal_pct'] for r in config_results])
                    logging.info(f'  Config Summary: Avg Best={avg_best:.1f}, Avg Near-optimal={avg_near_optimal:.1f}%')
        
        # Store results
        self.results = {
            'all_results': all_results,
            'theoretical_optimal': self.theoretical_optimal,
            'test_configs': self.test_configs,
            'approaches_tested': list(self.approaches.keys())
        }
        
        return self.results
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze benchmark results and provide recommendations"""
        if not self.results or not self.results['all_results']:
            return {'error': 'No results to analyze'}
        
        df = pd.DataFrame(self.results['all_results'])
        successful_df = df[df['success'] is True]
        
        if len(successful_df) == 0:
            return {'error': 'No successful runs to analyze'}
        
        analysis = {
            'summary_stats': {},
            'rankings': {},
            'recommendations': {}
        }
        
        # Calculate summary statistics by approach
        for approach in self.approaches:
            approach_data = successful_df[successful_df['approach'] == approach]
            
            if len(approach_data) > 0:
                analysis['summary_stats'][approach] = {
                    'avg_best_score': approach_data['best_score'].mean(),
                    'std_best_score': approach_data['best_score'].std(),
                    'avg_top5_score': approach_data['top_5_avg_score'].mean(),
                    'avg_near_optimal_pct': approach_data['near_optimal_pct'].mean(),
                    'avg_diversity': approach_data['avg_overlap'].mean(),
                    'avg_runtime': approach_data['runtime_seconds'].mean(),
                    'avg_memory_mb': approach_data['peak_memory_mb'].mean(),
                    'success_rate': len(approach_data) / len(df[df['approach'] == approach]) * 100
                }
        
        # Rankings by key criteria (user's priorities)
        criteria_weights = {
            'best_score': 0.4,      # #1 priority: Best individual lineup quality
            'near_optimal_pct': 0.3, # #2 priority: Near-optimal coverage
            'top5_score': 0.2,      # Consistency at the top
            'diversity': 0.1        # Diversity (lower overlap is better)
        }
        
        approach_scores = {}
        for approach in self.approaches:
            if approach in analysis['summary_stats']:
                stats = analysis['summary_stats'][approach]
                
                # Normalize scores (0-1 scale)
                best_score_norm = stats['avg_best_score'] / max([s['avg_best_score'] for s in analysis['summary_stats'].values()])
                near_optimal_norm = stats['avg_near_optimal_pct'] / 100.0
                top5_norm = stats['avg_top5_score'] / max([s['avg_top5_score'] for s in analysis['summary_stats'].values()])
                diversity_norm = 1.0 - stats['avg_diversity']  # Lower overlap is better
                
                # Calculate weighted score
                weighted_score = (
                    criteria_weights['best_score'] * best_score_norm +
                    criteria_weights['near_optimal_pct'] * near_optimal_norm +
                    criteria_weights['top5_score'] * top5_norm +
                    criteria_weights['diversity'] * diversity_norm
                )
                
                approach_scores[approach] = weighted_score
        
        # Rank approaches
        ranked_approaches = sorted(approach_scores.items(), key=lambda x: x[1], reverse=True)
        analysis['rankings']['overall'] = ranked_approaches
        
        # Individual metric rankings
        for metric in ['avg_best_score', 'avg_near_optimal_pct', 'avg_top5_score']:
            metric_ranking = sorted(
                [(approach, stats[metric]) for approach, stats in analysis['summary_stats'].items()],
                key=lambda x: x[1], reverse=True
            )
            analysis['rankings'][metric] = metric_ranking
        
        # Generate recommendations
        winner = ranked_approaches[0][0] if ranked_approaches else None
        
        analysis['recommendations'] = {
            'primary_recommendation': winner,
            'reasoning': self._generate_recommendation_reasoning(analysis, winner),
            'approaches_to_remove': [approach for approach, score in ranked_approaches[2:]] if len(ranked_approaches) > 2 else [],
            'performance_summary': self._generate_performance_summary(analysis)
        }
        
        return analysis
    
    @staticmethod
    def _generate_recommendation_reasoning(analysis: Dict, winner: str) -> str:
        """Generate reasoning for the recommendation"""
        if not winner or winner not in analysis['summary_stats']:
            return "Insufficient data for recommendation"
        
        stats = analysis['summary_stats'][winner]
        reasoning = f"{winner} is recommended because:\n"
        reasoning += f"- Best individual lineup quality: {stats['avg_best_score']:.1f} points average\n"
        reasoning += f"- Near-optimal coverage: {stats['avg_near_optimal_pct']:.1f}% of lineups within 5% of optimal\n"
        reasoning += f"- Top-5 consistency: {stats['avg_top5_score']:.1f} points average\n"
        reasoning += f"- Reasonable diversity: {stats['avg_diversity']:.3f} average overlap\n"
        reasoning += f"- Performance: {stats['avg_runtime']:.1f}s runtime, {stats['avg_memory_mb']:.0f}MB memory\n"
        
        return reasoning
    
    @staticmethod
    def _generate_performance_summary(analysis: Dict) -> str:
        """Generate overall performance summary"""
        summary = "Performance Summary:\n"
        
        for approach, stats in analysis['summary_stats'].items():
            summary += f"\n{approach}:\n"
            summary += f"  Best Score: {stats['avg_best_score']:.1f} ± {stats['avg_best_score']:.1f}\n"
            summary += f"  Near-Optimal: {stats['avg_near_optimal_pct']:.1f}%\n"
            summary += f"  Diversity: {stats['avg_diversity']:.3f}\n"
            summary += f"  Runtime: {stats['avg_runtime']:.1f}s\n"
        
        return summary
    
    def save_results(self, output_path: str):
        """Save detailed results to files"""
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        # Save raw results
        if self.results and 'all_results' in self.results:
            df = pd.DataFrame(self.results['all_results'])
            df.to_csv(output_dir / 'benchmark_raw_results.csv', index=False)
        
        # Save analysis
        analysis = self.analyze_results()
        
        # Save summary report
        with open(output_dir / 'benchmark_report.txt', 'w') as f:
            f.write("MULTILINEUP OPTIMIZATION BENCHMARK REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Theoretical Optimal: {self.theoretical_optimal:.1f} points\n\n")
            
            if 'recommendations' in analysis:
                f.write("RECOMMENDATION:\n")
                f.write(f"Primary: {analysis['recommendations']['primary_recommendation']}\n\n")
                f.write("Reasoning:\n")
                f.write(analysis['recommendations']['reasoning'])
                f.write("\n\n")
                
                f.write("Performance Summary:\n")
                f.write(analysis['recommendations']['performance_summary'])
        
        logging.info(f'Results saved to {output_dir}')


def run_benchmark():
    """Main function to run the benchmark"""
    # Use pool3.csv from appdata
    pool_path = Path(__file__).parent.parent / 'pangadfs' / 'app' / 'appdata' / 'pool3.csv'
    
    if not pool_path.exists():
        logging.error(f'Pool file not found: {pool_path}')
        return
    
    # Create benchmark instance
    benchmark = MultilineupBenchmark(str(pool_path))
    
    # Run comprehensive benchmark
    logging.info('Starting comprehensive multilineup optimization benchmark')
    results = benchmark.run_comprehensive_benchmark(runs_per_config=3)
    
    # Analyze results
    analysis = benchmark.analyze_results()
    
    # Save results
    output_dir = Path(__file__).parent / 'benchmark_results'
    benchmark.save_results(str(output_dir))
    
    # Print summary
    if 'recommendations' in analysis:
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 60)
        print(f"WINNER: {analysis['recommendations']['primary_recommendation']}")
        print("\nReasoning:")
        print(analysis['recommendations']['reasoning'])
        print("\nApproaches to remove:")
        for approach in analysis['recommendations']['approaches_to_remove']:
            print(f"- {approach}")
    
    return results, analysis


if __name__ == '__main__':
    run_benchmark()
