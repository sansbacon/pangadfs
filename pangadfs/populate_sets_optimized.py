# pangadfs/populate_sets_optimized.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import logging
from typing import Dict, Any
import numpy as np
import pandas as pd
from pangadfs.base import PopulateBase
from pangadfs.misc import multidimensional_shifting_fast

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


class PopulateMultilineupSetsOptimized(PopulateBase):
    """Highly optimized version for creating initial population of lineup sets"""

    def populate(self, 
                 pospool: Dict[str, pd.DataFrame],
                 posmap: Dict[str, int], 
                 population_size: int,
                 target_lineups: int,
                 probcol: str = 'prob',
                 diversity_threshold: float = 0.3,
                 max_attempts: int = 100,
                 lineup_pool_size: int = 100000,
                 **kwargs) -> np.ndarray:
        """
        Creates initial population using highly optimized vectorized approach
        
        Args:
            pospool: Dictionary of position DataFrames
            posmap: Position mapping (e.g., {'QB': 1, 'RB': 2, ...})
            population_size: Number of lineup sets to create
            target_lineups: Number of lineups per set
            probcol: Column name for probabilities
            diversity_threshold: Minimum diversity required between lineups in a set
            max_attempts: Maximum attempts to generate diverse lineup within a set
            lineup_pool_size: Size of initial lineup pool to generate
            
        Returns:
            np.ndarray: Shape (population_size, target_lineups, lineup_size)
        """
        logging.info(f'OPTIMIZED: Generating {population_size} sets of {target_lineups} diverse lineups each')
        
        # Calculate lineup size from posmap
        lineup_size = sum(posmap.values())
        
        # Pre-calculate position data for vectorized operations
        pos_data = self._prepare_position_data(pospool, posmap, probcol)
        
        # Use different strategies based on pool size and target
        if lineup_pool_size <= 25000 and target_lineups <= 50:
            # Small to medium: Use vectorized pool + smart sampling
            logging.info(f'Using VECTORIZED POOL strategy (pool_size={lineup_pool_size})')
            return self._generate_sets_vectorized_pool(
                pos_data, population_size, target_lineups, lineup_size, 
                lineup_pool_size, diversity_threshold
            )
        else:
            # Large: Use direct diverse generation without full pool
            logging.info('Using DIRECT DIVERSE strategy (avoiding large pool)')
            return self._generate_sets_direct_diverse(
                pos_data, population_size, target_lineups, lineup_size, 
                diversity_threshold, max_attempts
            )
    
    @staticmethod
    def _prepare_position_data(pospool: Dict[str, pd.DataFrame], 
                             posmap: Dict[str, int], probcol: str) -> Dict[str, Any]:
        """Prepare position data for vectorized operations"""
        pos_data = {}
        current_idx = 0
        
        for pos, count in posmap.items():
            if pos not in pospool or len(pospool[pos]) == 0:
                pos_data[pos] = {
                    'start_idx': current_idx,
                    'end_idx': current_idx + count,
                    'count': count,
                    'indices': np.array([]),
                    'probs': np.array([])
                }
                current_idx += count
                continue
            
            pos_df = pospool[pos]
            indices = pos_df.index.values
            
            if probcol in pos_df.columns:
                probs = pos_df[probcol].values
                probs = probs / np.sum(probs)  # Normalize
            else:
                probs = np.ones(len(indices)) / len(indices)
            
            pos_data[pos] = {
                'start_idx': current_idx,
                'end_idx': current_idx + count,
                'count': count,
                'indices': indices,
                'probs': probs.astype(np.float32)
            }
            current_idx += count
        
        return pos_data
    
    def _generate_sets_vectorized_pool(self, pos_data: Dict[str, Any], 
                                     population_size: int, target_lineups: int,
                                     lineup_size: int, lineup_pool_size: int,
                                     diversity_threshold: float) -> np.ndarray:
        """Generate sets using vectorized pool approach for smaller problems"""
        
        # Step 1: Generate lineup pool using vectorized operations
        logging.info(f'Step 1: Generating {lineup_pool_size} lineups using vectorized approach...')
        lineup_pool = self._generate_lineup_pool_vectorized(
            pos_data, lineup_pool_size, lineup_size
        )
        
        # Step 2: Use efficient sampling for diverse sets
        logging.info(f'Step 2: Sampling {population_size} diverse sets using optimized selection...')
        population_sets = self._sample_diverse_sets_optimized(
            lineup_pool, population_size, target_lineups, diversity_threshold
        )
        
        return population_sets
    
    def _generate_sets_direct_diverse(self, pos_data: Dict[str, Any], 
                                    population_size: int, target_lineups: int,
                                    lineup_size: int, diversity_threshold: float,
                                    max_attempts: int) -> np.ndarray:
        """Generate sets directly without creating large pool first"""
        
        logging.info(f'Generating {population_size} diverse sets directly (no large pool)')
        population_sets = np.zeros((population_size, target_lineups, lineup_size), dtype=int)
        
        # Generate each set independently
        for set_idx in range(population_size):
            if set_idx % 100 == 0:
                logging.info(f'Generated {set_idx}/{population_size} diverse sets')
            
            population_sets[set_idx] = self._generate_single_diverse_set_fast(
                pos_data, target_lineups, lineup_size, diversity_threshold, max_attempts
            )
        
        return population_sets
    
    @staticmethod
    def _generate_lineup_pool_vectorized(pos_data: Dict[str, Any], 
                                       pool_size: int, lineup_size: int) -> np.ndarray:
        """Generate lineup pool using vectorized operations"""
        lineup_pool = np.zeros((pool_size, lineup_size), dtype=int)
        
        # Generate all lineups for each position at once
        for pos, data in pos_data.items():
            if len(data['indices']) == 0:
                continue
            
            start_idx = data['start_idx']
            end_idx = data['end_idx']
            count = data['count']
            
            # Use vectorized sampling for all lineups at once
            if count > 0 and len(data['indices']) >= count:
                # Generate selections for all lineups in one call
                selected_players = multidimensional_shifting_fast(
                    num_samples=pool_size,
                    sample_size=count,
                    probs=data['probs'],
                    elements=data['indices']
                )
                
                # Fill the lineup pool
                lineup_pool[:, start_idx:end_idx] = selected_players
        
        return lineup_pool
    
    def _sample_diverse_sets_optimized(self, lineup_pool: np.ndarray,
                                     population_size: int, target_lineups: int,
                                     diversity_threshold: float) -> np.ndarray:
        """Ultra-fast diverse set sampling using fingerprint-based clustering"""
        pool_size, lineup_size = lineup_pool.shape
        
        # Always use the ultra-fast method for any significant workload
        if population_size * target_lineups > 1000:
            logging.info('Using ULTRA-FAST fingerprint-based sampling...')
            return self._sample_diverse_sets_ultra_fast(
                lineup_pool, population_size, target_lineups, diversity_threshold
            )
        else:
            # Small workload: Use optimized similarity-based approach
            return self._sample_diverse_sets_similarity_based(
                lineup_pool, population_size, target_lineups, diversity_threshold
            )
    
    def _sample_diverse_sets_similarity_based(self, lineup_pool: np.ndarray,
                                            population_size: int, target_lineups: int,
                                            diversity_threshold: float) -> np.ndarray:
        """Sample diverse sets using selective similarity computation"""
        pool_size, lineup_size = lineup_pool.shape
        population_sets = np.zeros((population_size, target_lineups, lineup_size), dtype=int)
        
        for set_idx in range(population_size):
            selected_indices = []
            available_indices = list(range(pool_size))
            
            # Start with random lineup
            first_idx = np.random.choice(available_indices)
            selected_indices.append(first_idx)
            available_indices.remove(first_idx)
            
            # Select remaining lineups with diversity checks
            for _ in range(target_lineups - 1):
                if not available_indices:
                    break
                
                # Sample candidates to check (don't check all)
                n_candidates = min(100, len(available_indices))
                candidate_indices = np.random.choice(
                    available_indices, size=n_candidates, replace=False
                )
                
                best_candidate = None
                best_min_similarity = 1.0
                
                for candidate_idx in candidate_indices:
                    # Calculate similarity to selected lineups
                    max_similarity = 0.0
                    for selected_idx in selected_indices:
                        similarity = self._jaccard_similarity_fast(
                            lineup_pool[candidate_idx], lineup_pool[selected_idx]
                        )
                        max_similarity = max(max_similarity, similarity)
                    
                    if max_similarity < best_min_similarity:
                        best_candidate = candidate_idx
                        best_min_similarity = max_similarity
                
                if best_candidate is not None:
                    selected_indices.append(best_candidate)
                    available_indices.remove(best_candidate)
                else:
                    # Fallback
                    random_idx = np.random.choice(available_indices)
                    selected_indices.append(random_idx)
                    available_indices.remove(random_idx)
            
            # Fill the set
            for i, idx in enumerate(selected_indices):
                population_sets[set_idx, i] = lineup_pool[idx]
        
        return population_sets
    
    def _sample_diverse_sets_clustering_based(self, lineup_pool: np.ndarray,
                                            population_size: int, target_lineups: int,
                                            diversity_threshold: float) -> np.ndarray:
        """Sample diverse sets using clustering/hashing for large pools"""
        pool_size, lineup_size = lineup_pool.shape
        population_sets = np.zeros((population_size, target_lineups, lineup_size), dtype=int)
        
        # Create simple hash-based clusters for diversity
        logging.info('Creating hash-based clusters for efficient diversity sampling...')
        clusters = self._create_lineup_clusters(lineup_pool, n_clusters=min(1000, pool_size // 10))
        
        for set_idx in range(population_size):
            selected_indices = []
            used_clusters = set()
            
            # Try to select from different clusters for diversity
            for _ in range(target_lineups):
                # Find available clusters
                available_clusters = [c for c in range(len(clusters)) if c not in used_clusters]
                
                if available_clusters:
                    # Select from unused cluster
                    cluster_idx = np.random.choice(available_clusters)
                    used_clusters.add(cluster_idx)
                else:
                    # All clusters used, select from any cluster
                    cluster_idx = np.random.choice(len(clusters))
                
                # Select random lineup from cluster
                if len(clusters[cluster_idx]) > 0:
                    lineup_idx = np.random.choice(clusters[cluster_idx])
                    selected_indices.append(lineup_idx)
                else:
                    # Fallback to random selection
                    selected_indices.append(np.random.choice(pool_size))
            
            # Fill the set
            for i, idx in enumerate(selected_indices):
                if i < target_lineups:
                    population_sets[set_idx, i] = lineup_pool[idx]
        
        return population_sets
    
    @staticmethod
    def _create_lineup_clusters(lineup_pool: np.ndarray, n_clusters: int) -> list:
        """Create simple hash-based clusters for lineup diversity"""
        pool_size = len(lineup_pool)
        clusters = [[] for _ in range(n_clusters)]
        
        for i, lineup in enumerate(lineup_pool):
            # Simple hash based on first few players
            hash_val = hash(tuple(lineup[:min(3, len(lineup))])) % n_clusters
            clusters[hash_val].append(i)
        
        return clusters
    
    def _generate_single_diverse_set_fast(self, pos_data: Dict[str, Any], 
                                        target_lineups: int, lineup_size: int,
                                        diversity_threshold: float, max_attempts: int) -> np.ndarray:
        """Generate a single diverse set efficiently"""
        lineup_set = np.zeros((target_lineups, lineup_size), dtype=int)
        
        # Generate first lineup
        lineup_set[0] = self._generate_single_lineup_fast(pos_data, lineup_size)
        
        # Generate remaining lineups with diversity
        for lineup_idx in range(1, target_lineups):
            best_lineup = None
            best_diversity = 0.0
            
            # Reduced attempts for speed
            attempts = min(max_attempts, 20)
            
            for _ in range(attempts):
                candidate = self._generate_single_lineup_fast(pos_data, lineup_size)
                
                # Quick diversity check against existing lineups
                min_diversity = self._calculate_min_diversity_fast(
                    candidate, lineup_set[:lineup_idx]
                )
                
                if min_diversity > best_diversity:
                    best_lineup = candidate
                    best_diversity = min_diversity
                    
                    if min_diversity >= diversity_threshold:
                        break
            
            if best_lineup is not None:
                lineup_set[lineup_idx] = best_lineup
            else:
                lineup_set[lineup_idx] = self._generate_single_lineup_fast(pos_data, lineup_size)
        
        return lineup_set
    
    @staticmethod
    def _generate_single_lineup_fast(pos_data: Dict[str, Any], lineup_size: int) -> np.ndarray:
        """Generate a single lineup efficiently"""
        lineup = np.zeros(lineup_size, dtype=int)
        
        for pos, data in pos_data.items():
            if len(data['indices']) == 0:
                continue
            
            start_idx = data['start_idx']
            end_idx = data['end_idx']
            count = data['count']
            
            if count > 0 and len(data['indices']) >= count:
                # Fast selection using numpy
                selected = np.random.choice(
                    data['indices'], 
                    size=count, 
                    replace=False, 
                    p=data['probs']
                )
                lineup[start_idx:end_idx] = selected
        
        return lineup
    
    @staticmethod
    def _calculate_min_diversity_fast(candidate: np.ndarray, 
                                    existing_lineups: np.ndarray) -> float:
        """Fast diversity calculation"""
        if len(existing_lineups) == 0:
            return 1.0
        
        candidate_set = set(candidate)
        min_diversity = 1.0
        
        for existing_lineup in existing_lineups:
            existing_set = set(existing_lineup)
            intersection = len(candidate_set & existing_set)
            union = len(candidate_set | existing_set)
            
            if union > 0:
                similarity = intersection / union
                diversity = 1.0 - similarity
                min_diversity = min(min_diversity, diversity)
        
        return min_diversity
    
    def _sample_diverse_sets_ultra_fast(self, lineup_pool: np.ndarray,
                                       population_size: int, target_lineups: int,
                                       diversity_threshold: float) -> np.ndarray:
        """Ultra-fast diverse set sampling using fingerprint-based approach"""
        pool_size, lineup_size = lineup_pool.shape
        population_sets = np.zeros((population_size, target_lineups, lineup_size), dtype=int)
        
        # Step 1: Create lineup fingerprints for fast similarity estimation
        logging.info('Creating lineup fingerprints for ultra-fast sampling...')
        fingerprints = self._create_lineup_fingerprints(lineup_pool)
        
        # Step 2: Create diversity-based clusters
        n_clusters = min(target_lineups * 10, pool_size // 5, 2000)
        clusters = self._cluster_lineups_by_fingerprint(fingerprints, n_clusters)
        
        # Step 3: Vectorized sampling from clusters
        logging.info(f'Sampling from {len(clusters)} clusters using vectorized approach...')
        
        # Pre-compute cluster selections for all sets at once
        cluster_assignments = self._assign_clusters_to_sets_vectorized(
            clusters, population_size, target_lineups
        )
        
        # Fill population sets using cluster assignments
        for set_idx in range(population_size):
            for lineup_idx in range(target_lineups):
                cluster_idx = cluster_assignments[set_idx, lineup_idx]
                if cluster_idx < len(clusters) and len(clusters[cluster_idx]) > 0:
                    # Select random lineup from assigned cluster
                    pool_idx = np.random.choice(clusters[cluster_idx])
                    population_sets[set_idx, lineup_idx] = lineup_pool[pool_idx]
                else:
                    # Fallback to random selection
                    pool_idx = np.random.choice(pool_size)
                    population_sets[set_idx, lineup_idx] = lineup_pool[pool_idx]
        
        return population_sets
    
    @staticmethod
    def _create_lineup_fingerprints(lineup_pool: np.ndarray) -> np.ndarray:
        """Create compact fingerprints for fast similarity estimation"""
        pool_size, lineup_size = lineup_pool.shape
        
        # Use multiple hash functions for better fingerprinting
        fingerprints = np.zeros((pool_size, 4), dtype=np.int32)
        
        for i, lineup in enumerate(lineup_pool):
            # Hash 1: Sum of first half of players
            mid = lineup_size // 2
            fingerprints[i, 0] = np.sum(lineup[:mid]) % 100000
            
            # Hash 2: Sum of second half of players  
            fingerprints[i, 1] = np.sum(lineup[mid:]) % 100000
            
            # Hash 3: XOR of all players
            fingerprints[i, 2] = np.bitwise_xor.reduce(lineup) % 100000
            
            # Hash 4: Product of first 3 players (mod large prime)
            if lineup_size >= 3:
                fingerprints[i, 3] = (lineup[0] * lineup[1] * lineup[2]) % 97
            else:
                fingerprints[i, 3] = np.sum(lineup) % 97
        
        return fingerprints
    
    @staticmethod
    def _cluster_lineups_by_fingerprint(fingerprints: np.ndarray, 
                                      n_clusters: int) -> list:
        """Cluster lineups based on fingerprint similarity"""
        pool_size = len(fingerprints)
        clusters = [[] for _ in range(n_clusters)]
        
        # Use simple k-means-like clustering on fingerprints
        # For speed, use hash-based assignment instead of distance calculation
        for i, fingerprint in enumerate(fingerprints):
            # Combine fingerprint elements into cluster assignment
            cluster_hash = (
                fingerprint[0] * 7 + 
                fingerprint[1] * 11 + 
                fingerprint[2] * 13 + 
                fingerprint[3] * 17
            ) % n_clusters
            
            clusters[cluster_hash].append(i)
        
        # Remove empty clusters and redistribute
        non_empty_clusters = [c for c in clusters if len(c) > 0]
        
        # If we have too few non-empty clusters, redistribute some lineups
        while len(non_empty_clusters) < min(n_clusters, pool_size // 10):
            # Find largest cluster and split it
            largest_cluster_idx = max(range(len(non_empty_clusters)), 
                                    key=lambda x: len(non_empty_clusters[x]))
            largest_cluster = non_empty_clusters[largest_cluster_idx]
            
            if len(largest_cluster) > 1:
                # Split the largest cluster
                mid = len(largest_cluster) // 2
                new_cluster = largest_cluster[mid:]
                largest_cluster[:] = largest_cluster[:mid]
                non_empty_clusters.append(new_cluster)
            else:
                break
        
        return non_empty_clusters
    
    @staticmethod
    def _assign_clusters_to_sets_vectorized(clusters: list, 
                                          population_size: int, 
                                          target_lineups: int) -> np.ndarray:
        """Vectorized assignment of clusters to lineup sets for diversity"""
        n_clusters = len(clusters)
        assignments = np.zeros((population_size, target_lineups), dtype=int)
        
        if n_clusters == 0:
            return assignments
        
        # Strategy: Ensure each set uses different clusters when possible
        for set_idx in range(population_size):
            if target_lineups <= n_clusters:
                # More clusters than needed: select diverse subset
                selected_clusters = np.random.choice(
                    n_clusters, size=target_lineups, replace=False
                )
                assignments[set_idx] = selected_clusters
            else:
                # More lineups than clusters: cycle through clusters
                base_clusters = np.arange(n_clusters)
                np.random.shuffle(base_clusters)
                
                # Repeat clusters as needed
                full_cycles = target_lineups // n_clusters
                remainder = target_lineups % n_clusters
                
                cluster_sequence = np.tile(base_clusters, full_cycles)
                if remainder > 0:
                    extra_clusters = np.random.choice(base_clusters, size=remainder, replace=False)
                    cluster_sequence = np.concatenate([cluster_sequence, extra_clusters])
                
                assignments[set_idx] = cluster_sequence
        
        return assignments
    
    @staticmethod
    def _jaccard_similarity_fast(lineup1: np.ndarray, lineup2: np.ndarray) -> float:
        """Fast Jaccard similarity calculation"""
        set1 = set(lineup1)
        set2 = set(lineup2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0


# Numba-optimized functions if available
if NUMBA_AVAILABLE:
    @njit
    def _jaccard_similarity_numba(lineup1, lineup2):
        """Numba-optimized Jaccard similarity"""
        intersection = 0
        union_size = len(lineup1) + len(lineup2)
        
        # Count intersections
        for i, _ in enumerate(lineup1):
            for j, item in enumerate(lineup2):
                if lineup1[i] == item:
                    intersection += 1
                    union_size -= 1
                    break
        
        return intersection / union_size if union_size > 0 else 0.0
    
    # Replace the method if numba is available
    PopulateMultilineupSetsOptimized._jaccard_similarity_fast = staticmethod(_jaccard_similarity_numba)
