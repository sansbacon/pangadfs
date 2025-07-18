# pangadfs/gui/utils/config_manager.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


class ConfigManager:
    """Manages configuration loading, saving, and validation"""
    
    def __init__(self):
        self.config_dir = Path.home() / '.pangadfs'
        self.config_dir.mkdir(exist_ok=True)
        self.default_config_file = self.config_dir / 'default_config.json'
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration optimized for multilineup generation"""
        default_config = {
            'ga_settings': {
                'csvpth': '',
                'points_column': 'proj',
                'position_column': 'pos',
                'salary_column': 'salary',
                # Optimized for set-based multilineup generation (aggressive balanced preset)
                'population_size': 1000,  # Balanced: much better quality while still efficient
                'n_generations': 100,     # More generations for better convergence
                'mutation_rate': 0.15,    # Higher for better exploration in multilineup
                'elite_divisor': 5,
                'stop_criteria': 10,
                'crossover_method': 'uniform',
                'select_method': 'tournament',
                'elite_method': 'fittest',
                'verbose': True,
                'enable_profiling': True,
                # Multilineup defaults - enabled by default
                'target_lineups': 10,    # Good balance of diversity and speed
                'diversity_method': 'jaccard',
                # Multi-objective optimizer settings (optimal defaults)
                'optimizer_type': 'multi_objective',  # Use multi-objective optimization
                'top_k_lineups': 10,                  # Top K lineups to optimize
                'multi_objective_weights': (0.6, 0.3, 0.1),  # (top_k, total, diversity)
                # Set-based optimizer fallback settings
                'set_diversity_weight': 0.05,         # Quality-first: minimal diversity penalty
                'tournament_size': 3,                  # Optimal for crossover
                'lineup_pool_size': 100000,           # Balanced preset: much better quality sampling
                # Post-processing fallback settings
                'diversity_weight': 0.05,             # Quality-first: minimal diversity penalty
                'min_overlap_threshold': 0.7,   # For post-processing approach
            },
            'site_settings': {
                'salary_cap': 50000,
                'lineup_size': 9,
                'posmap': {
                    'QB': 1,
                    'RB': 2,
                    'WR': 3,
                    'TE': 1,
                    'DST': 1,
                    'FLEX': 1
                },
                'flex_positions': ('RB', 'WR', 'TE'),
                'posfilter': {
                    'QB': 14,
                    'RB': 8,
                    'WR': 8,
                    'TE': 5,
                    'DST': 4,
                    'FLEX': 8
                }
            }
        }
        
        # Try to load saved default config
        if self.default_config_file.exists():
            try:
                with open(self.default_config_file, 'r') as f:
                    saved_config = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    self._deep_update(default_config, saved_config)
            except Exception:
                pass  # Use defaults if loading fails
        
        return default_config
    
    def load_config(self, filename: str) -> Dict[str, Any]:
        """Load configuration from file"""
        with open(filename, 'r') as f:
            config = json.load(f)
        
        # Ensure all required keys exist by merging with defaults
        default_config = self.get_default_config()
        self._deep_update(default_config, config)
        
        return default_config
    
    def save_config(self, config: Dict[str, Any], filename: str = None) -> str:
        """Save configuration to file"""
        if filename is None:
            # Generate default filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = str(self.config_dir / f'config_{timestamp}.json')
        
        # Convert Path objects to strings for JSON serialization
        json_config = self._prepare_for_json(config)
        
        with open(filename, 'w') as f:
            json.dump(json_config, f, indent=2)
        
        # Also save as default config
        with open(self.default_config_file, 'w') as f:
            json.dump(json_config, f, indent=2)
        
        return filename
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Validate GA settings
        ga_settings = config.get('ga_settings', {})
        
        # Required file path
        csvpth = ga_settings.get('csvpth')
        if not csvpth:
            errors.append("Player pool CSV file is required")
        elif not Path(csvpth).exists():
            errors.append(f"Player pool CSV file not found: {csvpth}")
        
        # Numeric validations
        numeric_validations = [
            ('population_size', 'Population size', 1, 100000),
            ('n_generations', 'Number of generations', 1, 1000),
            ('elite_divisor', 'Elite divisor', 1, 100),
            ('stop_criteria', 'Stop criteria', 1, 100),
            ('target_lineups', 'Target lineups', 1, 1000)
        ]
        
        for key, name, min_val, max_val in numeric_validations:
            value = ga_settings.get(key, 0)
            if not isinstance(value, int) or value < min_val or value > max_val:
                errors.append(f"{name} must be an integer between {min_val} and {max_val}")
        
        # Float validations
        float_validations = [
            ('mutation_rate', 'Mutation rate', 0.0, 1.0),
            ('diversity_weight', 'Diversity weight', 0.0, 1.0),
            ('min_overlap_threshold', 'Min overlap threshold', 0.0, 1.0)
        ]
        
        for key, name, min_val, max_val in float_validations:
            value = ga_settings.get(key, 0.0)
            if not isinstance(value, (int, float)) or value < min_val or value > max_val:
                errors.append(f"{name} must be a number between {min_val} and {max_val}")
        
        # Column name validations
        required_columns = ['points_column', 'position_column', 'salary_column']
        for col in required_columns:
            if not ga_settings.get(col):
                errors.append(f"{col.replace('_', ' ').title()} is required")
        
        # Validate site settings
        site_settings = config.get('site_settings', {})
        
        # Salary cap validation
        salary_cap = site_settings.get('salary_cap', 0)
        if not isinstance(salary_cap, int) or salary_cap < 1000:
            errors.append("Salary cap must be at least 1000")
        
        # Lineup size validation
        lineup_size = site_settings.get('lineup_size', 0)
        if not isinstance(lineup_size, int) or lineup_size < 1 or lineup_size > 20:
            errors.append("Lineup size must be between 1 and 20")
        
        # Position map validation
        posmap = site_settings.get('posmap', {})
        if not posmap:
            errors.append("Position map is required")
        else:
            total_positions = sum(posmap.values())
            if total_positions != lineup_size:
                errors.append(f"Position map total ({total_positions}) must equal lineup size ({lineup_size})")
        
        # Flex positions validation
        flex_positions = site_settings.get('flex_positions', ())
        if 'FLEX' in posmap and posmap['FLEX'] > 0 and not flex_positions:
            errors.append("Flex positions must be specified when FLEX position is used")
        
        return errors
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """Deep update base_dict with update_dict"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _prepare_for_json(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare config for JSON serialization by converting Path objects to strings"""
        json_config = {}
        
        for key, value in config.items():
            if isinstance(value, dict):
                json_config[key] = self._prepare_for_json(value)
            elif isinstance(value, Path):
                json_config[key] = str(value)
            elif isinstance(value, tuple):
                json_config[key] = list(value)
            else:
                json_config[key] = value
        
        return json_config
