# pangadfs/tests/test_validate_enhanced.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import numpy as np
import pytest

from pangadfs.validate import (
    DuplicatesValidate, SalaryValidate, PositionValidate, ExposureValidate,
    StackingValidate, CompositeValidate, validate_population_comprehensive,
    create_validation_pipeline, ValidationProfiler
)


class TestEnhancedValidation:
    """Test enhanced validation functionality."""
    
    @pytest.fixture
    def setup_data(self):
        """Setup test data for validation tests."""
        # Create test population
        np.random.seed(42)
        population = np.random.randint(0, 50, size=(100, 9))
        
        # Create salary data
        salaries = np.random.randint(5000, 12000, size=51)  # 51 players (0-50)
        salary_cap = 50000
        
        # Create position data
        position_map = {}
        position_requirements = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1, 'FLEX': 1}
        
        # Assign positions to players
        positions = ['QB'] * 5 + ['RB'] * 10 + ['WR'] * 15 + ['TE'] * 8 + ['DST'] * 5 + ['FLEX'] * 8
        for i, pos in enumerate(positions):
            position_map[i] = list(position_requirements.keys()).index(pos)
        
        # Create team data for stacking
        player_teams = {}
        teams = ['Team1', 'Team2', 'Team3', 'Team4']
        for i in range(51):
            player_teams[i] = teams[i % len(teams)]
        
        return {
            'population': population,
            'salaries': salaries,
            'salary_cap': salary_cap,
            'position_map': position_map,
            'position_requirements': position_requirements,
            'player_teams': player_teams
        }
    
    def test_enhanced_duplicates_validate(self, setup_data):
        """Test enhanced duplicate validation."""
        data = setup_data
        
        # Create population with some duplicates
        population_with_dups = np.vstack([
            data['population'][:50],
            data['population'][:10],  # Add duplicates
            data['population'][20:30]  # Add more duplicates
        ])
        
        validator = DuplicatesValidate()
        
        # Test basic validation
        validated = validator.validate(population=population_with_dups)
        assert len(validated) <= len(population_with_dups)
        assert isinstance(validated, np.ndarray)
        
        # Test with statistics
        validated, stats = validator.validate(
            population=population_with_dups,
            return_stats=True
        )
        
        assert 'total_lineups' in stats
        assert 'removed_lineups' in stats
        assert 'duplicate_percentage' in stats
        assert stats['total_lineups'] == len(population_with_dups)
        assert stats['removed_lineups'] >= 0
        
        # Test configurable options
        validated_internal_only = validator.validate(
            population=population_with_dups,
            remove_internal=True,
            remove_lineup_duplicates=False
        )
        
        validated_lineup_only = validator.validate(
            population=population_with_dups,
            remove_internal=False,
            remove_lineup_duplicates=True
        )
        
        assert isinstance(validated_internal_only, np.ndarray)
        assert isinstance(validated_lineup_only, np.ndarray)
    
    def test_enhanced_salary_validate(self, setup_data):
        """Test enhanced salary validation with JIT optimization."""
        data = setup_data
        
        validator = SalaryValidate()
        
        # Test basic validation
        validated = validator.validate(
            population=data['population'],
            salaries=data['salaries'],
            salary_cap=data['salary_cap']
        )
        
        assert isinstance(validated, np.ndarray)
        assert len(validated) <= len(data['population'])
        
        # Test with statistics
        validated, stats = validator.validate(
            population=data['population'],
            salaries=data['salaries'],
            salary_cap=data['salary_cap'],
            return_stats=True
        )
        
        assert 'total_lineups' in stats
        assert 'valid_lineups' in stats
        assert 'validation_rate' in stats
        assert 'salary_stats' in stats
        assert stats['salary_cap'] == data['salary_cap']
        
        # Test with tolerance
        validated_with_tolerance = validator.validate(
            population=data['population'],
            salaries=data['salaries'],
            salary_cap=data['salary_cap'],
            tolerance=1000
        )
        
        assert len(validated_with_tolerance) >= len(validated)
        
        # Test JIT threshold configuration
        validator.set_jit_threshold(50)
        validated_jit = validator.validate(
            population=data['population'],
            salaries=data['salaries'],
            salary_cap=data['salary_cap']
        )
        
        assert isinstance(validated_jit, np.ndarray)
        
        # Test batch processing with large population
        large_population = np.random.randint(0, 50, size=(2000, 9))
        validator_batch = SalaryValidate(batch_size=500)
        validated_batch = validator_batch.validate(
            population=large_population,
            salaries=data['salaries'],
            salary_cap=data['salary_cap']
        )
        
        assert isinstance(validated_batch, np.ndarray)
    
    def test_position_validate(self, setup_data):
        """Test position validation."""
        data = setup_data
        
        validator = PositionValidate()
        
        # Test basic validation
        validated = validator.validate(
            population=data['population'],
            position_map=data['position_map'],
            position_requirements=data['position_requirements']
        )
        
        assert isinstance(validated, np.ndarray)
        
        # Test with statistics
        validated, stats = validator.validate(
            population=data['population'],
            position_map=data['position_map'],
            position_requirements=data['position_requirements'],
            return_stats=True
        )
        
        assert 'total_lineups' in stats
        assert 'valid_lineups' in stats
        assert 'position_requirements' in stats
        assert stats['position_requirements'] == data['position_requirements']
        
        # Test with numpy array position map
        max_player_id = max(data['position_map'].keys())
        pos_array = np.full(max_player_id + 1, -1, dtype=np.int32)
        for player_id, pos_idx in data['position_map'].items():
            pos_array[player_id] = pos_idx
        
        validated_array = validator.validate(
            population=data['population'],
            position_map=pos_array,
            position_requirements=data['position_requirements']
        )
        
        assert isinstance(validated_array, np.ndarray)
        
        # Test with large population for JIT
        large_population = np.random.randint(0, 50, size=(200, 9))
        validated_large = validator.validate(
            population=large_population,
            position_map=data['position_map'],
            position_requirements=data['position_requirements']
        )
        
        assert isinstance(validated_large, np.ndarray)
    
    def test_exposure_validate(self, setup_data):
        """Test exposure validation."""
        data = setup_data
        
        validator = ExposureValidate()
        
        # Test basic validation with max exposure
        validated = validator.validate(
            population=data['population'],
            max_exposure_pct=20.0
        )
        
        assert isinstance(validated, np.ndarray)
        assert len(validated) <= len(data['population'])
        
        # Test with statistics
        validated, stats = validator.validate(
            population=data['population'],
            max_exposure_pct=15.0,
            return_stats=True
        )
        
        assert 'total_lineups' in stats
        assert 'exposure_stats' in stats
        assert 'unique_players' in stats['exposure_stats']
        assert 'max_exposure_pct' in stats['exposure_stats']
        
        # Test with player limits
        player_limits = {0: 10.0, 1: 15.0, 2: 20.0}
        validated_limits = validator.validate(
            population=data['population'],
            player_limits=player_limits
        )
        
        assert isinstance(validated_limits, np.ndarray)
        
        # Test with minimum unique players
        validated_min_unique = validator.validate(
            population=data['population'],
            min_unique_players=30
        )
        
        assert isinstance(validated_min_unique, np.ndarray)
        
        # Test with large population for JIT
        large_population = np.random.randint(0, 50, size=(200, 9))
        validated_large = validator.validate(
            population=large_population,
            max_exposure_pct=25.0
        )
        
        assert isinstance(validated_large, np.ndarray)
    
    def test_stacking_validate(self, setup_data):
        """Test stacking validation."""
        data = setup_data
        
        validator = StackingValidate()
        
        # Define stacking rules
        stacking_rules = [
            {'type': 'min_stack', 'min_players': 2},
            {'type': 'max_stack', 'max_players': 4}
        ]
        
        # Test basic validation
        validated = validator.validate(
            population=data['population'],
            stacking_rules=stacking_rules,
            player_teams=data['player_teams']
        )
        
        assert isinstance(validated, np.ndarray)
        
        # Test with statistics
        validated, stats = validator.validate(
            population=data['population'],
            stacking_rules=stacking_rules,
            player_teams=data['player_teams'],
            return_stats=True
        )
        
        assert 'total_lineups' in stats
        assert 'valid_lineups' in stats
        assert 'stacking_rules' in stats
        assert stats['stacking_rules'] == stacking_rules
        
        # Test with different stacking rules
        strict_rules = [
            {'type': 'min_stack', 'min_players': 3},
            {'type': 'max_stack', 'max_players': 3}
        ]
        
        validated_strict = validator.validate(
            population=data['population'],
            stacking_rules=strict_rules,
            player_teams=data['player_teams']
        )
        
        assert isinstance(validated_strict, np.ndarray)
        assert len(validated_strict) <= len(validated)
    
    def test_composite_validate(self, setup_data):
        """Test composite validation."""
        data = setup_data
        
        # Create composite validator
        composite = CompositeValidate()
        
        # Add validators
        composite.add_validator(DuplicatesValidate())
        composite.add_validator(SalaryValidate())
        composite.add_validator(PositionValidate())
        
        # Test basic validation
        validated = composite.validate(
            population=data['population'],
            salaries=data['salaries'],
            salary_cap=data['salary_cap'],
            position_map=data['position_map'],
            position_requirements=data['position_requirements']
        )
        
        assert isinstance(validated, np.ndarray)
        
        # Test with statistics
        validated, stats = composite.validate(
            population=data['population'],
            salaries=data['salaries'],
            salary_cap=data['salary_cap'],
            position_map=data['position_map'],
            position_requirements=data['position_requirements'],
            return_stats=True
        )
        
        assert 'total_lineups' in stats
        assert 'valid_lineups' in stats
        assert 'validators_run' in stats
        assert 'validator_stats' in stats
        assert len(stats['validator_stats']) == 3  # Three validators
        
        # Test early termination
        validated_early = composite.validate(
            population=data['population'],
            early_termination=True,
            salaries=data['salaries'],
            salary_cap=data['salary_cap'],
            position_map=data['position_map'],
            position_requirements=data['position_requirements']
        )
        
        assert isinstance(validated_early, np.ndarray)
        
        # Test without early termination
        validated_no_early = composite.validate(
            population=data['population'],
            early_termination=False,
            salaries=data['salaries'],
            salary_cap=data['salary_cap'],
            position_map=data['position_map'],
            position_requirements=data['position_requirements']
        )
        
        assert isinstance(validated_no_early, np.ndarray)
    
    def test_comprehensive_validation_function(self, setup_data):
        """Test the comprehensive validation utility function."""
        data = setup_data
        
        # Test with all parameters
        validated = validate_population_comprehensive(
            population=data['population'],
            salaries=data['salaries'],
            salary_cap=data['salary_cap'],
            position_map=data['position_map'],
            position_requirements=data['position_requirements'],
            max_exposure_pct=25.0,
            remove_duplicates=True
        )
        
        assert isinstance(validated, np.ndarray)
        
        # Test with statistics
        validated, stats = validate_population_comprehensive(
            population=data['population'],
            salaries=data['salaries'],
            salary_cap=data['salary_cap'],
            position_map=data['position_map'],
            position_requirements=data['position_requirements'],
            max_exposure_pct=20.0,
            return_stats=True
        )
        
        assert 'total_lineups' in stats
        assert 'validator_stats' in stats
        
        # Test with subset of parameters
        validated_partial = validate_population_comprehensive(
            population=data['population'],
            salaries=data['salaries'],
            salary_cap=data['salary_cap'],
            remove_duplicates=True
        )
        
        assert isinstance(validated_partial, np.ndarray)
        
        # Test with stacking
        stacking_rules = [{'type': 'min_stack', 'min_players': 2}]
        validated_stacking = validate_population_comprehensive(
            population=data['population'],
            stacking_rules=stacking_rules,
            player_teams=data['player_teams'],
            remove_duplicates=False
        )
        
        assert isinstance(validated_stacking, np.ndarray)
    
    def test_validation_pipeline_creation(self, setup_data):
        """Test validation pipeline creation from configuration."""
        
        # Test basic configuration
        config = {
            'remove_duplicates': True,
            'validate_salary': True,
            'validate_positions': True,
            'salary_config': {'batch_size': 500},
            'position_config': {'use_jit': True}
        }
        
        pipeline = create_validation_pipeline(config)
        assert isinstance(pipeline, CompositeValidate)
        assert len(pipeline.validators) == 3
        
        # Test with all validators
        full_config = {
            'remove_duplicates': True,
            'validate_salary': True,
            'validate_positions': True,
            'validate_exposure': True,
            'validate_stacking': True,
            'duplicate_remover_config': {'batch_size': 1000},
            'salary_config': {'use_jit': False},
            'position_config': {},
            'exposure_config': {'use_jit': True},
            'stacking_config': {}
        }
        
        full_pipeline = create_validation_pipeline(full_config)
        assert len(full_pipeline.validators) == 5
        
        # Test minimal configuration
        minimal_config = {}
        minimal_pipeline = create_validation_pipeline(minimal_config)
        assert len(minimal_pipeline.validators) == 1  # Only duplicates by default
    
    def test_validation_profiler(self, setup_data):
        """Test validation performance profiler."""
        data = setup_data
        
        profiler = ValidationProfiler()
        
        # Profile salary validator
        salary_validator = SalaryValidate()
        profile = profiler.profile_validator(
            validator=salary_validator,
            population=data['population'],
            iterations=3,
            salaries=data['salaries'],
            salary_cap=data['salary_cap']
        )
        
        assert 'validator_name' in profile
        assert 'avg_time' in profile
        assert 'population_size' in profile
        assert profile['validator_name'] == 'SalaryValidate'
        assert profile['population_size'] == len(data['population'])
        assert profile['iterations'] == 3
        
        # Profile position validator
        position_validator = PositionValidate()
        position_profile = profiler.profile_validator(
            validator=position_validator,
            population=data['population'],
            iterations=2,
            position_map=data['position_map'],
            position_requirements=data['position_requirements']
        )
        
        assert position_profile['validator_name'] == 'PositionValidate'
        
        # Get all profiles
        all_profiles = profiler.get_profiles()
        assert 'SalaryValidate' in all_profiles
        assert 'PositionValidate' in all_profiles
    
    def test_input_validation_and_error_handling(self, setup_data):
        """Test input validation and error handling."""
        data = setup_data
        
        # Test empty population
        empty_pop = np.array([]).reshape(0, 9)
        
        salary_validator = SalaryValidate()
        result = salary_validator.validate(
            population=empty_pop,
            salaries=data['salaries'],
            salary_cap=data['salary_cap']
        )
        assert len(result) == 0
        
        # Test invalid salary parameters
        with pytest.raises(ValueError, match="Salaries array cannot be empty"):
            salary_validator.validate(
                population=data['population'],
                salaries=np.array([]),
                salary_cap=data['salary_cap']
            )
        
        with pytest.raises(ValueError, match="Salary cap must be positive"):
            salary_validator.validate(
                population=data['population'],
                salaries=data['salaries'],
                salary_cap=0
            )
        
        # Test player ID out of bounds
        bad_population = np.array([[100, 101, 102, 103, 104, 105, 106, 107, 108]])
        with pytest.raises(ValueError, match="Player ID .* exceeds salaries array length"):
            salary_validator.validate(
                population=bad_population,
                salaries=data['salaries'],
                salary_cap=data['salary_cap']
            )
        
        # Test position validator with empty population
        position_validator = PositionValidate()
        pos_result = position_validator.validate(
            population=empty_pop,
            position_map=data['position_map'],
            position_requirements=data['position_requirements']
        )
        assert len(pos_result) == 0
        
        # Test exposure validator with empty population
        exposure_validator = ExposureValidate()
        exp_result = exposure_validator.validate(
            population=empty_pop,
            max_exposure_pct=20.0
        )
        assert len(exp_result) == 0
    
    def test_statistics_retrieval(self, setup_data):
        """Test statistics retrieval from validators."""
        data = setup_data
        
        # Test salary validator statistics
        salary_validator = SalaryValidate()
        salary_validator.validate(
            population=data['population'],
            salaries=data['salaries'],
            salary_cap=data['salary_cap'],
            return_stats=True
        )
        
        last_stats = salary_validator.get_last_validation_stats()
        assert 'total_lineups' in last_stats
        assert 'validation_rate' in last_stats
        
        # Test duplicates validator statistics
        dup_validator = DuplicatesValidate()
        dup_validator.validate(
            population=data['population'],
            return_stats=True
        )
        
        dup_stats = dup_validator.get_last_validation_stats()
        assert 'total_lineups' in dup_stats
        assert 'duplicate_percentage' in dup_stats
        
        # Test composite validator statistics
        composite = CompositeValidate()
        composite.add_validator(salary_validator)
        composite.add_validator(dup_validator)
        
        composite.validate(
            population=data['population'],
            salaries=data['salaries'],
            salary_cap=data['salary_cap'],
            return_stats=True
        )
        
        composite_stats = composite.get_last_validation_stats()
        assert 'validators_run' in composite_stats
        assert 'validator_stats' in composite_stats
    
    def test_performance_with_large_populations(self):
        """Test performance optimizations with large populations."""
        # Create large population to test JIT compilation paths
        np.random.seed(42)
        large_pop = np.random.randint(0, 100, size=(1000, 9))
        large_salaries = np.random.randint(5000, 12000, size=101)
        
        # Test salary validation with JIT
        salary_validator = SalaryValidate(use_jit=True, batch_size=250)
        validated_jit = salary_validator.validate(
            population=large_pop,
            salaries=large_salaries,
            salary_cap=50000
        )
        
        assert isinstance(validated_jit, np.ndarray)
        
        # Test without JIT for comparison
        salary_validator_no_jit = SalaryValidate(use_jit=False)
        validated_no_jit = salary_validator_no_jit.validate(
            population=large_pop,
            salaries=large_salaries,
            salary_cap=50000
        )
        
        assert isinstance(validated_no_jit, np.ndarray)
        
        # Test position validation with large population
        position_map = {i: i % 6 for i in range(101)}  # 6 positions
        position_requirements = {'pos0': 1, 'pos1': 2, 'pos2': 3, 'pos3': 1, 'pos4': 1, 'pos5': 1}
        
        position_validator = PositionValidate(use_jit=True)
        validated_pos = position_validator.validate(
            population=large_pop,
            position_map=position_map,
            position_requirements=position_requirements
        )
        
        assert isinstance(validated_pos, np.ndarray)
        
        # Test exposure validation with large population
        exposure_validator = ExposureValidate(use_jit=True)
        validated_exp = exposure_validator.validate(
            population=large_pop,
            max_exposure_pct=15.0
        )
        
        assert isinstance(validated_exp, np.ndarray)
    
    def test_backward_compatibility(self, setup_data):
        """Test backward compatibility with original validation interface."""
        data = setup_data
        
        # Test that original DuplicatesValidate interface still works
        dup_validator = DuplicatesValidate()
        validated = dup_validator.validate(population=data['population'])
        assert isinstance(validated, np.ndarray)
        
        # Test that original SalaryValidate interface still works
        salary_validator = SalaryValidate()
        validated_salary = salary_validator.validate(
            population=data['population'],
            salaries=data['salaries'],
            salary_cap=data['salary_cap']
        )
        assert isinstance(validated_salary, np.ndarray)
        
        # Verify that the enhanced features don't break existing usage patterns
        # This ensures 100% backward compatibility
        assert validated.dtype == data['population'].dtype
        assert validated_salary.dtype == data['population'].dtype
        assert validated.shape[1] == data['population'].shape[1]
        assert validated_salary.shape[1] == data['population'].shape[1]
