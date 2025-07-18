# pangadfs/gui/config_panel.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Dict, Any, Callable, Optional
import pandas as pd
from .widgets import PlayerPoolPanel
from .theme_manager import ThemeManager, create_primary_button, create_secondary_button
from .utils.config_manager import ConfigManager


class ConfigPanel:
    """Configuration panel for GA and site settings with mode-based interface"""
    
    def __init__(self, parent: tk.Widget, on_change_callback: Callable[[], None]):
        self.parent = parent
        self.on_change = on_change_callback
        self.config_vars = {}
        self.posmap_vars = {}
        self.posfilter_vars = {}
        
        # Initialize config manager to get default values
        self.config_manager = ConfigManager()
        self.default_config = self.config_manager.get_default_config()
        
        # Track current optimization mode
        self.optimization_mode = tk.StringVar(value='multi_lineup')
        self.optimization_mode.trace('w', self._on_mode_change)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the configuration UI"""
        # Create main horizontal paned window with minimal padding
        main_paned = ttk.PanedWindow(self.parent, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Left side - Configuration settings (35% width)
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=35)
        
        # Right side - Player pool preview (65% width)
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=65)
        
        # Set up left side with scrollable configuration
        self._setup_config_side(left_frame)
        
        # Set up right side with player pool panel
        self._setup_player_pool_side(right_frame)
    
    def _setup_config_side(self, parent):
        """Set up the configuration side (left)"""
        # Create scrollable frame for configuration
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create configuration sections
        self._create_mode_selector()
        self._create_file_section()
        self._create_optimization_settings()
        self._create_site_settings_section()
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    def _setup_player_pool_side(self, parent):
        """Set up the player pool side (right)"""
        # Create player pool panel
        self.player_pool_panel = PlayerPoolPanel(parent, self._on_player_pool_change)
    
    def _create_mode_selector(self):
        """Create the optimization mode selector at the top"""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Optimization Mode", padding=10)
        frame.pack(fill=tk.X, padx=3, pady=3)
        
        # Mode selection with radio buttons for clarity
        mode_frame = ttk.Frame(frame)
        mode_frame.pack(fill=tk.X)
        
        # Single lineup mode
        single_rb = ttk.Radiobutton(
            mode_frame,
            text="Single Lineup - Generate 1 optimal lineup",
            variable=self.optimization_mode,
            value='single_lineup'
        )
        single_rb.pack(anchor=tk.W, pady=2)
        
        # Multi-lineup mode
        multi_rb = ttk.Radiobutton(
            mode_frame,
            text="Multi-Lineup Optimization - Generate multiple diverse lineups (Recommended)",
            variable=self.optimization_mode,
            value='multi_lineup'
        )
        multi_rb.pack(anchor=tk.W, pady=2)
        
        # Help text
        help_frame = ttk.Frame(frame)
        help_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.mode_help_label = ttk.Label(
            help_frame,
            text="Choose between single optimal lineup or multiple diverse lineups",
            font=('TkDefaultFont', 8),
            foreground='gray'
        )
        self.mode_help_label.pack(anchor=tk.W)
    
    def _create_file_section(self):
        """Create file selection section"""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Player Pool File", padding=8)
        frame.pack(fill=tk.X, padx=3, pady=3)
        
        # File selection
        file_frame = ttk.Frame(frame)
        file_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(file_frame, text="CSV File:").pack(side=tk.LEFT)
        
        self.config_vars['csvpth'] = tk.StringVar()
        self.config_vars['csvpth'].trace('w', lambda *args: self.on_change())
        
        file_entry = ttk.Entry(file_frame, textvariable=self.config_vars['csvpth'], width=50)
        file_entry.pack(side=tk.LEFT, padx=(5, 0), fill=tk.X, expand=True)
        
        browse_btn = ttk.Button(file_frame, text="Browse...", command=self._browse_csv_file)
        browse_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Column mapping
        col_frame = ttk.Frame(frame)
        col_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(col_frame, text="Column Mapping:", font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W)
        
        # Create column mapping fields
        col_mapping_frame = ttk.Frame(col_frame)
        col_mapping_frame.pack(fill=tk.X, pady=2)
        
        columns = [
            ('points_column', 'Points/Projection Column:', 'proj'),
            ('position_column', 'Position Column:', 'pos'),
            ('salary_column', 'Salary Column:', 'salary')
        ]
        
        for i, (var_name, label, default) in enumerate(columns):
            row_frame = ttk.Frame(col_mapping_frame)
            row_frame.pack(fill=tk.X, pady=1)
            
            ttk.Label(row_frame, text=label, width=20).pack(side=tk.LEFT)
            
            self.config_vars[var_name] = tk.StringVar(value=default)
            self.config_vars[var_name].trace('w', lambda *args: self.on_change())
            
            entry = ttk.Entry(row_frame, textvariable=self.config_vars[var_name], width=15)
            entry.pack(side=tk.LEFT, padx=(5, 0))
    
    def _create_optimization_settings(self):
        """Create optimization settings that change based on mode"""
        self.optimization_frame = ttk.LabelFrame(self.scrollable_frame, text="Optimization Settings", padding=8)
        self.optimization_frame.pack(fill=tk.X, padx=3, pady=3)
        
        # This will be populated based on the selected mode
        self._update_optimization_settings()
    
    def _create_single_lineup_settings(self):
        """Create settings for single lineup optimization"""
        # Clear existing widgets
        for widget in self.optimization_frame.winfo_children():
            widget.destroy()
        
        # Single lineup description
        desc_frame = ttk.Frame(self.optimization_frame)
        desc_frame.pack(fill=tk.X, pady=(0, 10))
        
        desc_label = ttk.Label(
            desc_frame,
            text="Traditional GA optimization for one high-scoring lineup",
            font=('TkDefaultFont', 9),
            foreground='blue'
        )
        desc_label.pack(anchor=tk.W)
        
        # Basic GA settings for single lineup
        settings_frame = ttk.Frame(self.optimization_frame)
        settings_frame.pack(fill=tk.X)
        
        # Single lineup optimized settings
        single_settings = [
            ('population_size', 'Population Size:', 30000, int),
            ('n_generations', 'Generations:', 20, int),
            ('mutation_rate', 'Mutation Rate:', 0.05, float),
            ('elite_divisor', 'Elite Divisor:', 5, int),
            ('stop_criteria', 'Stop Criteria:', 10, int)
        ]
        
        self._create_settings_grid(settings_frame, single_settings)
        
        # Advanced settings (collapsed by default)
        advanced_frame = ttk.LabelFrame(self.optimization_frame, text="Advanced Settings", padding=5)
        advanced_frame.pack(fill=tk.X, pady=(10, 0))
        
        advanced_settings = [
            ('crossover_method', 'Crossover Method:', 'uniform', str),
            ('select_method', 'Selection Method:', 'tournament', str),
            ('elite_method', 'Elite Method:', 'fittest', str),
            ('verbose', 'Verbose Output:', True, bool),
            ('enable_profiling', 'Enable Profiling:', True, bool)
        ]
        
        self._create_settings_grid(advanced_frame, advanced_settings)
        
        # Set target_lineups to 1 for single lineup mode
        self.config_vars['target_lineups'] = tk.StringVar(value='1')
    
    def _create_multi_lineup_settings(self):
        """Create settings for multi-lineup optimization"""
        # Clear existing widgets
        for widget in self.optimization_frame.winfo_children():
            widget.destroy()
        
        # Multi-lineup description
        desc_frame = ttk.Frame(self.optimization_frame)
        desc_frame.pack(fill=tk.X, pady=(0, 10))
        
        desc_label = ttk.Label(
            desc_frame,
            text="Optimized for generating multiple diverse lineups (10-50x faster)",
            font=('TkDefaultFont', 9),
            foreground='green'
        )
        desc_label.pack(anchor=tk.W)
        
        # Quality preset selection (prominent)
        preset_frame = ttk.Frame(self.optimization_frame)
        preset_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(preset_frame, text="Quality Preset:", width=20, font=('TkDefaultFont', 9, 'bold')).pack(side=tk.LEFT)
        
        self.config_vars['quality_preset'] = tk.StringVar(value='balanced')
        self.config_vars['quality_preset'].trace('w', lambda *args: (self._update_quality_preset(), self.on_change()))
        
        preset_combo = ttk.Combobox(
            preset_frame,
            textvariable=self.config_vars['quality_preset'],
            values=['speed', 'balanced', 'quality'],
            state='readonly',
            width=15
        )
        preset_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # Preset help text
        self.preset_help_label = ttk.Label(
            preset_frame,
            text="Balanced: Good quality + speed (Recommended)",
            font=('TkDefaultFont', 8),
            foreground='gray'
        )
        self.preset_help_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Target lineups setting
        target_frame = ttk.Frame(self.optimization_frame)
        target_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(target_frame, text="Number of Lineups:", width=20, font=('TkDefaultFont', 9, 'bold')).pack(side=tk.LEFT)
        
        self.config_vars['target_lineups'] = tk.StringVar(value='10')
        self.config_vars['target_lineups'].trace('w', lambda *args: self.on_change())
        
        target_entry = ttk.Entry(target_frame, textvariable=self.config_vars['target_lineups'], width=10)
        target_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        ttk.Label(target_frame, text="(Recommended: 5-20)", font=('TkDefaultFont', 8)).pack(side=tk.LEFT, padx=(5, 0))
        
        # Algorithm info (no selection needed - always use proven post-processing approach)
        algorithm_frame = ttk.Frame(self.optimization_frame)
        algorithm_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(algorithm_frame, text="Algorithm:", width=20, font=('TkDefaultFont', 9, 'bold')).pack(side=tk.LEFT)
        
        algorithm_info_label = ttk.Label(
            algorithm_frame,
            text="Post-processing Multi-lineup (Proven, Reliable)",
            font=('TkDefaultFont', 9),
            foreground='green'
        )
        algorithm_info_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Algorithm description
        algorithm_desc_label = ttk.Label(
            algorithm_frame,
            text="Uses traditional GA + diversity post-processing for reliable results",
            font=('TkDefaultFont', 8),
            foreground='gray'
        )
        algorithm_desc_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # GA settings optimized for multi-lineup
        ga_frame = ttk.LabelFrame(self.optimization_frame, text="Algorithm Settings", padding=5)
        ga_frame.pack(fill=tk.X, pady=(10, 0))
        
        multi_settings = [
            ('population_size', 'Population Size:', 1000, int),
            ('n_generations', 'Generations:', 100, int),
            ('mutation_rate', 'Mutation Rate:', 0.15, float),
            ('set_diversity_weight', 'Diversity Weight:', 0.3, float),
            ('lineup_pool_size', 'Lineup Pool Size:', 100000, int)
        ]
        
        self._create_settings_grid(ga_frame, multi_settings)
        
        # Advanced settings (collapsed)
        advanced_frame = ttk.LabelFrame(self.optimization_frame, text="Advanced Settings", padding=5)
        advanced_frame.pack(fill=tk.X, pady=(10, 0))
        
        advanced_settings = [
            ('tournament_size', 'Tournament Size:', 3, int),
            ('elite_divisor', 'Elite Divisor:', 5, int),
            ('stop_criteria', 'Stop Criteria:', 10, int),
            ('diversity_method', 'Diversity Method:', 'jaccard', str),
            ('verbose', 'Verbose Output:', True, bool),
            ('enable_profiling', 'Enable Profiling:', True, bool)
        ]
        
        self._create_settings_grid(advanced_frame, advanced_settings)
    
    def _create_settings_grid(self, parent: tk.Widget, settings: list):
        """Create a grid of settings widgets"""
        for i, (var_name, label, default, var_type) in enumerate(settings):
            row_frame = ttk.Frame(parent)
            row_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(row_frame, text=label, width=20).pack(side=tk.LEFT)
            
            if var_type == bool:
                if var_name not in self.config_vars:
                    self.config_vars[var_name] = tk.BooleanVar(value=default)
                widget = ttk.Checkbutton(row_frame, variable=self.config_vars[var_name])
            elif var_name in ['crossover_method', 'select_method', 'elite_method', 'diversity_method']:
                if var_name not in self.config_vars:
                    self.config_vars[var_name] = tk.StringVar(value=default)
                
                # Create combobox with appropriate values
                if var_name == 'crossover_method':
                    values = ['uniform', 'single_point', 'two_point']
                elif var_name == 'select_method':
                    values = ['tournament', 'roulette', 'rank']
                elif var_name == 'elite_method':
                    values = ['fittest', 'diverse']
                elif var_name == 'diversity_method':
                    values = ['jaccard', 'hamming']
                else:
                    values = [default]
                
                widget = ttk.Combobox(row_frame, textvariable=self.config_vars[var_name], 
                                    values=values, state='readonly', width=15)
            else:
                if var_name not in self.config_vars:
                    self.config_vars[var_name] = tk.StringVar(value=str(default))
                widget = ttk.Entry(row_frame, textvariable=self.config_vars[var_name], width=15)
            
            widget.pack(side=tk.LEFT, padx=(5, 0))
            
            # Add trace for change detection
            if var_name not in self.config_vars or not hasattr(self.config_vars[var_name], '_trace_vinfo'):
                self.config_vars[var_name].trace('w', lambda *args: self.on_change())
    
    def _create_site_settings_section(self):
        """Create site settings section"""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Site Settings", padding=8)
        frame.pack(fill=tk.X, padx=3, pady=3)
        
        # Basic settings
        basic_frame = ttk.Frame(frame)
        basic_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Get default basic settings from config manager
        default_site_settings = self.default_config.get('site_settings', {})
        basic_settings = [
            ('salary_cap', 'Salary Cap:', default_site_settings.get('salary_cap', 50000), int),
            ('lineup_size', 'Lineup Size:', default_site_settings.get('lineup_size', 9), int)
        ]
        
        self._create_settings_grid(basic_frame, basic_settings)
        
        # Position settings
        pos_frame = ttk.LabelFrame(frame, text="Position Settings", padding=5)
        pos_frame.pack(fill=tk.X, pady=5)
        
        # Position map
        posmap_frame = ttk.Frame(pos_frame)
        posmap_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(posmap_frame, text="Position Map (positions per lineup):", font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W)
        
        posmap_grid = ttk.Frame(posmap_frame)
        posmap_grid.pack(fill=tk.X, pady=2)
        
        # Get default position map from config manager
        default_posmap = self.default_config.get('site_settings', {}).get('posmap', {
            'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1, 'FLEX': 1
        })
        
        for i, (pos, count) in enumerate(default_posmap.items()):
            row = i // 3
            col = i % 3
            
            pos_frame_item = ttk.Frame(posmap_grid)
            pos_frame_item.grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
            
            ttk.Label(pos_frame_item, text=f"{pos}:", width=6).pack(side=tk.LEFT)
            
            self.posmap_vars[pos] = tk.IntVar(value=count)
            self.posmap_vars[pos].trace('w', lambda *args: self.on_change())
            
            entry = ttk.Entry(pos_frame_item, textvariable=self.posmap_vars[pos], width=5)
            entry.pack(side=tk.LEFT)
        
        # Flex positions
        flex_frame = ttk.Frame(pos_frame)
        flex_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(flex_frame, text="Flex Positions:", font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W)
        
        # Get default flex positions from config manager
        default_flex_positions = self.default_config.get('site_settings', {}).get('flex_positions', ('RB', 'WR', 'TE'))
        flex_positions_str = ','.join(default_flex_positions)
        self.config_vars['flex_positions'] = tk.StringVar(value=flex_positions_str)
        self.config_vars['flex_positions'].trace('w', lambda *args: self.on_change())
        
        flex_entry = ttk.Entry(flex_frame, textvariable=self.config_vars['flex_positions'], width=30)
        flex_entry.pack(anchor=tk.W, pady=2)
        
        ttk.Label(flex_frame, text="(comma-separated, e.g., RB,WR,TE)", font=('TkDefaultFont', 8)).pack(anchor=tk.W)
        
        # Position filters
        posfilter_frame = ttk.Frame(pos_frame)
        posfilter_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(posfilter_frame, text="Position Filters (minimum points):", font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W)
        
        posfilter_grid = ttk.Frame(posfilter_frame)
        posfilter_grid.pack(fill=tk.X, pady=2)
        
        # Get default position filters from config manager
        default_posfilter = self.default_config.get('site_settings', {}).get('posfilter', {
            'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'DST': 0, 'FLEX': 0
        })
        
        for i, (pos, thresh) in enumerate(default_posfilter.items()):
            row = i // 3
            col = i % 3
            
            pos_filter_item = ttk.Frame(posfilter_grid)
            pos_filter_item.grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
            
            ttk.Label(pos_filter_item, text=f"{pos}:", width=6).pack(side=tk.LEFT)
            
            self.posfilter_vars[pos] = tk.StringVar(value=str(thresh))
            self.posfilter_vars[pos].trace('w', lambda *args: (self.on_change(), self._update_position_filters()))
            
            entry = ttk.Entry(pos_filter_item, textvariable=self.posfilter_vars[pos], width=8)
            entry.pack(side=tk.LEFT)
    
    def _on_mode_change(self, *args):
        """Handle optimization mode change"""
        self._update_optimization_settings()
        self._update_mode_help()
        self.on_change()
    
    def _update_optimization_settings(self):
        """Update optimization settings based on selected mode"""
        mode = self.optimization_mode.get()
        
        if mode == 'single_lineup':
            self._create_single_lineup_settings()
        else:  # multi_lineup
            self._create_multi_lineup_settings()
    
    def _update_mode_help(self):
        """Update mode help text"""
        mode = self.optimization_mode.get()
        
        if mode == 'single_lineup':
            self.mode_help_label.config(
                text="Single Lineup Optimization: Traditional GA for one high-scoring lineup"
            )
        else:
            self.mode_help_label.config(
                text="Multi-Lineup Optimization: Proven algorithm for multiple diverse lineups"
            )
    
    def _update_optimizer_help(self):
        """Update optimizer help text"""
        if hasattr(self, 'optimizer_help_label'):
            optimizer_type = self.config_vars.get('optimizer_type', tk.StringVar(value='set_based')).get()
            
            if optimizer_type == 'set_based':
                self.optimizer_help_label.config(
                    text="Set-based: Ultra-fast algorithm (Recommended)"
                )
            else:
                self.optimizer_help_label.config(
                    text="Post-processing: Traditional approach (slower)"
                )
    
    def _update_quality_preset(self):
        """Update settings based on quality preset selection"""
        if not hasattr(self, 'preset_help_label'):
            return
            
        preset = self.config_vars.get('quality_preset', tk.StringVar(value='balanced')).get()
        
        # Define preset configurations for post-processing optimizer only
        presets = {
            'speed': {
                'population_size': 500,
                'n_generations': 50,
                'mutation_rate': 0.20,
                'diversity_weight': 0.15,
                'min_overlap_threshold': 0.5,
                'help_text': "Speed: Fast results with good diversity",
                'description': "Optimized for speed with good diversity"
            },
            'balanced': {
                'population_size': 1000,
                'n_generations': 100,
                'mutation_rate': 0.15,
                'diversity_weight': 0.20,
                'min_overlap_threshold': 0.4,
                'help_text': "Balanced: Good quality + diversity (Recommended)",
                'description': "Recommended balance of quality and diversity"
            },
            'quality': {
                'population_size': 2000,
                'n_generations': 200,
                'mutation_rate': 0.10,
                'diversity_weight': 0.25,
                'min_overlap_threshold': 0.3,
                'help_text': "Quality: Maximum quality + diversity",
                'description': "Best quality with strong diversity enforcement"
            }
        }
        
        if preset in presets:
            preset_config = presets[preset]
            
            # Update settings
            for setting, value in preset_config.items():
                if setting in ['help_text', 'description']:
                    continue
                if setting in self.config_vars:
                    self.config_vars[setting].set(str(value))
            
            # Update help text
            self.preset_help_label.config(text=preset_config['help_text'])
    
    def _browse_csv_file(self):
        """Browse for CSV file"""
        filename = filedialog.askopenfilename(
            title="Select Player Pool CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.config_vars['csvpth'].set(filename)
            self._auto_detect_columns(filename)
            self._load_player_pool()
    
    def _auto_detect_columns(self, filename: str):
        """Auto-detect column names from CSV file"""
        try:
            # Read just the header
            df = pd.read_csv(filename, nrows=0)
            columns = df.columns.tolist()
            
            # Try to auto-detect common column names
            column_mapping = {
                'points_column': ['proj', 'projection', 'points', 'fpts', 'fantasy_points'],
                'position_column': ['pos', 'position', 'positions'],
                'salary_column': ['salary', 'cost', 'price']
            }
            
            for config_key, possible_names in column_mapping.items():
                for col in columns:
                    if col.lower() in [name.lower() for name in possible_names]:
                        self.config_vars[config_key].set(col)
                        break
                        
        except Exception as e:
            # If auto-detection fails, just continue with defaults
            pass
    
    def _load_player_pool(self):
        """Load player pool into the preview panel"""
        csv_path = self.config_vars.get('csvpth', tk.StringVar()).get()
        if not csv_path or not Path(csv_path).exists():
            return
        
        # Get column mapping
        column_mapping = {
            'position_column': self.config_vars.get('position_column', tk.StringVar(value='pos')).get(),
            'salary_column': self.config_vars.get('salary_column', tk.StringVar(value='salary')).get(),
            'points_column': self.config_vars.get('points_column', tk.StringVar(value='proj')).get()
        }
        
        # Load into player pool panel
        self.player_pool_panel.load_player_pool(csv_path, column_mapping)
        
        # Apply current position filters
        self._update_position_filters()
    
    def _on_player_pool_change(self):
        """Handle changes in the player pool panel"""
        # Notify parent of change
        self.on_change()
    
    def _update_position_filters(self):
        """Update position filters in the player pool panel"""
        # Get current position filter values
        position_filters = {}
        for pos, var in self.posfilter_vars.items():
            try:
                value = float(var.get())
                position_filters[pos] = value
            except ValueError:
                position_filters[pos] = 0.0
        
        # Apply to player pool panel
        self.player_pool_panel.set_position_filters(position_filters)
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary"""
        config = {
            'ga_settings': {},
            'site_settings': {}
        }
        
        # Basic GA settings
        ga_settings = [
            'csvpth', 'points_column', 'position_column', 'salary_column',
            'population_size', 'n_generations', 'mutation_rate', 'elite_divisor',
            'stop_criteria', 'crossover_method', 'select_method', 'elite_method',
            'verbose', 'enable_profiling', 'target_lineups'
        ]
        
        # Add mode-specific settings
        mode = self.optimization_mode.get()
        if mode == 'multi_lineup':
            # Add multilineup-specific settings for post-processing optimizer
            ga_settings.extend([
                'diversity_method', 'set_diversity_weight', 'tournament_size', 
                'lineup_pool_size', 'diversity_weight', 'min_overlap_threshold'
            ])
        
        for setting in ga_settings:
            if setting in self.config_vars:
                value = self.config_vars[setting].get()
                
                # Convert types
                if setting in ['population_size', 'n_generations', 'elite_divisor', 'stop_criteria', 'target_lineups', 'tournament_size', 'lineup_pool_size']:
                    try:
                        value = int(value)
                    except ValueError:
                        value = 0
                elif setting in ['mutation_rate', 'diversity_weight', 'min_overlap_threshold', 'set_diversity_weight']:
                    try:
                        value = float(value)
                    except ValueError:
                        value = 0.0
                elif setting == 'csvpth' and value:
                    value = Path(value)
                
                config['ga_settings'][setting] = value
        
        # Site settings
        config['site_settings']['salary_cap'] = int(self.config_vars.get('salary_cap', tk.StringVar(value='50000')).get() or 50000)
        config['site_settings']['lineup_size'] = int(self.config_vars.get('lineup_size', tk.StringVar(value='9')).get() or 9)
        
        # Position map
        posmap = {}
        for pos, var in self.posmap_vars.items():
            posmap[pos] = var.get()
        config['site_settings']['posmap'] = posmap
        
        # Flex positions
        flex_str = self.config_vars.get('flex_positions', tk.StringVar(value='RB,WR,TE')).get()
        flex_positions = tuple(pos.strip() for pos in flex_str.split(',') if pos.strip())
        config['site_settings']['flex_positions'] = flex_positions
        
        # Position filters
        posfilter = {}
        for pos, var in self.posfilter_vars.items():
            try:
                posfilter[pos] = float(var.get())
            except ValueError:
                posfilter[pos] = 0.0
        config['site_settings']['posfilter'] = posfilter
        
        return config
    
    def load_config(self, config: Dict[str, Any]):
        """Load configuration into the UI"""
        # Load GA settings
        ga_settings = config.get('ga_settings', {})
        
        # Determine mode based on target_lineups
        target_lineups = ga_settings.get('target_lineups', 10)
        if target_lineups == 1:
            self.optimization_mode.set('single_lineup')
        else:
            self.optimization_mode.set('multi_lineup')
        
        # Update UI based on mode
        self._update_optimization_settings()
        
        # Load settings
        for setting, value in ga_settings.items():
            if setting in self.config_vars:
                if isinstance(value, Path):
                    self.config_vars[setting].set(str(value))
                else:
                    self.config_vars[setting].set(value)
        
        # Load site settings
        site_settings = config.get('site_settings', {})
        
        if 'salary_cap' in site_settings:
            self.config_vars['salary_cap'].set(str(site_settings['salary_cap']))
        
        if 'lineup_size' in site_settings:
            self.config_vars['lineup_size'].set(str(site_settings['lineup_size']))
        
        # Load position map
        posmap = site_settings.get('posmap', {})
        for pos, count in posmap.items():
            if pos in self.posmap_vars:
                self.posmap_vars[pos].set(count)
        
        # Load flex positions
        flex_positions = site_settings.get('flex_positions', ('RB', 'WR', 'TE'))
        flex_str = ','.join(flex_positions)
        self.config_vars['flex_positions'].set(flex_str)
        
        # Load position filters
        posfilter = site_settings.get('posfilter', {})
        for pos, thresh in posfilter.items():
            if pos in self.posfilter_vars:
                self.posfilter_vars[pos].set(str(thresh))
    
    def set_player_pool_file(self, filename: str):
        """Set the player pool file"""
        self.config_vars['csvpth'].set(filename)
        self._auto_detect_columns(filename)
        self._load_player_pool()
    
    def get_modified_player_data(self) -> Optional[pd.DataFrame]:
        """Get the modified player pool data"""
        return self.player_pool_panel.get_modified_data()
    
    def has_player_pool_modifications(self) -> bool:
        """Check if there are player pool modifications"""
        return self.player_pool_panel.has_modifications()
    
    def get_optimizer_name(self) -> str:
        """Get the optimizer name based on current settings"""
        mode = self.optimization_mode.get()
        
        if mode == 'single_lineup':
            return 'optimize_default'
        else:
            # Always use post-processing multilineup optimizer (proven, reliable)
            return 'optimize_multilineup'
