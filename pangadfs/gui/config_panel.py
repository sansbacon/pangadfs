# pangadfs/gui/config_panel.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
from typing import Dict, Any, Callable, Optional
import pandas as pd
from .widgets import PlayerPoolPanel


class ConfigPanel:
    """Configuration panel for GA and site settings"""
    
    def __init__(self, parent: tk.Widget, on_change_callback: Callable[[], None]):
        self.parent = parent
        self.on_change = on_change_callback
        self.config_vars = {}
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the configuration UI"""
        # Create main horizontal paned window
        main_paned = ttk.PanedWindow(self.parent, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left side - Configuration settings (40% width)
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=2)
        
        # Right side - Player pool preview (60% width)
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=3)
        
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
        self._create_file_section()
        self._create_ga_settings_section()
        self._create_multilineup_section()
        self._create_site_settings_section()
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    def _setup_player_pool_side(self, parent):
        """Set up the player pool side (right)"""
        # Create player pool panel
        self.player_pool_panel = PlayerPoolPanel(parent, self._on_player_pool_change)
    
    def _create_file_section(self):
        """Create file selection section"""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Player Pool File", padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)
        
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
    
    def _create_ga_settings_section(self):
        """Create GA settings section"""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Genetic Algorithm Settings", padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create two columns for GA settings
        left_frame = ttk.Frame(frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_frame = ttk.Frame(frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Left column settings
        left_settings = [
            ('population_size', 'Population Size:', 30000, int),
            ('n_generations', 'Generations:', 20, int),
            ('mutation_rate', 'Mutation Rate:', 0.05, float),
            ('elite_divisor', 'Elite Divisor:', 5, int),
            ('stop_criteria', 'Stop Criteria:', 10, int)
        ]
        
        # Right column settings
        right_settings = [
            ('crossover_method', 'Crossover Method:', 'uniform', str),
            ('select_method', 'Selection Method:', 'tournament', str),
            ('elite_method', 'Elite Method:', 'fittest', str),
            ('verbose', 'Verbose Output:', True, bool),
            ('enable_profiling', 'Enable Profiling:', True, bool)
        ]
        
        self._create_settings_column(left_frame, left_settings)
        self._create_settings_column(right_frame, right_settings)
    
    def _create_multilineup_section(self):
        """Create multilineup settings section"""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Multilineup Settings", padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Enable multilineup checkbox
        self.config_vars['enable_multilineup'] = tk.BooleanVar(value=False)
        self.config_vars['enable_multilineup'].trace('w', self._on_multilineup_toggle)
        
        enable_cb = ttk.Checkbutton(
            frame, 
            text="Enable Multilineup Optimization", 
            variable=self.config_vars['enable_multilineup']
        )
        enable_cb.pack(anchor=tk.W, pady=(0, 10))
        
        # Multilineup settings frame (initially disabled)
        self.multilineup_frame = ttk.Frame(frame)
        self.multilineup_frame.pack(fill=tk.X)
        
        multilineup_settings = [
            ('target_lineups', 'Target Lineups:', 150, int),
            ('diversity_weight', 'Diversity Weight (0-1):', 0.3, float),
            ('min_overlap_threshold', 'Min Overlap Threshold (0-1):', 0.3, float),
            ('diversity_method', 'Diversity Method:', 'jaccard', str)
        ]
        
        self._create_settings_column(self.multilineup_frame, multilineup_settings)
        
        # Initially disable multilineup settings
        self._set_multilineup_enabled(False)
    
    def _create_site_settings_section(self):
        """Create site settings section"""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Site Settings", padding=10)
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Basic settings
        basic_frame = ttk.Frame(frame)
        basic_frame.pack(fill=tk.X, pady=(0, 10))
        
        basic_settings = [
            ('salary_cap', 'Salary Cap:', 50000, int),
            ('lineup_size', 'Lineup Size:', 9, int)
        ]
        
        self._create_settings_column(basic_frame, basic_settings)
        
        # Position settings
        pos_frame = ttk.LabelFrame(frame, text="Position Settings", padding=5)
        pos_frame.pack(fill=tk.X, pady=5)
        
        # Position map
        posmap_frame = ttk.Frame(pos_frame)
        posmap_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(posmap_frame, text="Position Map (positions per lineup):", font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W)
        
        posmap_grid = ttk.Frame(posmap_frame)
        posmap_grid.pack(fill=tk.X, pady=2)
        
        # Default position map
        default_posmap = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1, 'FLEX': 1}
        
        self.posmap_vars = {}
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
        
        self.config_vars['flex_positions'] = tk.StringVar(value='RB,WR,TE')
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
        
        # Default position filters
        default_posfilter = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'DST': 0, 'FLEX': 0}
        
        self.posfilter_vars = {}
        for i, (pos, thresh) in enumerate(default_posfilter.items()):
            row = i // 3
            col = i % 3
            
            pos_filter_item = ttk.Frame(posfilter_grid)
            pos_filter_item.grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
            
            ttk.Label(pos_filter_item, text=f"{pos}:", width=6).pack(side=tk.LEFT)
            
            self.posfilter_vars[pos] = tk.StringVar(value=str(thresh))
            self.posfilter_vars[pos].trace('w', lambda *args: self.on_change())
            
            entry = ttk.Entry(pos_filter_item, textvariable=self.posfilter_vars[pos], width=8)
            entry.pack(side=tk.LEFT)
    
    def _create_settings_column(self, parent: tk.Widget, settings: list):
        """Create a column of settings widgets"""
        for var_name, label, default, var_type in settings:
            row_frame = ttk.Frame(parent)
            row_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(row_frame, text=label, width=20).pack(side=tk.LEFT)
            
            if var_type == bool:
                self.config_vars[var_name] = tk.BooleanVar(value=default)
                widget = ttk.Checkbutton(row_frame, variable=self.config_vars[var_name])
            elif var_name in ['crossover_method', 'select_method', 'elite_method', 'diversity_method']:
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
                self.config_vars[var_name] = tk.StringVar(value=str(default))
                widget = ttk.Entry(row_frame, textvariable=self.config_vars[var_name], width=15)
            
            widget.pack(side=tk.LEFT, padx=(5, 0))
            
            # Add trace for change detection
            self.config_vars[var_name].trace('w', lambda *args: self.on_change())
    
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
    
    def _on_multilineup_toggle(self, *args):
        """Handle multilineup enable/disable"""
        enabled = self.config_vars['enable_multilineup'].get()
        self._set_multilineup_enabled(enabled)
        
        # Set target_lineups based on multilineup setting
        if enabled:
            if 'target_lineups' not in self.config_vars:
                self.config_vars['target_lineups'] = tk.StringVar(value='150')
            else:
                current_val = self.config_vars['target_lineups'].get()
                if current_val == '1':
                    self.config_vars['target_lineups'].set('150')
        else:
            if 'target_lineups' in self.config_vars:
                self.config_vars['target_lineups'].set('1')
        
        self.on_change()
    
    def _set_multilineup_enabled(self, enabled: bool):
        """Enable/disable multilineup settings widgets"""
        state = 'normal' if enabled else 'disabled'
        
        for widget in self.multilineup_frame.winfo_children():
            self._set_widget_state_recursive(widget, state)
    
    def _set_widget_state_recursive(self, widget, state):
        """Recursively set widget state"""
        try:
            widget.configure(state=state)
        except tk.TclError:
            pass  # Some widgets don't support state
        
        for child in widget.winfo_children():
            self._set_widget_state_recursive(child, state)
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary"""
        config = {
            'ga_settings': {},
            'site_settings': {}
        }
        
        # GA settings
        ga_settings = [
            'csvpth', 'points_column', 'position_column', 'salary_column',
            'population_size', 'n_generations', 'mutation_rate', 'elite_divisor',
            'stop_criteria', 'crossover_method', 'select_method', 'elite_method',
            'verbose', 'enable_profiling'
        ]
        
        # Add multilineup settings if enabled
        if self.config_vars.get('enable_multilineup', tk.BooleanVar()).get():
            ga_settings.extend(['target_lineups', 'diversity_weight', 'min_overlap_threshold', 'diversity_method'])
        else:
            # Ensure target_lineups is 1 for single lineup mode
            config['ga_settings']['target_lineups'] = 1
        
        for setting in ga_settings:
            if setting in self.config_vars:
                value = self.config_vars[setting].get()
                
                # Convert types
                if setting in ['population_size', 'n_generations', 'elite_divisor', 'stop_criteria', 'target_lineups']:
                    try:
                        value = int(value)
                    except ValueError:
                        value = 0
                elif setting in ['mutation_rate', 'diversity_weight', 'min_overlap_threshold']:
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
        
        # Set multilineup checkbox based on target_lineups
        target_lineups = ga_settings.get('target_lineups', 1)
        self.config_vars['enable_multilineup'].set(target_lineups > 1)
    
    def set_player_pool_file(self, filename: str):
        """Set the player pool file"""
        self.config_vars['csvpth'].set(filename)
        self._auto_detect_columns(filename)
        self._load_player_pool()
    
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
    
    def _on_player_pool_change(self):
        """Handle changes in the player pool panel"""
        # Notify parent of change
        self.on_change()
    
    def get_modified_player_data(self) -> Optional[pd.DataFrame]:
        """Get the modified player pool data"""
        return self.player_pool_panel.get_modified_data()
    
    def has_player_pool_modifications(self) -> bool:
        """Check if there are player pool modifications"""
        return self.player_pool_panel.has_modifications()
