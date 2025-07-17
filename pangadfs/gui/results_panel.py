# pangadfs/gui/results_panel.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class ResultsPanel:
    """Panel for displaying optimization results"""
    
    def __init__(self, parent: tk.Widget):
        self.parent = parent
        self.results = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the results UI"""
        # Create notebook for different result views
        self.notebook = ttk.Notebook(self.parent)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Summary tab
        self._create_summary_tab()
        
        # Lineups tab
        self._create_lineups_tab()
        
        # Individual lineup tab
        self._create_individual_tab()
        
        # Diversity analysis tab
        self._create_diversity_tab()
        
        # Profiling tab
        self._create_profiling_tab()
    
    def _create_summary_tab(self):
        """Create summary results tab"""
        summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(summary_frame, text="Summary")
        
        # Main content frame
        content_frame = ttk.Frame(summary_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Key metrics section
        metrics_frame = ttk.LabelFrame(content_frame, text="Key Metrics", padding=10)
        metrics_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create metrics grid
        self.metrics_vars = {}
        metrics = [
            ('num_lineups', 'Number of Lineups:', '0'),
            ('best_score', 'Best Score:', '0.00'),
            ('avg_score', 'Average Score:', '0.00'),
            ('score_range', 'Score Range:', '0.00'),
            ('avg_overlap', 'Average Overlap:', '0.000'),
            ('min_overlap', 'Minimum Overlap:', '0.000')
        ]
        
        for i, (key, label, default) in enumerate(metrics):
            row = i // 2
            col = i % 2
            
            metric_frame = ttk.Frame(metrics_frame)
            metric_frame.grid(row=row, column=col, sticky=tk.W, padx=10, pady=5)
            
            ttk.Label(metric_frame, text=label, font=('TkDefaultFont', 9, 'bold')).pack(side=tk.LEFT)
            
            self.metrics_vars[key] = tk.StringVar(value=default)
            ttk.Label(metric_frame, textvariable=self.metrics_vars[key], width=15).pack(side=tk.LEFT, padx=(5, 0))
        
        # Best lineup section
        best_frame = ttk.LabelFrame(content_frame, text="Best Lineup", padding=10)
        best_frame.pack(fill=tk.BOTH, expand=True)
        
        # Best lineup table
        self.best_lineup_tree = ttk.Treeview(best_frame, height=10)
        self.best_lineup_tree.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar for best lineup
        best_scrollbar = ttk.Scrollbar(best_frame, orient=tk.VERTICAL, command=self.best_lineup_tree.yview)
        best_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.best_lineup_tree.configure(yscrollcommand=best_scrollbar.set)
    
    def _create_lineups_tab(self):
        """Create lineups overview tab"""
        lineups_frame = ttk.Frame(self.notebook)
        self.notebook.add(lineups_frame, text="All Lineups")
        
        # Control frame
        control_frame = ttk.Frame(lineups_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Sort controls
        ttk.Label(control_frame, text="Sort by:").pack(side=tk.LEFT)
        
        self.sort_var = tk.StringVar(value="Score (High to Low)")
        sort_combo = ttk.Combobox(
            control_frame, 
            textvariable=self.sort_var,
            values=["Score (High to Low)", "Score (Low to High)", "Lineup Number"],
            state="readonly",
            width=20
        )
        sort_combo.pack(side=tk.LEFT, padx=(5, 10))
        sort_combo.bind('<<ComboboxSelected>>', self._sort_lineups)
        
        # Filter controls
        ttk.Label(control_frame, text="Show top:").pack(side=tk.LEFT)
        
        self.filter_var = tk.StringVar(value="All")
        filter_combo = ttk.Combobox(
            control_frame,
            textvariable=self.filter_var,
            values=["All", "10", "25", "50", "100"],
            state="readonly",
            width=10
        )
        filter_combo.pack(side=tk.LEFT, padx=(5, 0))
        filter_combo.bind('<<ComboboxSelected>>', self._filter_lineups)
        
        # Lineups table
        table_frame = ttk.Frame(lineups_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.lineups_tree = ttk.Treeview(table_frame)
        self.lineups_tree.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.lineups_tree.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.lineups_tree.configure(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.lineups_tree.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.lineups_tree.configure(xscrollcommand=h_scrollbar.set)
        
        # Bind double-click to show individual lineup
        self.lineups_tree.bind('<Double-1>', self._on_lineup_double_click)
    
    def _create_individual_tab(self):
        """Create individual lineup view tab"""
        individual_frame = ttk.Frame(self.notebook)
        self.notebook.add(individual_frame, text="Lineup Details")
        
        # Selection frame
        selection_frame = ttk.Frame(individual_frame)
        selection_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(selection_frame, text="Select Lineup:").pack(side=tk.LEFT)
        
        self.lineup_selection_var = tk.StringVar()
        self.lineup_selection_combo = ttk.Combobox(
            selection_frame,
            textvariable=self.lineup_selection_var,
            state="readonly",
            width=20
        )
        self.lineup_selection_combo.pack(side=tk.LEFT, padx=(5, 0))
        self.lineup_selection_combo.bind('<<ComboboxSelected>>', self._show_selected_lineup)
        
        # Individual lineup details
        details_frame = ttk.LabelFrame(individual_frame, text="Lineup Details", padding=10)
        details_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Lineup info
        info_frame = ttk.Frame(details_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.lineup_info_vars = {}
        info_items = [
            ('lineup_score', 'Score:', '0.00'),
            ('lineup_salary', 'Total Salary:', '$0'),
            ('lineup_remaining', 'Remaining Salary:', '$0'),
            ('lineup_rank', 'Rank:', '0')
        ]
        
        for i, (key, label, default) in enumerate(info_items):
            item_frame = ttk.Frame(info_frame)
            item_frame.grid(row=i//2, column=i%2, sticky=tk.W, padx=10, pady=2)
            
            ttk.Label(item_frame, text=label, font=('TkDefaultFont', 9, 'bold')).pack(side=tk.LEFT)
            
            self.lineup_info_vars[key] = tk.StringVar(value=default)
            ttk.Label(item_frame, textvariable=self.lineup_info_vars[key]).pack(side=tk.LEFT, padx=(5, 0))
        
        # Individual lineup table
        self.individual_tree = ttk.Treeview(details_frame)
        self.individual_tree.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar for individual lineup
        individual_scrollbar = ttk.Scrollbar(details_frame, orient=tk.VERTICAL, command=self.individual_tree.yview)
        individual_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.individual_tree.configure(yscrollcommand=individual_scrollbar.set)
    
    def _create_diversity_tab(self):
        """Create diversity analysis tab"""
        diversity_frame = ttk.Frame(self.notebook)
        self.notebook.add(diversity_frame, text="Diversity Analysis")
        
        # Diversity metrics
        metrics_frame = ttk.LabelFrame(diversity_frame, text="Diversity Metrics", padding=10)
        metrics_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.diversity_vars = {}
        diversity_metrics = [
            ('diversity_method', 'Method:', 'N/A'),
            ('avg_diversity', 'Average Diversity:', '0.000'),
            ('min_diversity', 'Minimum Diversity:', '0.000'),
            ('max_diversity', 'Maximum Diversity:', '0.000')
        ]
        
        for i, (key, label, default) in enumerate(diversity_metrics):
            metric_frame = ttk.Frame(metrics_frame)
            metric_frame.grid(row=i//2, column=i%2, sticky=tk.W, padx=10, pady=5)
            
            ttk.Label(metric_frame, text=label, font=('TkDefaultFont', 9, 'bold')).pack(side=tk.LEFT)
            
            self.diversity_vars[key] = tk.StringVar(value=default)
            ttk.Label(metric_frame, textvariable=self.diversity_vars[key]).pack(side=tk.LEFT, padx=(5, 0))
        
        # Diversity matrix (simplified view)
        matrix_frame = ttk.LabelFrame(diversity_frame, text="Diversity Matrix (Sample)", padding=10)
        matrix_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.diversity_text = tk.Text(matrix_frame, height=15, font=('Consolas', 9))
        self.diversity_text.pack(fill=tk.BOTH, expand=True)
        
        diversity_scrollbar = ttk.Scrollbar(matrix_frame, orient=tk.VERTICAL, command=self.diversity_text.yview)
        diversity_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.diversity_text.configure(yscrollcommand=diversity_scrollbar.set)
    
    def _create_profiling_tab(self):
        """Create profiling results tab"""
        profiling_frame = ttk.Frame(self.notebook)
        self.notebook.add(profiling_frame, text="Performance")
        
        # Profiling text area
        self.profiling_text = tk.Text(profiling_frame, font=('Consolas', 9))
        self.profiling_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        profiling_scrollbar = ttk.Scrollbar(profiling_frame, orient=tk.VERTICAL, command=self.profiling_text.yview)
        profiling_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.profiling_text.configure(yscrollcommand=profiling_scrollbar.set)
    
    def display_results(self, results: Dict[str, Any]):
        """Display optimization results"""
        self.results = results
        
        # Determine if this is multilineup mode
        lineups = results.get('lineups', [])
        scores = results.get('scores', [])
        
        if not lineups:
            # Single lineup mode
            lineups = [results.get('best_lineup')]
            scores = [results.get('best_score', 0)]
        
        is_multilineup = len(lineups) > 1
        
        # Enable/disable tabs based on lineup count
        self._configure_tabs_for_mode(is_multilineup)
        
        # Update summary
        self._update_summary()
        
        # Update lineups table
        self._update_lineups_table()
        
        # Update individual lineup selection
        self._update_individual_selection()
        
        # Update diversity analysis
        self._update_diversity_analysis()
        
        # Update profiling
        self._update_profiling()
    
    def _configure_tabs_for_mode(self, is_multilineup: bool):
        """Configure tabs based on single vs multilineup mode"""
        if is_multilineup:
            # Enable all tabs for multilineup mode
            self.notebook.tab(1, state="normal")  # All Lineups tab
            self.notebook.tab(2, state="normal")  # Lineup Details tab
            self.notebook.tab(3, state="normal")  # Diversity Analysis tab
            
            # Update tab text to reflect multilineup context
            self.notebook.tab(1, text="All Lineups")
            self.notebook.tab(2, text="Lineup Details")
            self.notebook.tab(3, text="Diversity Analysis")
        else:
            # Disable multilineup-specific tabs for single lineup mode
            self.notebook.tab(1, state="disabled")  # All Lineups tab (not needed for single lineup)
            self.notebook.tab(2, state="disabled")  # Lineup Details tab (redundant with Summary for single lineup)
            self.notebook.tab(3, state="disabled")  # Diversity Analysis tab (not meaningful for single lineup)
            
            # Ensure we're on an enabled tab
            current_tab = self.notebook.index(self.notebook.select())
            if current_tab in [1, 2, 3]:  # If on a disabled tab
                self.notebook.select(0)  # Switch to Summary tab
    
    def _update_summary(self):
        """Update summary metrics"""
        if not self.results:
            return
        
        # Get lineups and scores
        lineups = self.results.get('lineups', [])
        scores = self.results.get('scores', [])
        
        if not lineups:
            # Single lineup mode
            lineups = [self.results.get('best_lineup')]
            scores = [self.results.get('best_score', 0)]
        
        # Update metrics
        self.metrics_vars['num_lineups'].set(str(len(lineups)))
        self.metrics_vars['best_score'].set(f"{max(scores):.2f}" if scores else "0.00")
        self.metrics_vars['avg_score'].set(f"{np.mean(scores):.2f}" if scores else "0.00")
        self.metrics_vars['score_range'].set(f"{max(scores) - min(scores):.2f}" if len(scores) > 1 else "0.00")
        
        # Diversity metrics
        diversity_metrics = self.results.get('diversity_metrics', {})
        self.metrics_vars['avg_overlap'].set(f"{diversity_metrics.get('avg_overlap', 0):.3f}")
        self.metrics_vars['min_overlap'].set(f"{diversity_metrics.get('min_overlap', 0):.3f}")
        
        # Update best lineup table
        self._update_best_lineup_table()
    
    def _update_best_lineup_table(self):
        """Update best lineup table"""
        # Clear existing items
        for item in self.best_lineup_tree.get_children():
            self.best_lineup_tree.delete(item)
        
        best_lineup = self.results.get('best_lineup')
        if best_lineup is None:
            return
        
        # Set up columns
        columns = ['Position', 'Player', 'Team', 'Salary', 'Projection']
        self.best_lineup_tree['columns'] = columns
        self.best_lineup_tree['show'] = 'headings'
        
        for col in columns:
            self.best_lineup_tree.heading(col, text=col)
            self.best_lineup_tree.column(col, width=100)
        
        # Add data
        for _, player in best_lineup.iterrows():
            values = [
                player.get('pos', ''),
                player.get('player', ''),
                player.get('team', ''),
                f"${player.get('salary', 0):,}",
                f"{player.get('proj', 0):.2f}"
            ]
            self.best_lineup_tree.insert('', tk.END, values=values)
    
    def _update_lineups_table(self):
        """Update lineups overview table"""
        # Clear existing items
        for item in self.lineups_tree.get_children():
            self.lineups_tree.delete(item)
        
        lineups = self.results.get('lineups', [])
        scores = self.results.get('scores', [])
        
        if not lineups:
            return
        
        # Set up columns
        columns = ['Rank', 'Score', 'Total Salary', 'QB', 'RB1', 'RB2', 'WR1', 'WR2', 'WR3', 'TE', 'FLEX', 'DST']
        self.lineups_tree['columns'] = columns
        self.lineups_tree['show'] = 'headings'
        
        for col in columns:
            self.lineups_tree.heading(col, text=col)
            if col in ['Rank', 'Score']:
                self.lineups_tree.column(col, width=60)
            elif col == 'Total Salary':
                self.lineups_tree.column(col, width=80)
            else:
                self.lineups_tree.column(col, width=100)
        
        # Add data
        for i, (lineup, score) in enumerate(zip(lineups, scores)):
            # Calculate total salary
            total_salary = lineup.get('salary', pd.Series()).sum()
            
            # Get players by position (simplified)
            players_by_pos = {}
            for _, player in lineup.iterrows():
                pos = player.get('pos', '')
                if pos not in players_by_pos:
                    players_by_pos[pos] = []
                players_by_pos[pos].append(player.get('player', ''))
            
            # Build row values
            values = [
                str(i + 1),
                f"{score:.2f}",
                f"${total_salary:,}"
            ]
            
            # Add position players (simplified mapping)
            for pos in ['QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'FLEX', 'DST']:
                if pos in players_by_pos and players_by_pos[pos]:
                    values.append(players_by_pos[pos].pop(0))
                else:
                    values.append('')
            
            self.lineups_tree.insert('', tk.END, values=values)
    
    def _update_individual_selection(self):
        """Update individual lineup selection dropdown"""
        lineups = self.results.get('lineups', [])
        scores = self.results.get('scores', [])
        
        if not lineups:
            return
        
        # Create selection options
        options = []
        for i, score in enumerate(scores):
            options.append(f"Lineup {i+1} (Score: {score:.2f})")
        
        self.lineup_selection_combo['values'] = options
        if options:
            self.lineup_selection_combo.set(options[0])
            self._show_selected_lineup()
    
    def _update_diversity_analysis(self):
        """Update diversity analysis"""
        diversity_metrics = self.results.get('diversity_metrics', {})
        
        # Update diversity metrics
        self.diversity_vars['diversity_method'].set(
            self.results.get('ga_settings', {}).get('diversity_method', 'N/A')
        )
        self.diversity_vars['avg_diversity'].set(f"{diversity_metrics.get('avg_overlap', 0):.3f}")
        self.diversity_vars['min_diversity'].set(f"{diversity_metrics.get('min_overlap', 0):.3f}")
        
        # Calculate max diversity if matrix available
        diversity_matrix = diversity_metrics.get('diversity_matrix')
        if diversity_matrix is not None:
            max_diversity = np.max(diversity_matrix)
            self.diversity_vars['max_diversity'].set(f"{max_diversity:.3f}")
            
            # Display sample of diversity matrix
            self._display_diversity_matrix(diversity_matrix)
    
    def _display_diversity_matrix(self, matrix: np.ndarray):
        """Display diversity matrix (sample)"""
        self.diversity_text.delete(1.0, tk.END)
        
        # Show first 10x10 of matrix if larger
        display_size = min(10, matrix.shape[0])
        sample_matrix = matrix[:display_size, :display_size]
        
        # Create header
        header = "     " + "".join(f"{i+1:>8}" for i in range(display_size))
        self.diversity_text.insert(tk.END, header + "\n")
        
        # Add matrix rows
        for i in range(display_size):
            row = f"{i+1:>3}: " + "".join(f"{sample_matrix[i,j]:>8.3f}" for j in range(display_size))
            self.diversity_text.insert(tk.END, row + "\n")
        
        if matrix.shape[0] > display_size:
            self.diversity_text.insert(tk.END, f"\n... (showing first {display_size}x{display_size} of {matrix.shape[0]}x{matrix.shape[1]} matrix)")
    
    def _update_profiling(self):
        """Update profiling information"""
        self.profiling_text.delete(1.0, tk.END)
        
        profiling_data = self.results.get('profiling')
        if profiling_data:
            # Display profiling information
            for section, data in profiling_data.items():
                self.profiling_text.insert(tk.END, f"{section}:\n")
                if isinstance(data, dict):
                    for key, value in data.items():
                        self.profiling_text.insert(tk.END, f"  {key}: {value}\n")
                else:
                    self.profiling_text.insert(tk.END, f"  {data}\n")
                self.profiling_text.insert(tk.END, "\n")
        else:
            self.profiling_text.insert(tk.END, "No profiling data available.\nEnable profiling in configuration to see performance metrics.")
    
    def _sort_lineups(self, event=None):
        """Sort lineups table"""
        # This would implement sorting logic
        pass
    
    def _filter_lineups(self, event=None):
        """Filter lineups table"""
        # This would implement filtering logic
        pass
    
    def _on_lineup_double_click(self, event):
        """Handle double-click on lineup"""
        selection = self.lineups_tree.selection()
        if selection:
            item = self.lineups_tree.item(selection[0])
            rank = item['values'][0]
            self.lineup_selection_combo.set(f"Lineup {rank} (Score: {item['values'][1]})")
            self._show_selected_lineup()
            self.notebook.select(2)  # Switch to individual tab
    
    def _show_selected_lineup(self, event=None):
        """Show selected individual lineup"""
        selection = self.lineup_selection_var.get()
        if not selection:
            return
        
        # Extract lineup number from selection
        try:
            lineup_num = int(selection.split()[1]) - 1
            lineups = self.results.get('lineups', [])
            scores = self.results.get('scores', [])
            
            if 0 <= lineup_num < len(lineups):
                lineup = lineups[lineup_num]
                score = scores[lineup_num]
                
                # Update lineup info
                total_salary = lineup.get('salary', pd.Series()).sum()
                salary_cap = self.results.get('site_settings', {}).get('salary_cap', 50000)
                
                self.lineup_info_vars['lineup_score'].set(f"{score:.2f}")
                self.lineup_info_vars['lineup_salary'].set(f"${total_salary:,}")
                self.lineup_info_vars['lineup_remaining'].set(f"${salary_cap - total_salary:,}")
                self.lineup_info_vars['lineup_rank'].set(str(lineup_num + 1))
                
                # Update individual lineup table
                self._update_individual_table(lineup)
                
        except (ValueError, IndexError):
            pass
    
    def _update_individual_table(self, lineup: pd.DataFrame):
        """Update individual lineup table"""
        # Clear existing items
        for item in self.individual_tree.get_children():
            self.individual_tree.delete(item)
        
        # Set up columns
        columns = ['Position', 'Player', 'Team', 'Salary', 'Projection']
        self.individual_tree['columns'] = columns
        self.individual_tree['show'] = 'headings'
        
        for col in columns:
            self.individual_tree.heading(col, text=col)
            self.individual_tree.column(col, width=120)
        
        # Add data
        for _, player in lineup.iterrows():
            values = [
                player.get('pos', ''),
                player.get('player', ''),
                player.get('team', ''),
                f"${player.get('salary', 0):,}",
                f"{player.get('proj', 0):.2f}"
            ]
            self.individual_tree.insert('', tk.END, values=values)
