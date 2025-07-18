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
        """Create diversity analysis tab with player overlap analysis"""
        diversity_frame = ttk.Frame(self.notebook)
        self.notebook.add(diversity_frame, text="Diversity Analysis")
        
        # Create main scrollable frame
        canvas = tk.Canvas(diversity_frame)
        scrollbar = ttk.Scrollbar(diversity_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Summary metrics at the top
        summary_frame = ttk.LabelFrame(scrollable_frame, text="Overlap Summary", padding=10)
        summary_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.overlap_summary_vars = {}
        summary_metrics = [
            ('total_players', 'Total Unique Players:', '0'),
            ('core_players', 'Core Players (70%+ usage):', '0'),
            ('shared_players', 'Shared Players (30-70%):', '0'),
            ('unique_players', 'Unique Players (1 lineup):', '0'),
            ('avg_overlap', 'Average Lineup Overlap:', '0.000'),
            ('overlap_risk', 'Overlap Risk Score:', 'Low')
        ]
        
        for i, (key, label, default) in enumerate(summary_metrics):
            metric_frame = ttk.Frame(summary_frame)
            metric_frame.grid(row=i//2, column=i%2, sticky=tk.W, padx=15, pady=3)
            
            ttk.Label(metric_frame, text=label, font=('TkDefaultFont', 9, 'bold')).pack(side=tk.LEFT)
            
            self.overlap_summary_vars[key] = tk.StringVar(value=default)
            label_widget = ttk.Label(metric_frame, textvariable=self.overlap_summary_vars[key])
            label_widget.pack(side=tk.LEFT, padx=(5, 0))
        
        # Player usage analysis
        usage_frame = ttk.LabelFrame(scrollable_frame, text="Player Usage Analysis", padding=10)
        usage_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Controls for player usage table
        usage_controls = ttk.Frame(usage_frame)
        usage_controls.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(usage_controls, text="Sort by:").pack(side=tk.LEFT)
        self.usage_sort_var = tk.StringVar(value="Usage Count")
        usage_sort_combo = ttk.Combobox(
            usage_controls,
            textvariable=self.usage_sort_var,
            values=["Usage Count", "Usage %", "Player Name", "Position", "Salary", "Projection"],
            state="readonly",
            width=15
        )
        usage_sort_combo.pack(side=tk.LEFT, padx=(5, 15))
        usage_sort_combo.bind('<<ComboboxSelected>>', self._sort_player_usage)
        
        ttk.Label(usage_controls, text="Filter:").pack(side=tk.LEFT)
        self.usage_filter_var = tk.StringVar(value="All Players")
        usage_filter_combo = ttk.Combobox(
            usage_controls,
            textvariable=self.usage_filter_var,
            values=["All Players", "Core Players (70%+)", "Shared Players (30-70%)", "Unique Players (1 lineup)"],
            state="readonly",
            width=20
        )
        usage_filter_combo.pack(side=tk.LEFT, padx=(5, 0))
        usage_filter_combo.bind('<<ComboboxSelected>>', self._filter_player_usage)
        
        # Player usage table
        self.player_usage_tree = ttk.Treeview(usage_frame, height=12)
        self.player_usage_tree.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar for player usage
        usage_scrollbar = ttk.Scrollbar(usage_frame, orient=tk.VERTICAL, command=self.player_usage_tree.yview)
        usage_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.player_usage_tree.configure(yscrollcommand=usage_scrollbar.set)
        
        # Position overlap breakdown
        position_frame = ttk.LabelFrame(scrollable_frame, text="Position Overlap Breakdown", padding=10)
        position_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.position_overlap_tree = ttk.Treeview(position_frame, height=6)
        self.position_overlap_tree.pack(fill=tk.BOTH, expand=True)
        
        # Stack analysis
        stack_frame = ttk.LabelFrame(scrollable_frame, text="Stack Analysis", padding=10)
        stack_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.stack_text = tk.Text(stack_frame, height=8, font=('TkDefaultFont', 9))
        self.stack_text.pack(fill=tk.BOTH, expand=True)
        
        stack_scrollbar = ttk.Scrollbar(stack_frame, orient=tk.VERTICAL, command=self.stack_text.yview)
        stack_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.stack_text.configure(yscrollcommand=stack_scrollbar.set)
        
        # Insights and recommendations
        insights_frame = ttk.LabelFrame(scrollable_frame, text="Insights & Recommendations", padding=10)
        insights_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.insights_text = tk.Text(insights_frame, height=6, font=('TkDefaultFont', 9), wrap=tk.WORD)
        self.insights_text.pack(fill=tk.BOTH, expand=True)
        
        insights_scrollbar = ttk.Scrollbar(insights_frame, orient=tk.VERTICAL, command=self.insights_text.yview)
        insights_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.insights_text.configure(yscrollcommand=insights_scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", _on_mousewheel)
    
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
            self.notebook.tab(1, text=f"All Lineups ({len(self.results.get('lineups', []))})")
            self.notebook.tab(2, text="Lineup Details")
            self.notebook.tab(3, text="Diversity Analysis")
        else:
            # For single lineup mode, disable tabs that don't make sense
            self.notebook.tab(1, state="normal")  # Show single lineup in "All Lineups" tab
            self.notebook.tab(2, state="disabled")  # Lineup Details not meaningful for single lineup
            self.notebook.tab(3, state="disabled")  # Diversity Analysis not meaningful for single lineup
            
            # Update tab text for single lineup context
            self.notebook.tab(1, text="Lineup View")
            self.notebook.tab(2, text="Lineup Details")
            self.notebook.tab(3, text="Diversity Analysis")
    
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
        
        # Sort lineups by score in descending order (highest score first)
        lineup_score_pairs = list(zip(lineups, scores))
        lineup_score_pairs.sort(key=lambda x: x[1], reverse=True)
        sorted_lineups, sorted_scores = zip(*lineup_score_pairs) if lineup_score_pairs else ([], [])
        
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
        
        # Add data (now sorted by score)
        for i, (lineup, score) in enumerate(zip(sorted_lineups, sorted_scores)):
            # Calculate total salary
            total_salary = lineup.get('salary', pd.Series()).sum()
            
            # Get players by position with proper FLEX handling
            players_by_pos = {}
            flex_eligible = []  # Track RB/WR/TE that could be FLEX
            
            for _, player in lineup.iterrows():
                # Access position and player name safely
                pos = player.get('pos', '')
                player_name = player.get('player', '')
                
                if pos not in players_by_pos:
                    players_by_pos[pos] = []
                players_by_pos[pos].append(player_name)
                
                # Track FLEX-eligible players
                if pos in ['RB', 'WR', 'TE']:
                    flex_eligible.append((pos, player_name))
            
            # Build row values
            values = [
                str(i + 1),
                f"{score:.2f}",
                f"${total_salary:,}"
            ]
            
            # Add position players with proper FLEX assignment
            # QB (1)
            if 'QB' in players_by_pos and players_by_pos['QB']:
                values.append(players_by_pos['QB'][0])
            else:
                values.append('')
            
            # RB (2) - RB1, RB2
            rb_count = 0
            for _ in range(2):  # RB1, RB2
                if 'RB' in players_by_pos and len(players_by_pos['RB']) > rb_count:
                    values.append(players_by_pos['RB'][rb_count])
                    rb_count += 1
                else:
                    values.append('')
            
            # WR (3) - WR1, WR2, WR3
            wr_count = 0
            for _ in range(3):  # WR1, WR2, WR3
                if 'WR' in players_by_pos and len(players_by_pos['WR']) > wr_count:
                    values.append(players_by_pos['WR'][wr_count])
                    wr_count += 1
                else:
                    values.append('')
            
            # TE (1)
            if 'TE' in players_by_pos and players_by_pos['TE']:
                values.append(players_by_pos['TE'][0])
            else:
                values.append('')
            
            # FLEX (1) - Find the extra RB/WR/TE
            flex_player = ''
            # Check for extra RB
            if 'RB' in players_by_pos and len(players_by_pos['RB']) > 2:
                flex_player = players_by_pos['RB'][2]
            # Check for extra WR
            elif 'WR' in players_by_pos and len(players_by_pos['WR']) > 3:
                flex_player = players_by_pos['WR'][3]
            # Check for extra TE
            elif 'TE' in players_by_pos and len(players_by_pos['TE']) > 1:
                flex_player = players_by_pos['TE'][1]
            values.append(flex_player)
            
            # DST (1)
            if 'DST' in players_by_pos and players_by_pos['DST']:
                values.append(players_by_pos['DST'][0])
            else:
                values.append('')
            
            self.lineups_tree.insert('', tk.END, values=values)
    
    def _update_individual_selection(self):
        """Update individual lineup selection dropdown"""
        lineups = self.results.get('lineups', [])
        scores = self.results.get('scores', [])
        
        if not lineups:
            return
        
        # Sort lineups by score in descending order (same as table)
        lineup_score_pairs = list(zip(lineups, scores))
        lineup_score_pairs.sort(key=lambda x: x[1], reverse=True)
        sorted_lineups, sorted_scores = zip(*lineup_score_pairs) if lineup_score_pairs else ([], [])
        
        # Create selection options (now sorted by score)
        options = []
        for i, score in enumerate(sorted_scores):
            options.append(f"Lineup {i+1} (Score: {score:.2f})")
        
        self.lineup_selection_combo['values'] = options
        if options:
            self.lineup_selection_combo.set(options[0])
            self._show_selected_lineup()
    
    def _update_diversity_analysis(self):
        """Update diversity analysis with player overlap analysis"""
        lineups = self.results.get('lineups', [])
        
        if not lineups or len(lineups) <= 1:
            # Single lineup mode - show message
            self._show_single_lineup_message()
            return
        
        # Perform player overlap analysis
        overlap_analysis = self._analyze_player_overlap(lineups)
        
        # Update summary metrics
        self._update_overlap_summary(overlap_analysis)
        
        # Update player usage table
        self._update_player_usage_table(overlap_analysis['player_usage'])
        
        # Update position overlap breakdown
        self._update_position_overlap_table(overlap_analysis['position_overlap'])
        
        # Update stack analysis
        self._update_stack_analysis(overlap_analysis['stacks'])
        
        # Update insights and recommendations
        self._update_insights(overlap_analysis['insights'])
    
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
            
            # Sort lineups by score in descending order (same as table and dropdown)
            lineup_score_pairs = list(zip(lineups, scores))
            lineup_score_pairs.sort(key=lambda x: x[1], reverse=True)
            sorted_lineups, sorted_scores = zip(*lineup_score_pairs) if lineup_score_pairs else ([], [])
            
            if 0 <= lineup_num < len(sorted_lineups):
                lineup = sorted_lineups[lineup_num]
                score = sorted_scores[lineup_num]
                
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
    
    def _analyze_player_overlap(self, lineups):
        """Analyze player overlap across lineups"""
        if not lineups or len(lineups) <= 1:
            return {
                'player_usage': {},
                'position_overlap': {},
                'stacks': {},
                'insights': []
            }
        
        # Collect all players and their usage
        player_usage = {}
        position_usage = {}
        team_usage = {}
        
        total_lineups = len(lineups)
        
        for lineup_idx, lineup in enumerate(lineups):
            for _, player in lineup.iterrows():
                player_name = player.get('player', '')
                position = player.get('pos', '')
                team = player.get('team', '')
                salary = player.get('salary', 0)
                projection = player.get('proj', 0)
                
                # Track player usage
                if player_name not in player_usage:
                    player_usage[player_name] = {
                        'count': 0,
                        'lineups': [],
                        'position': position,
                        'team': team,
                        'salary': salary,
                        'projection': projection
                    }
                
                player_usage[player_name]['count'] += 1
                player_usage[player_name]['lineups'].append(lineup_idx + 1)
                
                # Track position usage
                if position not in position_usage:
                    position_usage[position] = {}
                if player_name not in position_usage[position]:
                    position_usage[position][player_name] = 0
                position_usage[position][player_name] += 1
                
                # Track team usage
                if team not in team_usage:
                    team_usage[team] = {}
                if player_name not in team_usage[team]:
                    team_usage[team][player_name] = 0
                team_usage[team][player_name] += 1
        
        # Calculate usage percentages
        for player_data in player_usage.values():
            player_data['usage_pct'] = (player_data['count'] / total_lineups) * 100
        
        # Analyze position overlap
        position_overlap = self._analyze_position_overlap(position_usage, total_lineups)
        
        # Detect stacks
        stacks = self._detect_stacks(lineups, team_usage)
        
        # Generate insights
        insights = self._generate_overlap_insights(player_usage, position_overlap, stacks, total_lineups)
        
        return {
            'player_usage': player_usage,
            'position_overlap': position_overlap,
            'stacks': stacks,
            'insights': insights
        }
    
    def _analyze_position_overlap(self, position_usage, total_lineups):
        """Analyze overlap by position"""
        position_overlap = {}
        
        for position, players in position_usage.items():
            # Find most used player at this position
            most_used_player = max(players.items(), key=lambda x: x[1])
            most_used_count = most_used_player[1]
            most_used_pct = (most_used_count / total_lineups) * 100
            
            # Calculate diversity score (lower = more diverse)
            unique_players = len(players)
            diversity_score = 100 - most_used_pct
            
            position_overlap[position] = {
                'most_used_player': most_used_player[0],
                'most_used_count': most_used_count,
                'most_used_pct': most_used_pct,
                'unique_players': unique_players,
                'diversity_score': diversity_score,
                'all_players': players
            }
        
        return position_overlap
    
    def _detect_stacks(self, lineups, team_usage):
        """Detect QB-WR stacks and team correlations"""
        stacks = {
            'qb_wr_stacks': {},
            'team_stacks': {},
            'game_stacks': {}
        }
        
        # Analyze QB-WR stacks
        for lineup_idx, lineup in enumerate(lineups):
            qb_team = None
            wr_players = []
            
            for _, player in lineup.iterrows():
                position = player.get('pos', '')
                team = player.get('team', '')
                player_name = player.get('player', '')
                
                if position == 'QB':
                    qb_team = team
                elif position == 'WR':
                    wr_players.append((player_name, team))
            
            # Check for QB-WR stacks
            if qb_team:
                for wr_name, wr_team in wr_players:
                    if qb_team == wr_team:
                        stack_key = f"{qb_team} QB-WR"
                        if stack_key not in stacks['qb_wr_stacks']:
                            stacks['qb_wr_stacks'][stack_key] = {
                                'count': 0,
                                'lineups': []
                            }
                        stacks['qb_wr_stacks'][stack_key]['count'] += 1
                        stacks['qb_wr_stacks'][stack_key]['lineups'].append(lineup_idx + 1)
        
        # Analyze team stacks (2+ players from same team)
        for lineup_idx, lineup in enumerate(lineups):
            team_counts = {}
            
            for _, player in lineup.iterrows():
                team = player.get('team', '')
                if team not in team_counts:
                    team_counts[team] = 0
                team_counts[team] += 1
            
            # Find teams with 2+ players
            for team, count in team_counts.items():
                if count >= 2:
                    stack_key = f"{team} ({count} players)"
                    if stack_key not in stacks['team_stacks']:
                        stacks['team_stacks'][stack_key] = {
                            'count': 0,
                            'lineups': []
                        }
                    stacks['team_stacks'][stack_key]['count'] += 1
                    stacks['team_stacks'][stack_key]['lineups'].append(lineup_idx + 1)
        
        return stacks
    
    def _generate_overlap_insights(self, player_usage, position_overlap, stacks, total_lineups):
        """Generate actionable insights and recommendations"""
        insights = []
        
        # Analyze overall overlap risk
        core_players = [p for p, data in player_usage.items() if data['usage_pct'] >= 70]
        shared_players = [p for p, data in player_usage.items() if 30 <= data['usage_pct'] < 70]
        unique_players = [p for p, data in player_usage.items() if data['usage_pct'] < 30]
        
        # Overall risk assessment
        if len(core_players) > 3:
            insights.append(f"⚠️ HIGH OVERLAP RISK: {len(core_players)} core players used in 70%+ of lineups")
        elif len(core_players) > 1:
            insights.append(f"⚠️ MODERATE OVERLAP RISK: {len(core_players)} core players used in 70%+ of lineups")
        else:
            insights.append(f"✅ GOOD DIVERSITY: Only {len(core_players)} core players")
        
        # Position-specific insights
        high_overlap_positions = []
        for pos, data in position_overlap.items():
            if data['most_used_pct'] >= 80:
                high_overlap_positions.append(f"{pos} ({data['most_used_pct']:.0f}% use {data['most_used_player']})")
        
        if high_overlap_positions:
            insights.append(f"🔴 High position overlap: {', '.join(high_overlap_positions)}")
        
        # Stack insights
        if stacks['qb_wr_stacks']:
            most_common_stack = max(stacks['qb_wr_stacks'].items(), key=lambda x: x[1]['count'])
            insights.append(f"📊 Most common stack: {most_common_stack[0]} in {most_common_stack[1]['count']} lineups")
        
        # Recommendations
        if len(core_players) > 2:
            insights.append(f"💡 RECOMMENDATION: Consider replacing {core_players[0]} in some lineups to reduce overlap")
        
        if any(data['most_used_pct'] > 90 for data in position_overlap.values()):
            worst_pos = max(position_overlap.items(), key=lambda x: x[1]['most_used_pct'])
            insights.append(f"💡 RECOMMENDATION: {worst_pos[0]} position needs more diversity")
        
        if len(unique_players) < total_lineups * 2:
            insights.append(f"💡 RECOMMENDATION: Consider increasing diversity settings for more unique players")
        
        return insights
    
    def _show_single_lineup_message(self):
        """Show message for single lineup mode"""
        # Clear all diversity widgets and show message
        for var in self.overlap_summary_vars.values():
            var.set("N/A")
        
        # Clear tables
        for item in self.player_usage_tree.get_children():
            self.player_usage_tree.delete(item)
        for item in self.position_overlap_tree.get_children():
            self.position_overlap_tree.delete(item)
        
        # Show message in insights
        self.insights_text.delete(1.0, tk.END)
        self.insights_text.insert(tk.END, "Diversity analysis is only available for multiple lineups.\n\n")
        self.insights_text.insert(tk.END, "To enable diversity analysis:\n")
        self.insights_text.insert(tk.END, "1. Set 'Target Lineups' to 2 or more in Configuration\n")
        self.insights_text.insert(tk.END, "2. Select 'Post-processing (Multilineup)' optimizer\n")
        self.insights_text.insert(tk.END, "3. Run optimization to generate multiple diverse lineups")
        
        self.stack_text.delete(1.0, tk.END)
        self.stack_text.insert(tk.END, "Stack analysis requires multiple lineups.")
    
    def _update_overlap_summary(self, overlap_analysis):
        """Update overlap summary metrics"""
        player_usage = overlap_analysis['player_usage']
        total_lineups = len(self.results.get('lineups', []))
        
        # Count players by usage category
        core_players = len([p for p, data in player_usage.items() if data['usage_pct'] >= 70])
        shared_players = len([p for p, data in player_usage.items() if 30 <= data['usage_pct'] < 70])
        unique_players = len([p for p, data in player_usage.items() if data['usage_pct'] < 30])
        
        # Calculate average overlap
        diversity_metrics = self.results.get('diversity_metrics', {})
        avg_overlap = diversity_metrics.get('avg_overlap', 0)
        
        # Determine risk level
        if core_players > 3 or avg_overlap > 0.7:
            risk_level = "High"
        elif core_players > 1 or avg_overlap > 0.5:
            risk_level = "Moderate"
        else:
            risk_level = "Low"
        
        # Update summary variables
        self.overlap_summary_vars['total_players'].set(str(len(player_usage)))
        self.overlap_summary_vars['core_players'].set(str(core_players))
        self.overlap_summary_vars['shared_players'].set(str(shared_players))
        self.overlap_summary_vars['unique_players'].set(str(unique_players))
        self.overlap_summary_vars['avg_overlap'].set(f"{avg_overlap:.3f}")
        self.overlap_summary_vars['overlap_risk'].set(risk_level)
    
    def _update_player_usage_table(self, player_usage):
        """Update player usage table"""
        # Clear existing items
        for item in self.player_usage_tree.get_children():
            self.player_usage_tree.delete(item)
        
        # Set up columns
        columns = ['Player', 'Position', 'Team', 'Used In', 'Usage %', 'Salary', 'Projection']
        self.player_usage_tree['columns'] = columns
        self.player_usage_tree['show'] = 'headings'
        
        for col in columns:
            self.player_usage_tree.heading(col, text=col)
            if col == 'Player':
                self.player_usage_tree.column(col, width=120)
            elif col in ['Position', 'Team']:
                self.player_usage_tree.column(col, width=60)
            elif col in ['Used In', 'Usage %']:
                self.player_usage_tree.column(col, width=80)
            else:
                self.player_usage_tree.column(col, width=90)
        
        # Sort players by usage count (descending)
        sorted_players = sorted(player_usage.items(), key=lambda x: x[1]['count'], reverse=True)
        
        # Add data with color coding
        for player_name, data in sorted_players:
            usage_pct = data['usage_pct']
            
            values = [
                player_name,
                data['position'],
                data['team'],
                f"{data['count']}/{len(self.results.get('lineups', []))}",
                f"{usage_pct:.1f}%",
                f"${data['salary']:,}",
                f"{data['projection']:.2f}"
            ]
            
            # Add with tags for color coding
            if usage_pct >= 70:
                tag = 'high_overlap'
            elif usage_pct >= 30:
                tag = 'medium_overlap'
            else:
                tag = 'low_overlap'
            
            item = self.player_usage_tree.insert('', tk.END, values=values, tags=(tag,))
        
        # Configure tags for color coding
        self.player_usage_tree.tag_configure('high_overlap', background='#ffcccc')  # Light red
        self.player_usage_tree.tag_configure('medium_overlap', background='#fff2cc')  # Light yellow
        self.player_usage_tree.tag_configure('low_overlap', background='#ccffcc')  # Light green
    
    def _update_position_overlap_table(self, position_overlap):
        """Update position overlap breakdown table"""
        # Clear existing items
        for item in self.position_overlap_tree.get_children():
            self.position_overlap_tree.delete(item)
        
        # Set up columns
        columns = ['Position', 'Most Used Player', 'Usage %', 'Unique Players', 'Diversity Score']
        self.position_overlap_tree['columns'] = columns
        self.position_overlap_tree['show'] = 'headings'
        
        for col in columns:
            self.position_overlap_tree.heading(col, text=col)
            if col == 'Position':
                self.position_overlap_tree.column(col, width=80)
            elif col == 'Most Used Player':
                self.position_overlap_tree.column(col, width=150)
            else:
                self.position_overlap_tree.column(col, width=100)
        
        # Add data
        for position, data in position_overlap.items():
            values = [
                position,
                data['most_used_player'],
                f"{data['most_used_pct']:.1f}%",
                str(data['unique_players']),
                f"{data['diversity_score']:.1f}"
            ]
            
            # Color code by overlap level
            if data['most_used_pct'] >= 80:
                tag = 'high_overlap'
            elif data['most_used_pct'] >= 50:
                tag = 'medium_overlap'
            else:
                tag = 'low_overlap'
            
            self.position_overlap_tree.insert('', tk.END, values=values, tags=(tag,))
        
        # Configure tags
        self.position_overlap_tree.tag_configure('high_overlap', background='#ffcccc')
        self.position_overlap_tree.tag_configure('medium_overlap', background='#fff2cc')
        self.position_overlap_tree.tag_configure('low_overlap', background='#ccffcc')
    
    def _update_stack_analysis(self, stacks):
        """Update stack analysis text"""
        self.stack_text.delete(1.0, tk.END)
        
        # QB-WR Stacks
        if stacks['qb_wr_stacks']:
            self.stack_text.insert(tk.END, "QB-WR Stacks:\n")
            for stack_name, data in sorted(stacks['qb_wr_stacks'].items(), key=lambda x: x[1]['count'], reverse=True):
                self.stack_text.insert(tk.END, f"  • {stack_name}: {data['count']} lineups\n")
            self.stack_text.insert(tk.END, "\n")
        
        # Team Stacks
        if stacks['team_stacks']:
            self.stack_text.insert(tk.END, "Team Stacks (2+ players):\n")
            for stack_name, data in sorted(stacks['team_stacks'].items(), key=lambda x: x[1]['count'], reverse=True):
                self.stack_text.insert(tk.END, f"  • {stack_name}: {data['count']} lineups\n")
            self.stack_text.insert(tk.END, "\n")
        
        if not stacks['qb_wr_stacks'] and not stacks['team_stacks']:
            self.stack_text.insert(tk.END, "No significant stacks detected.\n")
            self.stack_text.insert(tk.END, "This indicates good lineup diversity with minimal correlation risk.")
    
    def _update_insights(self, insights):
        """Update insights and recommendations"""
        self.insights_text.delete(1.0, tk.END)
        
        if insights:
            for insight in insights:
                self.insights_text.insert(tk.END, insight + "\n\n")
        else:
            self.insights_text.insert(tk.END, "No specific insights available.")
    
    def _sort_player_usage(self, event=None):
        """Sort player usage table"""
        # This would implement sorting logic based on selected column
        pass
    
    def _filter_player_usage(self, event=None):
        """Filter player usage table"""
        # This would implement filtering logic based on selected filter
        pass
