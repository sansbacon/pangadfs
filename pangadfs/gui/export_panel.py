# pangadfs/gui/export_panel.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Dict, Any, Optional, Callable
import pandas as pd
import json
from pathlib import Path


class ExportPanel:
    """Panel for exporting optimization results"""
    
    def __init__(self, parent: tk.Widget, get_results_callback: Callable[[], Optional[Dict[str, Any]]]):
        self.parent = parent
        self.get_results = get_results_callback
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the export UI"""
        # Main frame
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Export options section
        options_frame = ttk.LabelFrame(main_frame, text="Export Options", padding=10)
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Format selection
        format_frame = ttk.Frame(options_frame)
        format_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(format_frame, text="Export Format:", font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W)
        
        self.format_var = tk.StringVar(value="CSV")
        format_options = ["CSV", "Excel", "JSON"]
        
        format_radio_frame = ttk.Frame(format_frame)
        format_radio_frame.pack(fill=tk.X, pady=2)
        
        for i, fmt in enumerate(format_options):
            ttk.Radiobutton(
                format_radio_frame, 
                text=fmt, 
                variable=self.format_var, 
                value=fmt,
                command=self._on_format_change
            ).pack(side=tk.LEFT, padx=(0, 20))
        
        # Content selection
        content_frame = ttk.Frame(options_frame)
        content_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(content_frame, text="Export Content:", font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W)
        
        # Content checkboxes
        self.content_vars = {}
        content_options = [
            ('lineups', 'All Lineups', True),
            ('summary', 'Summary Statistics', True),
            ('diversity', 'Diversity Metrics', False),
            ('profiling', 'Performance Data', False)
        ]
        
        content_grid = ttk.Frame(content_frame)
        content_grid.pack(fill=tk.X, pady=5)
        
        for i, (key, label, default) in enumerate(content_options):
            self.content_vars[key] = tk.BooleanVar(value=default)
            cb = ttk.Checkbutton(content_grid, text=label, variable=self.content_vars[key])
            cb.grid(row=i//2, column=i%2, sticky=tk.W, padx=(0, 20), pady=2)
        
        # Lineup selection
        self.lineup_frame = ttk.Frame(options_frame)
        self.lineup_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.lineup_label = ttk.Label(self.lineup_frame, text="Lineup Selection:", font=('TkDefaultFont', 9, 'bold'))
        self.lineup_label.pack(anchor=tk.W)
        
        self.lineup_selection_var = tk.StringVar(value="all")
        
        lineup_options = [
            ("all", "All Lineups"),
            ("top", "Top N Lineups"),
            ("range", "Lineup Range")
        ]
        
        self.lineup_radio_frame = ttk.Frame(self.lineup_frame)
        self.lineup_radio_frame.pack(fill=tk.X, pady=2)
        
        self.lineup_radio_buttons = []
        for i, (value, label) in enumerate(lineup_options):
            rb = ttk.Radiobutton(
                self.lineup_radio_frame, 
                text=label, 
                variable=self.lineup_selection_var, 
                value=value,
                command=self._on_selection_change
            )
            rb.pack(side=tk.LEFT, padx=(0, 20))
            self.lineup_radio_buttons.append(rb)
        
        # Selection parameters
        self.params_frame = ttk.Frame(self.lineup_frame)
        self.params_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Top N parameter
        self.top_n_frame = ttk.Frame(self.params_frame)
        ttk.Label(self.top_n_frame, text="Number of lineups:").pack(side=tk.LEFT)
        self.top_n_var = tk.StringVar(value="10")
        ttk.Entry(self.top_n_frame, textvariable=self.top_n_var, width=10).pack(side=tk.LEFT, padx=(5, 0))
        
        # Range parameters
        self.range_frame = ttk.Frame(self.params_frame)
        ttk.Label(self.range_frame, text="From:").pack(side=tk.LEFT)
        self.range_start_var = tk.StringVar(value="1")
        ttk.Entry(self.range_frame, textvariable=self.range_start_var, width=8).pack(side=tk.LEFT, padx=(5, 10))
        ttk.Label(self.range_frame, text="To:").pack(side=tk.LEFT)
        self.range_end_var = tk.StringVar(value="10")
        ttk.Entry(self.range_frame, textvariable=self.range_end_var, width=8).pack(side=tk.LEFT, padx=(5, 0))
        
        self._on_selection_change()  # Initialize visibility
        
        # File naming section
        naming_frame = ttk.LabelFrame(main_frame, text="File Naming", padding=10)
        naming_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Filename template
        filename_frame = ttk.Frame(naming_frame)
        filename_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(filename_frame, text="Filename:").pack(side=tk.LEFT)
        self.filename_var = tk.StringVar(value="pangadfs_results")
        ttk.Entry(filename_frame, textvariable=self.filename_var, width=30).pack(side=tk.LEFT, padx=(5, 0), fill=tk.X, expand=True)
        
        # Auto-naming options
        auto_frame = ttk.Frame(naming_frame)
        auto_frame.pack(fill=tk.X)
        
        self.auto_timestamp_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(auto_frame, text="Add timestamp", variable=self.auto_timestamp_var).pack(side=tk.LEFT)
        
        self.auto_lineups_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(auto_frame, text="Add lineup count", variable=self.auto_lineups_var).pack(side=tk.LEFT, padx=(20, 0))
        
        # Export actions section
        actions_frame = ttk.LabelFrame(main_frame, text="Export Actions", padding=10)
        actions_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Export buttons
        button_frame = ttk.Frame(actions_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(
            button_frame, 
            text="Export to File...", 
            command=self._export_to_file,
            style="Accent.TButton"
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            button_frame, 
            text="Quick Export", 
            command=self._quick_export
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            button_frame, 
            text="Preview Export", 
            command=self._preview_export
        ).pack(side=tk.LEFT)
        
        # Status section
        status_frame = ttk.LabelFrame(main_frame, text="Export Status", padding=10)
        status_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status text area
        self.status_text = tk.Text(status_frame, height=8, font=('Consolas', 9))
        self.status_text.pack(fill=tk.BOTH, expand=True)
        
        status_scrollbar = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.configure(yscrollcommand=status_scrollbar.set)
        
        # Initial status message
        self._log_status("Export panel ready. Configure options and click 'Export to File' to begin.")
    
    def configure_for_results(self, results: Dict[str, Any]):
        """Configure export panel based on results (single vs multilineup)"""
        if not results:
            return
        
        # Determine if this is multilineup mode
        lineups = results.get('lineups', [])
        if not lineups:
            lineups = [results.get('best_lineup')]
        
        is_multilineup = len(lineups) > 1
        
        if is_multilineup:
            # Enable all lineup selection options for multilineup
            self.lineup_label.config(text="Lineup Selection:")
            for rb in self.lineup_radio_buttons:
                rb.config(state="normal")
            
            # Enable diversity content option
            if 'diversity' in self.content_vars:
                # Find the diversity checkbox and enable it
                for widget in self.parent.winfo_children():
                    self._enable_widget_recursive(widget, "diversity")
        else:
            # Disable multilineup-specific options for single lineup
            self.lineup_label.config(text="Lineup Selection: (Single lineup mode)")
            
            # Disable "Top N" and "Range" options, keep only "All Lineups"
            for i, rb in enumerate(self.lineup_radio_buttons):
                if i == 0:  # "All Lineups" option
                    rb.config(state="normal")
                    rb.config(text="Export Lineup")
                else:
                    rb.config(state="disabled")
            
            # Set to "all" and disable diversity content
            self.lineup_selection_var.set("all")
            if 'diversity' in self.content_vars:
                self.content_vars['diversity'].set(False)
                # Find and disable the diversity checkbox
                for widget in self.parent.winfo_children():
                    self._disable_widget_recursive(widget, "diversity")
    
    def _enable_widget_recursive(self, widget, target_text):
        """Recursively find and enable widget with target text"""
        try:
            if (
                hasattr(widget, 'cget')
                and 'text' in widget.keys()
                and target_text.lower() in widget.cget('text').lower()
            ):
                widget.config(state="normal")
        except:
            pass
        
        for child in widget.winfo_children():
            self._enable_widget_recursive(child, target_text)
    
    def _disable_widget_recursive(self, widget, target_text):
        """Recursively find and disable widget with target text"""
        try:
            if (
                hasattr(widget, 'cget')
                and 'text' in widget.keys()
                and target_text.lower() in widget.cget('text').lower()
            ):
                widget.config(state="disabled")
        except:
            pass
        
        for child in widget.winfo_children():
            self._disable_widget_recursive(child, target_text)
    
    def _on_format_change(self):
        """Handle format selection change"""
        fmt = self.format_var.get()
        
        # Update content options based on format
        if fmt == "JSON":
            # JSON can include all data types
            for var in self.content_vars.values():
                var.set(True)
        elif fmt == "Excel":
            # Excel is good for structured data
            self.content_vars['lineups'].set(True)
            self.content_vars['summary'].set(True)
        else:  # CSV
            # CSV is best for lineup data
            self.content_vars['lineups'].set(True)
            self.content_vars['summary'].set(False)
            self.content_vars['diversity'].set(False)
            self.content_vars['profiling'].set(False)
    
    def _on_selection_change(self):
        """Handle lineup selection change"""
        selection = self.lineup_selection_var.get()
        
        # Hide all parameter frames
        self.top_n_frame.pack_forget()
        self.range_frame.pack_forget()
        
        # Show relevant parameter frame
        if selection == "top":
            self.top_n_frame.pack(fill=tk.X)
        elif selection == "range":
            self.range_frame.pack(fill=tk.X)
    
    def _export_to_file(self):
        """Export results to a user-selected file"""
        results = self.get_results()
        if not results:
            messagebox.showwarning("No Results", "No optimization results available to export.")
            return
        
        # Generate filename
        filename = self._generate_filename()
        
        # Get file extension based on format
        fmt = self.format_var.get()
        if fmt == "CSV":
            ext = ".csv"
            filetypes = [("CSV files", "*.csv"), ("All files", "*.*")]
        elif fmt == "Excel":
            ext = ".xlsx"
            filetypes = [("Excel files", "*.xlsx"), ("All files", "*.*")]
        else:  # JSON
            ext = ".json"
            filetypes = [("JSON files", "*.json"), ("All files", "*.*")]
        
        # Ask user for save location
        save_path = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=ext,
            initialvalue=filename + ext,
            filetypes=filetypes
        )
        
        if save_path:
            try:
                self._perform_export(results, save_path)
                self._log_status(f"✓ Export completed successfully: {Path(save_path).name}")
                messagebox.showinfo("Export Complete", f"Results exported to:\n{save_path}")
            except Exception as e:
                self._log_status(f"✗ Export failed: {str(e)}")
                messagebox.showerror("Export Error", f"Failed to export results:\n{str(e)}")
    
    def _quick_export(self):
        """Quick export to default location"""
        results = self.get_results()
        if not results:
            messagebox.showwarning("No Results", "No optimization results available to export.")
            return
        
        try:
            # Generate filename and path
            filename = self._generate_filename()
            fmt = self.format_var.get()
            
            if fmt == "CSV":
                ext = ".csv"
            elif fmt == "Excel":
                ext = ".xlsx"
            else:  # JSON
                ext = ".json"
            
            save_path = Path.cwd() / (filename + ext)
            
            self._perform_export(results, str(save_path))
            self._log_status(f"✓ Quick export completed: {save_path.name}")
            messagebox.showinfo("Export Complete", f"Results exported to:\n{save_path}")
            
        except Exception as e:
            self._log_status(f"✗ Quick export failed: {str(e)}")
            messagebox.showerror("Export Error", f"Failed to export results:\n{str(e)}")
    
    def _preview_export(self):
        """Preview what will be exported"""
        results = self.get_results()
        if not results:
            messagebox.showwarning("No Results", "No optimization results available to preview.")
            return
        
        try:
            # Get selected lineups
            selected_lineups, selected_scores = self._get_selected_lineups(results)
            
            # Create preview text
            preview_text = f"Export Preview\n{'='*50}\n\n"
            preview_text += f"Format: {self.format_var.get()}\n"
            preview_text += f"Filename: {self._generate_filename()}\n\n"
            
            # Content summary
            preview_text += "Content to export:\n"
            for key, var in self.content_vars.items():
                if var.get():
                    preview_text += f"  ✓ {key.title()}\n"
            
            preview_text += f"\nLineups: {len(selected_lineups)} of {len(results.get('lineups', []))}\n"
            
            if selected_scores:
                preview_text += f"Score range: {min(selected_scores):.2f} - {max(selected_scores):.2f}\n"
            
            # Show preview window
            self._show_preview_window(preview_text)
            
        except Exception as e:
            messagebox.showerror("Preview Error", f"Failed to generate preview:\n{str(e)}")
    
    def _perform_export(self, results: Dict[str, Any], save_path: str):
        """Perform the actual export"""
        fmt = self.format_var.get()
        
        # Get selected lineups
        selected_lineups, selected_scores = self._get_selected_lineups(results)
        
        self._log_status(f"Starting {fmt} export to {Path(save_path).name}...")
        self._log_status(f"Exporting {len(selected_lineups)} lineups...")
        
        if fmt == "CSV":
            self._export_csv(results, selected_lineups, selected_scores, save_path)
        elif fmt == "Excel":
            self._export_excel(results, selected_lineups, selected_scores, save_path)
        else:  # JSON
            self._export_json(results, selected_lineups, selected_scores, save_path)
    
    def _export_csv(self, results: Dict[str, Any], lineups: list, scores: list, save_path: str):
        """Export to CSV format"""
        if not self.content_vars['lineups'].get():
            raise ValueError("CSV export requires lineup data to be selected")
        
        # Create lineup data
        export_data = []
        
        for i, (lineup, score) in enumerate(zip(lineups, scores)):
            # Calculate total salary
            total_salary = lineup.get('salary', pd.Series()).sum()
            
            # Create row for each player in lineup
            for _, player in lineup.iterrows():
                row = {
                    'Lineup_ID': i + 1,
                    'Lineup_Score': score,
                    'Total_Salary': total_salary,
                    'Position': player.get('pos', ''),
                    'Player': player.get('player', ''),
                    'Team': player.get('team', ''),
                    'Salary': player.get('salary', 0),
                    'Projection': player.get('proj', 0)
                }
                export_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(export_data)
        df.to_csv(save_path, index=False)
        
        self._log_status(f"CSV export completed: {len(export_data)} player entries")
    
    def _export_excel(self, results: Dict[str, Any], lineups: list, scores: list, save_path: str):
        """Export to Excel format"""
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            
            # Export lineups if selected
            if self.content_vars['lineups'].get():
                lineup_data = []
                
                for i, (lineup, score) in enumerate(zip(lineups, scores)):
                    total_salary = lineup.get('salary', pd.Series()).sum()
                    
                    for _, player in lineup.iterrows():
                        row = {
                            'Lineup_ID': i + 1,
                            'Lineup_Score': score,
                            'Total_Salary': total_salary,
                            'Position': player.get('pos', ''),
                            'Player': player.get('player', ''),
                            'Team': player.get('team', ''),
                            'Salary': player.get('salary', 0),
                            'Projection': player.get('proj', 0)
                        }
                        lineup_data.append(row)
                
                df_lineups = pd.DataFrame(lineup_data)
                df_lineups.to_excel(writer, sheet_name='Lineups', index=False)
                self._log_status(f"Exported lineups sheet: {len(lineup_data)} entries")
            
            # Export summary if selected
            if self.content_vars['summary'].get():
                summary_data = {
                    'Metric': ['Number of Lineups', 'Best Score', 'Average Score', 'Score Range'],
                    'Value': [
                        len(lineups),
                        max(scores) if scores else 0,
                        sum(scores) / len(scores) if scores else 0,
                        max(scores) - min(scores) if len(scores) > 1 else 0
                    ]
                }
                
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name='Summary', index=False)
                self._log_status("Exported summary sheet")
            
            # Export diversity metrics if selected
            if self.content_vars['diversity'].get():
                diversity_metrics = results.get('diversity_metrics', {})
                if diversity_metrics:
                    diversity_data = {
                        'Metric': list(diversity_metrics.keys()),
                        'Value': [str(v) for v in diversity_metrics.values()]
                    }
                    
                    df_diversity = pd.DataFrame(diversity_data)
                    df_diversity.to_excel(writer, sheet_name='Diversity', index=False)
                    self._log_status("Exported diversity sheet")
        
        self._log_status("Excel export completed")
    
    def _export_json(self, results: Dict[str, Any], lineups: list, scores: list, save_path: str):
        """Export to JSON format"""
        export_data = {}
        
        # Export lineups if selected
        if self.content_vars['lineups'].get():
            lineup_data = []
            
            for i, (lineup, score) in enumerate(zip(lineups, scores)):
                lineup_dict = {
                    'lineup_id': i + 1,
                    'score': score,
                    'total_salary': int(lineup.get('salary', pd.Series()).sum()),
                    'players': []
                }
                
                for _, player in lineup.iterrows():
                    player_dict = {
                        'position': player.get('pos', ''),
                        'name': player.get('player', ''),
                        'team': player.get('team', ''),
                        'salary': int(player.get('salary', 0)),
                        'projection': float(player.get('proj', 0))
                    }
                    lineup_dict['players'].append(player_dict)
                
                lineup_data.append(lineup_dict)
            
            export_data['lineups'] = lineup_data
            self._log_status(f"Prepared lineups data: {len(lineup_data)} lineups")
        
        # Export summary if selected
        if self.content_vars['summary'].get():
            export_data['summary'] = {
                'num_lineups': len(lineups),
                'best_score': max(scores) if scores else 0,
                'average_score': sum(scores) / len(scores) if scores else 0,
                'score_range': max(scores) - min(scores) if len(scores) > 1 else 0
            }
            self._log_status("Prepared summary data")
        
        # Export diversity metrics if selected
        if self.content_vars['diversity'].get():
            diversity_metrics = results.get('diversity_metrics', {})
            if diversity_metrics:
                # Convert numpy arrays to lists for JSON serialization
                json_diversity = {}
                for key, value in diversity_metrics.items():
                    if hasattr(value, 'tolist'):  # numpy array
                        json_diversity[key] = value.tolist()
                    else:
                        json_diversity[key] = value
                
                export_data['diversity_metrics'] = json_diversity
                self._log_status("Prepared diversity data")
        
        # Export profiling if selected
        if self.content_vars['profiling'].get():
            profiling_data = results.get('profiling')
            if profiling_data:
                export_data['profiling'] = profiling_data
                self._log_status("Prepared profiling data")
        
        # Save JSON file
        with open(save_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self._log_status("JSON export completed")
    
    def _get_selected_lineups(self, results: Dict[str, Any]):
        """Get selected lineups based on user selection"""
        all_lineups = results.get('lineups', [])
        all_scores = results.get('scores', [])
        
        if not all_lineups:
            # Single lineup mode
            all_lineups = [results.get('best_lineup')]
            all_scores = [results.get('best_score', 0)]
        
        selection = self.lineup_selection_var.get()
        
        if selection == "all":
            return all_lineups, all_scores
        elif selection == "top":
            try:
                n = int(self.top_n_var.get())
                n = min(n, len(all_lineups))
                return all_lineups[:n], all_scores[:n]
            except ValueError:
                return all_lineups, all_scores
        elif selection == "range":
            try:
                start = int(self.range_start_var.get()) - 1  # Convert to 0-based
                end = int(self.range_end_var.get())
                start = max(0, start)
                end = min(end, len(all_lineups))
                return all_lineups[start:end], all_scores[start:end]
            except ValueError:
                return all_lineups, all_scores
        
        return all_lineups, all_scores
    
    def _generate_filename(self):
        """Generate filename based on options"""
        base_name = self.filename_var.get() or "pangadfs_results"
        
        # Add lineup count if selected
        if self.auto_lineups_var.get():
            results = self.get_results()
            if results:
                lineups = results.get('lineups', [])
                if not lineups:
                    lineups = [results.get('best_lineup')]
                base_name += f"_{len(lineups)}lineups"
        
        # Add timestamp if selected
        if self.auto_timestamp_var.get():
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name += f"_{timestamp}"
        
        return base_name
    
    def _show_preview_window(self, preview_text: str):
        """Show preview in a popup window"""
        preview_window = tk.Toplevel(self.parent)
        preview_window.title("Export Preview")
        preview_window.geometry("500x400")
        
        # Preview text
        text_widget = tk.Text(preview_window, wrap=tk.WORD, font=('Consolas', 9))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(preview_window, orient=tk.VERTICAL, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.insert(tk.END, preview_text)
        text_widget.configure(state=tk.DISABLED)
        
        # Close button
        ttk.Button(preview_window, text="Close", command=preview_window.destroy).pack(pady=10)
    
    def _log_status(self, message: str):
        """Log a status message"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
