# pangadfs/gui/widgets/player_pool_panel.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Dict, Any, Optional, Set, Callable
import pandas as pd
from pathlib import Path


class PlayerPoolPanel:
    """Panel for displaying and editing player pool data"""
    
    def __init__(self, parent: tk.Widget, on_change_callback: Callable[[], None]):
        self.parent = parent
        self.on_change = on_change_callback
        
        # Data management
        self.original_data: Optional[pd.DataFrame] = None
        self.modified_data: Optional[pd.DataFrame] = None
        self.excluded_players: Set[str] = set()
        self.projection_changes: Dict[str, float] = {}
        
        # UI state
        self.sort_column = None
        self.sort_reverse = False
        self.position_filter = "All"
        self.search_text = ""
        self.item_to_player: Dict[str, str] = {}  # Maps tree item IDs to player names
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the player pool UI"""
        # Main frame
        main_frame = ttk.LabelFrame(self.parent, text="Player Pool Preview", padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control frame
        self._create_control_frame(main_frame)
        
        # Position filters from config (will be set externally)
        self.position_filters: Dict[str, float] = {}
        self.show_excluded = True
        
        # Data table frame
        self._create_table_frame(main_frame)
        
        # Status frame
        self._create_status_frame(main_frame)
        
        # Initially show empty state
        self._show_empty_state()
    
    def _create_control_frame(self, parent):
        """Create control frame with filters and actions"""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Top row - filters
        filter_frame = ttk.Frame(control_frame)
        filter_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Position filter
        ttk.Label(filter_frame, text="Position:").pack(side=tk.LEFT)
        
        self.position_var = tk.StringVar(value="All")
        self.position_combo = ttk.Combobox(
            filter_frame,
            textvariable=self.position_var,
            values=["All", "QB", "RB", "WR", "TE", "DST"],
            state="readonly",
            width=8
        )
        self.position_combo.pack(side=tk.LEFT, padx=(5, 10))
        self.position_combo.bind('<<ComboboxSelected>>', self._on_filter_change)
        
        # Search
        ttk.Label(filter_frame, text="Search:").pack(side=tk.LEFT)
        
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(filter_frame, textvariable=self.search_var, width=20)
        search_entry.pack(side=tk.LEFT, padx=(5, 10))
        self.search_var.trace('w', self._on_search_change)
        
        # Clear filters button
        clear_btn = ttk.Button(filter_frame, text="Clear Filters", command=self._clear_filters)
        clear_btn.pack(side=tk.LEFT, padx=(5, 0))
        
        # Show/Hide excluded players toggle
        self.show_excluded_var = tk.BooleanVar(value=True)
        show_excluded_cb = ttk.Checkbutton(
            filter_frame, 
            text="Show Excluded", 
            variable=self.show_excluded_var,
            command=self._on_show_excluded_change
        )
        show_excluded_cb.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Bottom row - actions
        action_frame = ttk.Frame(control_frame)
        action_frame.pack(fill=tk.X)
        
        # Action buttons
        ttk.Button(action_frame, text="Exclude Selected", command=self._exclude_selected).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(action_frame, text="Include Selected", command=self._include_selected).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(action_frame, text="Reset All Changes", command=self._reset_all_changes).pack(side=tk.LEFT, padx=(0, 10))
        
        # Export button
        ttk.Button(action_frame, text="Export Modified Pool...", command=self._export_modified_pool).pack(side=tk.RIGHT)
    
    def _create_table_frame(self, parent):
        """Create the data table frame"""
        table_frame = ttk.Frame(parent)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create treeview with scrollbars
        self.tree = ttk.Treeview(table_frame, selectmode='extended')
        
        # Vertical scrollbar
        v_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=v_scrollbar.set)
        
        # Horizontal scrollbar
        h_scrollbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.configure(xscrollcommand=h_scrollbar.set)
        
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Bind events
        self.tree.bind('<Double-1>', self._on_double_click)
        self.tree.bind('<Button-3>', self._on_right_click)  # Right-click context menu
        self.tree.bind('<Button-1>', self._on_header_click)  # Header click for sorting
        
        # Create context menu
        self._create_context_menu()
    
    def _create_status_frame(self, parent):
        """Create status frame"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X)
        
        self.status_label = ttk.Label(status_frame, text="No player pool loaded")
        self.status_label.pack(side=tk.LEFT)
        
        # Position counts frame
        self.position_counts_frame = ttk.Frame(status_frame)
        self.position_counts_frame.pack(side=tk.RIGHT)
    
    def _create_context_menu(self):
        """Create right-click context menu"""
        self.context_menu = tk.Menu(self.tree, tearoff=0)
        self.context_menu.add_command(label="Edit Projection", command=self._edit_projection)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Exclude Player", command=self._exclude_selected)
        self.context_menu.add_command(label="Include Player", command=self._include_selected)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Reset Player Changes", command=self._reset_player_changes)
    
    def _show_empty_state(self):
        """Show empty state when no data is loaded"""
        # Clear tree
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Configure tree for empty state
        self.tree['columns'] = ()
        self.tree['show'] = 'tree'
        
        # Add empty message
        self.tree.insert('', tk.END, text="No player pool loaded. Select a CSV file to preview players.")
        
        # Update status
        self.status_label.config(text="No player pool loaded")
        
        # Clear position counts
        for widget in self.position_counts_frame.winfo_children():
            widget.destroy()
    
    def load_player_pool(self, csv_path: str, column_mapping: Dict[str, str]):
        """Load player pool from CSV file"""
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            
            # Validate required columns exist
            required_cols = ['player', 'position', 'salary', 'points']
            mapped_cols = {
                'player': 'player',  # Assume player column exists
                'position': column_mapping.get('position_column', 'pos'),
                'salary': column_mapping.get('salary_column', 'salary'),
                'points': column_mapping.get('points_column', 'proj')
            }
            
            missing_cols = []
            for logical_name, actual_col in mapped_cols.items():
                if actual_col not in df.columns:
                    missing_cols.append(f"{logical_name} ({actual_col})")
            
            if missing_cols:
                messagebox.showerror("Error", f"Missing required columns: {', '.join(missing_cols)}")
                return False
            
            # Store original data
            self.original_data = df.copy()
            self.modified_data = df.copy()
            
            # Reset modifications
            self.excluded_players.clear()
            self.projection_changes.clear()
            
            # Store column mapping
            self.column_mapping = mapped_cols
            
            # Set up tree columns
            self._setup_tree_columns()
            
            # Load data into tree
            self._refresh_tree_data()
            
            # Update status
            self._update_status()
            
            # Notify parent of change
            self.on_change()
            
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load player pool: {str(e)}")
            return False
    
    def _setup_tree_columns(self):
        """Set up tree columns based on data"""
        if self.original_data is None:
            return
        
        # Define columns to show
        display_columns = ['Status', 'Player', 'Team', 'Position', 'Salary', 'Projection']
        
        self.tree['columns'] = display_columns
        self.tree['show'] = 'headings'
        
        # Configure columns
        column_widths = {
            'Status': 60,
            'Player': 150,
            'Team': 60,
            'Position': 70,
            'Salary': 80,
            'Projection': 80
        }
        
        for col in display_columns:
            self.tree.heading(col, text=col, command=lambda c=col: self._sort_by_column(c))
            self.tree.column(col, width=column_widths.get(col, 100), minwidth=50)
    
    def _refresh_tree_data(self):
        """Refresh tree data with current filters and modifications"""
        if self.modified_data is None:
            return
        
        # Clear existing items and mapping
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.item_to_player.clear()
        
        # Apply filters
        filtered_data = self._apply_filters(self.modified_data)
        
        # Sort data if needed
        if self.sort_column and self.sort_column in ['Player', 'Team', 'Position', 'Salary', 'Projection']:
            sort_col = self._get_data_column(self.sort_column)
            if sort_col in filtered_data.columns:
                filtered_data = filtered_data.sort_values(sort_col, ascending=not self.sort_reverse)
        
        # Add data to tree
        for idx, row in filtered_data.iterrows():
            player_name = row.get('player', '')
            
            # Determine status
            status = "✓"  # Included
            if player_name in self.excluded_players:
                status = "✗"  # Excluded
            elif player_name in self.projection_changes:
                status = "✎"  # Modified
            
            # Get display values
            values = [
                status,
                player_name,
                row.get('team', ''),
                row.get(self.column_mapping['position'], ''),
                f"${row.get(self.column_mapping['salary'], 0):,}",
                f"{row.get(self.column_mapping['points'], 0):.1f}"
            ]
            
            # Insert item with tags for styling
            tags = []
            if player_name in self.excluded_players:
                tags.append('excluded')
            elif player_name in self.projection_changes:
                tags.append('modified')
            
            item_id = self.tree.insert('', tk.END, values=values, tags=tags)
            
            # Store player name in mapping dictionary
            self.item_to_player[item_id] = player_name
        
        # Configure tags for styling
        self.tree.tag_configure('excluded', background='#ffcccc')  # Light red
        self.tree.tag_configure('modified', background='#ffffcc')  # Light yellow
    
    def _apply_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply current filters to data"""
        filtered = data.copy()
        
        # Position filter
        if self.position_filter != "All":
            pos_col = self.column_mapping['position']
            filtered = filtered[filtered[pos_col] == self.position_filter]
        
        # Search filter
        if self.search_text:
            search_lower = self.search_text.lower()
            player_matches = filtered['player'].str.lower().str.contains(search_lower, na=False)
            team_matches = filtered.get('team', pd.Series()).str.lower().str.contains(search_lower, na=False)
            filtered = filtered[player_matches | team_matches]
        
        # Show/hide excluded players filter
        if not self.show_excluded_var.get():
            filtered = filtered[~filtered['player'].isin(self.excluded_players)]
        
        return filtered
    
    def _get_data_column(self, display_column: str) -> str:
        """Map display column to data column"""
        mapping = {
            'Player': 'player',
            'Team': 'team',
            'Position': self.column_mapping['position'],
            'Salary': self.column_mapping['salary'],
            'Projection': self.column_mapping['points']
        }
        return mapping.get(display_column, display_column.lower())
    
    def _sort_by_column(self, column: str):
        """Sort data by column"""
        if self.sort_column == column:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_column = column
            self.sort_reverse = False
        
        self._refresh_tree_data()
    
    def _on_filter_change(self, event=None):
        """Handle filter change"""
        self.position_filter = self.position_var.get()
        self._refresh_tree_data()
        self._update_status()
    
    def _on_search_change(self, *args):
        """Handle search text change"""
        self.search_text = self.search_var.get()
        self._refresh_tree_data()
        self._update_status()
    
    def _clear_filters(self):
        """Clear all filters"""
        self.position_var.set("All")
        self.search_var.set("")
        self.position_filter = "All"
        self.search_text = ""
        self._refresh_tree_data()
        self._update_status()
    
    def _on_show_excluded_change(self):
        """Handle show/hide excluded players toggle"""
        self._refresh_tree_data()
        self._update_status()
    
    def _on_double_click(self, event):
        """Handle double-click on tree item"""
        item = self.tree.selection()[0] if self.tree.selection() else None
        if item:
            # Check if clicked on projection column
            column = self.tree.identify_column(event.x)
            if column == '#6':  # Projection column (0-indexed, #6 is 6th column)
                self._edit_projection()
    
    def _on_right_click(self, event):
        """Handle right-click on tree item"""
        item = self.tree.identify_row(event.y)
        if item:
            self.tree.selection_set(item)
            self.context_menu.post(event.x_root, event.y_root)
    
    def _on_header_click(self, event):
        """Handle header click for sorting"""
        # This is handled by the heading command, but we could add additional logic here
        pass
    
    def _edit_projection(self):
        """Edit projection for selected player"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a player to edit.")
            return
        
        item = selection[0]
        player_name = self.item_to_player.get(item)
        if not player_name:
            messagebox.showerror("Error", "Could not identify selected player.")
            return
            
        current_projection = self.tree.item(item)['values'][5]  # Projection column
        
        # Remove formatting
        current_value = current_projection.replace('$', '').replace(',', '')
        
        # Create edit dialog
        dialog = ProjectionEditDialog(self.tree, player_name, current_value)
        new_value = dialog.result
        
        if new_value is not None:
            # Update projection changes
            self.projection_changes[player_name] = new_value
            
            # Update modified data
            player_mask = self.modified_data['player'] == player_name
            self.modified_data.loc[player_mask, self.column_mapping['points']] = new_value
            
            # Refresh display
            self._refresh_tree_data()
            self._update_status()
            
            # Notify parent of change
            self.on_change()
    
    def _exclude_selected(self):
        """Exclude selected players"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select players to exclude.")
            return
        
        for item in selection:
            player_name = self.item_to_player.get(item)
            if player_name:
                self.excluded_players.add(player_name)
        
        self._refresh_tree_data()
        self._update_status()
        self.on_change()
    
    def _include_selected(self):
        """Include selected players"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select players to include.")
            return
        
        for item in selection:
            player_name = self.item_to_player.get(item)
            if player_name:
                self.excluded_players.discard(player_name)
        
        self._refresh_tree_data()
        self._update_status()
        self.on_change()
    
    def _reset_player_changes(self):
        """Reset changes for selected player"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a player to reset.")
            return
        
        for item in selection:
            player_name = self.item_to_player.get(item)
            if not player_name:
                continue
            
            # Remove from excluded players
            self.excluded_players.discard(player_name)
            
            # Remove projection changes
            if player_name in self.projection_changes:
                del self.projection_changes[player_name]
                
                # Restore original projection
                original_value = self.original_data[self.original_data['player'] == player_name][self.column_mapping['points']].iloc[0]
                player_mask = self.modified_data['player'] == player_name
                self.modified_data.loc[player_mask, self.column_mapping['points']] = original_value
        
        self._refresh_tree_data()
        self._update_status()
        self.on_change()
    
    def _reset_all_changes(self):
        """Reset all changes"""
        if not self.excluded_players and not self.projection_changes:
            messagebox.showinfo("Info", "No changes to reset.")
            return
        
        if messagebox.askyesno("Confirm Reset", "Reset all player exclusions and projection changes?"):
            self.excluded_players.clear()
            self.projection_changes.clear()
            
            # Restore original data
            if self.original_data is not None:
                self.modified_data = self.original_data.copy()
            
            self._refresh_tree_data()
            self._update_status()
            self.on_change()
    
    def _export_modified_pool(self):
        """Export modified player pool to CSV"""
        if self.modified_data is None:
            messagebox.showwarning("Warning", "No player pool loaded.")
            return
        
        # Get export data (exclude excluded players)
        export_data = self.modified_data[~self.modified_data['player'].isin(self.excluded_players)].copy()
        
        if export_data.empty:
            messagebox.showwarning("Warning", "No players to export (all players excluded).")
            return
        
        # Get save filename
        filename = filedialog.asksaveasfilename(
            title="Export Modified Player Pool",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                export_data.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Modified player pool exported to {Path(filename).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export player pool: {str(e)}")
    
    def _update_status(self):
        """Update status display"""
        if self.modified_data is None:
            self.status_label.config(text="No player pool loaded")
            return
        
        # Apply current filters to get visible count
        filtered_data = self._apply_filters(self.modified_data)
        total_players = len(self.modified_data)
        visible_players = len(filtered_data)
        excluded_count = len(self.excluded_players)
        included_count = total_players - excluded_count
        modified_count = len(self.projection_changes)
        
        status_text = f"Total: {total_players} | Included: {included_count} | Excluded: {excluded_count}"
        
        if visible_players != total_players:
            status_text = f"Showing: {visible_players} | " + status_text
        
        if modified_count > 0:
            status_text += f" | Modified: {modified_count}"
        
        self.status_label.config(text=status_text)
        
        # Update position counts
        self._update_position_counts(filtered_data)
    
    def _update_position_counts(self, data: pd.DataFrame):
        """Update position counts display"""
        # Clear existing counts
        for widget in self.position_counts_frame.winfo_children():
            widget.destroy()
        
        if data.empty:
            return
        
        # Count by position (excluding excluded players)
        available_data = data[~data['player'].isin(self.excluded_players)]
        pos_col = self.column_mapping['position']
        
        if pos_col in available_data.columns:
            pos_counts = available_data[pos_col].value_counts().sort_index()
            
            for i, (pos, count) in enumerate(pos_counts.items()):
                if i > 0:
                    ttk.Label(self.position_counts_frame, text="|").pack(side=tk.LEFT, padx=2)
                
                ttk.Label(self.position_counts_frame, text=f"{pos}: {count}").pack(side=tk.LEFT, padx=2)
    
    def get_modified_data(self) -> Optional[pd.DataFrame]:
        """Get the modified player pool data (excluding excluded players)"""
        if self.modified_data is None:
            return None
        
        # Return data excluding excluded players
        return self.modified_data[~self.modified_data['player'].isin(self.excluded_players)].copy()
    
    def has_modifications(self) -> bool:
        """Check if there are any modifications"""
        return bool(self.excluded_players or self.projection_changes)
    
    def set_position_filters(self, position_filters: Dict[str, float]):
        """Set position filters from configuration and apply them"""
        self.position_filters = position_filters.copy()
        self._apply_position_filters()
    
    def _apply_position_filters(self):
        """Apply position filters by excluding players below thresholds"""
        if not self.position_filters or self.modified_data is None:
            return
        
        pos_col = self.column_mapping.get('position')
        points_col = self.column_mapping.get('points')
        
        if not pos_col or not points_col:
            return
        
        # First, remove any players that were previously excluded by position filters
        # We'll track this by storing which players were excluded by filters
        if not hasattr(self, 'filter_excluded_players'):
            self.filter_excluded_players = set()
        
        # Remove previously filter-excluded players from the main excluded set
        self.excluded_players -= self.filter_excluded_players
        self.filter_excluded_players.clear()
        
        # Apply current position filters
        for position, min_points in self.position_filters.items():
            if min_points > 0:  # Only apply if threshold is set
                # Find players in this position below the threshold
                position_mask = self.modified_data[pos_col] == position
                points_mask = self.modified_data[points_col] < min_points
                below_threshold = self.modified_data[position_mask & points_mask]['player'].tolist()
                
                # Add to both sets
                self.filter_excluded_players.update(below_threshold)
                self.excluded_players.update(below_threshold)
        
        # Refresh display
        self._refresh_tree_data()
        self._update_status()
        self.on_change()


class ProjectionEditDialog:
    """Dialog for editing player projections"""
    
    def __init__(self, parent, player_name: str, current_value: str):
        self.result = None
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(f"Edit Projection - {player_name}")
        self.dialog.geometry("300x150")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (300 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (150 // 2)
        self.dialog.geometry(f"300x150+{x}+{y}")
        
        # Create content
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text=f"Player: {player_name}", font=('TkDefaultFont', 10, 'bold')).pack(pady=(0, 10))
        
        ttk.Label(main_frame, text="New Projection:").pack(anchor=tk.W)
        
        self.value_var = tk.StringVar(value=current_value)
        entry = ttk.Entry(main_frame, textvariable=self.value_var, width=20)
        entry.pack(pady=5, fill=tk.X)
        entry.select_range(0, tk.END)
        entry.focus()
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="Cancel", command=self._cancel).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="OK", command=self._ok).pack(side=tk.RIGHT)
        
        # Bind events
        entry.bind('<Return>', lambda e: self._ok())
        entry.bind('<Escape>', lambda e: self._cancel())
        
        # Wait for dialog to close
        self.dialog.wait_window()
    
    def _ok(self):
        """Handle OK button"""
        try:
            value = float(self.value_var.get())
            self.result = value
            self.dialog.destroy()
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number.")
    
    def _cancel(self):
        """Handle Cancel button"""
        self.result = None
        self.dialog.destroy()
