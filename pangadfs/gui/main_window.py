# pangadfs/gui/main_window.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
from pathlib import Path
from typing import Dict, Any, Optional

from .config_panel import ConfigPanel
from .execution_panel import ExecutionPanel
from .results_panel import ResultsPanel
from .export_panel import ExportPanel
from .utils.config_manager import ConfigManager
from .theme_manager import ThemeManager


class MainWindow:
    """Main application window for pangadfs GUI"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PangaDFS - Fantasy Sports Lineup Optimizer")
        self.root.geometry("1400x900")
        self.root.minsize(1000, 700)
        
        # Initialize theme manager first
        self.theme_manager = ThemeManager(self.root, 'auto')
        
        # Application state
        self.config_manager = ConfigManager()
        self.current_results = None
        self.optimization_thread = None
        self.result_queue = queue.Queue()
        
        # Initialize UI
        self._setup_menu()
        self._setup_main_interface()
        self._setup_status_bar()
        
        # Load default configuration
        self._load_default_config()
        
        # Set up periodic checking for results
        self.root.after(100, self._check_result_queue)
    
    def _setup_menu(self):
        """Set up the application menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Configuration", command=self._new_config)
        file_menu.add_command(label="Open Configuration...", command=self._open_config)
        file_menu.add_command(label="Save Configuration", command=self._save_config)
        file_menu.add_command(label="Save Configuration As...", command=self._save_config_as)
        file_menu.add_separator()
        file_menu.add_command(label="Open Player Pool...", command=self._open_player_pool)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Validate Configuration", command=self._validate_config)
        tools_menu.add_command(label="Preview Player Pool", command=self._preview_player_pool)
        tools_menu.add_separator()
        
        # Theme submenu
        theme_menu = tk.Menu(tools_menu, tearoff=0)
        tools_menu.add_cascade(label="Theme", menu=theme_menu)
        theme_menu.add_command(label="Light Theme", command=lambda: self._switch_theme('light'))
        theme_menu.add_command(label="Dark Theme", command=lambda: self._switch_theme('dark'))
        theme_menu.add_separator()
        theme_menu.add_command(label="Blue Theme", command=lambda: self._switch_theme('blue'))
        theme_menu.add_command(label="Green Theme", command=lambda: self._switch_theme('green'))
        theme_menu.add_command(label="Purple Theme", command=lambda: self._switch_theme('purple'))
        theme_menu.add_command(label="Warm Theme", command=lambda: self._switch_theme('warm'))
        theme_menu.add_separator()
        theme_menu.add_command(label="Auto (System)", command=lambda: self._switch_theme('auto'))
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
        help_menu.add_command(label="Documentation", command=self._show_documentation)
    
    def _setup_main_interface(self):
        """Set up the main tabbed interface"""
        # Create main notebook (tabbed interface) with no padding to maximize space
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Configuration tab
        config_frame = ttk.Frame(self.notebook)
        self.notebook.add(config_frame, text="⚙️ Configuration")
        self.config_panel = ConfigPanel(config_frame, self._on_config_changed)
        
        # Execution tab
        execution_frame = ttk.Frame(self.notebook)
        self.notebook.add(execution_frame, text="▶️ Run Optimization")
        self.execution_panel = ExecutionPanel(execution_frame, self._run_optimization, self._stop_optimization)
        
        # Results tab
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="📊 Results")
        self.results_panel = ResultsPanel(results_frame)
        
        # Export tab
        export_frame = ttk.Frame(self.notebook)
        self.notebook.add(export_frame, text="💾 Export")
        self.export_panel = ExportPanel(export_frame, self._get_current_results)
        
        # Initially disable results and export tabs
        self.notebook.tab(2, state="disabled")  # Results tab
        self.notebook.tab(3, state="disabled")  # Export tab
    
    def _setup_status_bar(self):
        """Set up the status bar at the bottom"""
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = ttk.Label(self.status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=5, pady=2)
        
        # Progress bar (initially hidden)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.status_frame, 
            variable=self.progress_var, 
            mode='indeterminate'
        )
    
    def _load_default_config(self):
        """Load default configuration"""
        try:
            default_config = self.config_manager.get_default_config()
            self.config_panel.load_config(default_config)
            self._update_status("Default configuration loaded")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load default configuration: {str(e)}")
    
    def _on_config_changed(self):
        """Called when configuration is modified"""
        self._update_status("Configuration modified")
    
    def _new_config(self):
        """Create a new configuration"""
        if messagebox.askyesno("New Configuration", "This will reset all settings. Continue?"):
            self._load_default_config()
    
    def _open_config(self):
        """Open a configuration file"""
        filename = filedialog.askopenfilename(
            title="Open Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                config = self.config_manager.load_config(filename)
                self.config_panel.load_config(config)
                self._update_status(f"Configuration loaded from {Path(filename).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")
    
    def _save_config(self):
        """Save current configuration"""
        try:
            config = self.config_panel.get_config()
            filename = self.config_manager.save_config(config)
            self._update_status(f"Configuration saved to {Path(filename).name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")
    
    def _save_config_as(self):
        """Save configuration with a new filename"""
        filename = filedialog.asksaveasfilename(
            title="Save Configuration As",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                config = self.config_panel.get_config()
                self.config_manager.save_config(config, filename)
                self._update_status(f"Configuration saved to {Path(filename).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")
    
    def _open_player_pool(self):
        """Open a player pool CSV file"""
        filename = filedialog.askopenfilename(
            title="Open Player Pool",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.config_panel.set_player_pool_file(filename)
            self._update_status(f"Player pool set to {Path(filename).name}")
    
    def _validate_config(self):
        """Validate current configuration"""
        try:
            config = self.config_panel.get_config()
            errors = self.config_manager.validate_config(config)
            if errors:
                error_msg = "Configuration errors found:\n\n" + "\n".join(errors)
                messagebox.showerror("Configuration Errors", error_msg)
            else:
                messagebox.showinfo("Validation", "Configuration is valid!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to validate configuration: {str(e)}")
    
    def _preview_player_pool(self):
        """Preview the selected player pool file"""
        try:
            config = self.config_panel.get_config()
            csv_path = config.get('ga_settings', {}).get('csvpth')
            if not csv_path:
                messagebox.showwarning("Warning", "No player pool file selected")
                return
            
            # This would open a preview window - simplified for now
            messagebox.showinfo("Preview", f"Player pool preview for: {Path(csv_path).name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to preview player pool: {str(e)}")
    
    def _run_optimization(self):
        """Start the optimization process"""
        try:
            # Validate configuration first
            config = self.config_panel.get_config()
            errors = self.config_manager.validate_config(config)
            if errors:
                error_msg = "Please fix configuration errors before running:\n\n" + "\n".join(errors)
                messagebox.showerror("Configuration Errors", error_msg)
                return
            
            # Start optimization in background thread
            self.optimization_thread = threading.Thread(
                target=self._optimization_worker,
                args=(config,),
                daemon=True
            )
            self.optimization_thread.start()
            
            # Update UI state
            self._set_optimization_running(True)
            self._update_status("Optimization running...")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start optimization: {str(e)}")
    
    def _stop_optimization(self):
        """Stop the optimization process"""
        # Note: This is a simplified implementation
        # In a real implementation, you'd need to properly signal the GA to stop
        self._set_optimization_running(False)
        self._update_status("Optimization stopped")
    
    def _optimization_worker(self, config: Dict[str, Any]):
        """Worker function that runs optimization in background thread"""
        try:
            # Import here to avoid circular imports
            from pangadfs.ga import GeneticAlgorithm
            from stevedore.driver import DriverManager
            from stevedore.named import NamedExtensionManager
            import tempfile
            import os
            
            # Check if we have modified player data
            modified_data = self.config_panel.get_modified_player_data()
            temp_csv_path = None
            
            if modified_data is not None:
                # Create temporary CSV file with modified data
                temp_fd, temp_csv_path = tempfile.mkstemp(suffix='.csv')
                try:
                    with os.fdopen(temp_fd, 'w', newline='', encoding='utf-8'):
                        pass  # Just close the file descriptor
                    modified_data.to_csv(temp_csv_path, index=False)
                    
                    # Update config to use temporary CSV
                    config = config.copy()  # Don't modify original
                    config['ga_settings'] = config['ga_settings'].copy()
                    config['ga_settings']['csvpth'] = Path(temp_csv_path)
                except Exception as e:
                    if temp_csv_path and os.path.exists(temp_csv_path):
                        os.unlink(temp_csv_path)
                    raise e
            
            try:
                # Set up driver managers based on config
                dmgrs = {}
                emgrs = {}
                
                # Determine which optimizer to use based on config panel selection
                optimizer_name = self.config_panel.get_optimizer_name()
                
                for ns in GeneticAlgorithm.PLUGIN_NAMESPACES:
                    pns = f'pangadfs.{ns}'
                    if ns == 'validate':
                        emgrs['validate'] = NamedExtensionManager(
                            namespace=pns, 
                            names=['validate_salary', 'validate_duplicates', 'validate_positions'], 
                            invoke_on_load=True, 
                            name_order=True)
                    elif ns == 'optimize':
                        dmgrs[ns] = DriverManager(
                            namespace=pns, 
                            name=optimizer_name,
                            invoke_on_load=True)
                    else:
                        dmgrs[ns] = DriverManager(
                            namespace=pns, 
                            name=f'{ns}_default', 
                            invoke_on_load=True)
                
                # Create enhanced config with proper parameter mapping
                enhanced_config = self._prepare_ga_config(config)
                
                # Create GA and run optimization
                ga = GeneticAlgorithm(ctx=enhanced_config, driver_managers=dmgrs, extension_managers=emgrs)
                results = ga.optimize()
                
                # Send results back to main thread
                self.result_queue.put(('success', results))
                
            finally:
                # Clean up temporary file
                if temp_csv_path and os.path.exists(temp_csv_path):
                    try:
                        os.unlink(temp_csv_path)
                    except:
                        pass  # Ignore cleanup errors
            
        except Exception as e:
            # Send error back to main thread
            self.result_queue.put(('error', str(e)))
    
    def _check_result_queue(self):
        """Check for results from optimization thread"""
        try:
            while True:
                result_type, result_data = self.result_queue.get_nowait()
                
                if result_type == 'success':
                    self._on_optimization_complete(result_data)
                elif result_type == 'error':
                    self._on_optimization_error(result_data)
                
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self._check_result_queue)
    
    def _on_optimization_complete(self, results: Dict[str, Any]):
        """Handle successful optimization completion"""
        self.current_results = results
        self._set_optimization_running(False)
        
        # Enable results and export tabs
        self.notebook.tab(2, state="normal")  # Results tab
        self.notebook.tab(3, state="normal")  # Export tab
        
        # Update results panel
        self.results_panel.display_results(results)
        
        # Configure export panel for single vs multilineup mode
        self.export_panel.configure_for_results(results)
        
        # Switch to results tab
        self.notebook.select(2)
        
        # Update status
        if 'lineups' in results and results['lineups']:
            num_lineups = len(results['lineups'])
        else:
            num_lineups = 1
        best_score = results.get('best_score', 0)
        self._update_status(f"Optimization complete: {num_lineups} lineup(s), best score: {best_score:.2f}")
    
    def _on_optimization_error(self, error_msg: str):
        """Handle optimization error"""
        self._set_optimization_running(False)
        self._update_status("Optimization failed")
        messagebox.showerror("Optimization Error", f"Optimization failed: {error_msg}")
    
    def _set_optimization_running(self, running: bool):
        """Update UI state for optimization running/stopped"""
        if running:
            self.progress_bar.pack(side=tk.RIGHT, padx=5, pady=2)
            self.progress_bar.start()
        else:
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
        
        # Update execution panel
        self.execution_panel.set_running(running)
    
    @staticmethod
    def _prepare_ga_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare configuration for GA with proper parameter mapping"""
        ga_settings = config.get('ga_settings', {})
        site_settings = config.get('site_settings', {})
        
        # Create column mapping for GA
        column_mapping = {
            'points': ga_settings.get('points_column', 'proj'),
            'position': ga_settings.get('position_column', 'pos'),
            'salary': ga_settings.get('salary_column', 'salary')
        }
        
        # Enhanced config with all required parameters
        enhanced_config = {
            'ga_settings': {
                **ga_settings,
                'column_mapping': column_mapping,
                'posfilter': site_settings.get('posfilter', {}),
                'posmap': site_settings.get('posmap', {}),
                'flex_positions': site_settings.get('flex_positions', ('RB', 'WR', 'TE')),
                'salary_cap': site_settings.get('salary_cap', 50000),
                'lineup_size': site_settings.get('lineup_size', 9)
            },
            'site_settings': site_settings
        }
        
        return enhanced_config
    
    def _get_current_results(self) -> Optional[Dict[str, Any]]:
        """Get current optimization results"""
        return self.current_results
    
    def _update_status(self, message: str):
        """Update status bar message"""
        self.status_label.config(text=message)
    
    @staticmethod
    def _show_about():
        """Show about dialog"""
        about_text = """PangaDFS GUI v0.1.0

Fantasy Sports Lineup Optimizer

Built on the pangadfs framework by Eric Truett
Licensed under the MIT License"""
        messagebox.showinfo("About PangaDFS", about_text)
    
    @staticmethod
    def _show_documentation():
        """Show documentation"""
        messagebox.showinfo("Documentation", "Documentation is available in the docs/ folder and MULTILINEUP_README.md")
    
    def _switch_theme(self, theme_name: str):
        """Switch to a different theme"""
        if theme_name == 'auto':
            # Re-detect system theme
            self.theme_manager = ThemeManager(self.root, 'auto')
        else:
            self.theme_manager.switch_theme(theme_name)
        
        self._update_status(f"Switched to {theme_name} theme")
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()


def main():
    """Main entry point for GUI application"""
    app = MainWindow()
    app.run()


if __name__ == '__main__':
    main()
