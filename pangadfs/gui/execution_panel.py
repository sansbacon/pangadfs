# pangadfs/gui/execution_panel.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import tkinter as tk
from tkinter import ttk, scrolledtext
from typing import Callable
import logging


class ExecutionPanel:
    """Panel for running optimization and displaying progress"""
    
    def __init__(self, parent: tk.Widget, run_callback: Callable[[], None], stop_callback: Callable[[], None]):
        self.parent = parent
        self.run_callback = run_callback
        self.stop_callback = stop_callback
        self.is_running = False
        
        self._setup_ui()
        self._setup_logging()
    
    def _setup_ui(self):
        """Set up the execution UI"""
        # Main frame
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control section
        control_frame = ttk.LabelFrame(main_frame, text="Optimization Control", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Run button
        self.run_button = ttk.Button(
            control_frame, 
            text="Start Optimization", 
            command=self.run_callback,
            style="Accent.TButton"
        )
        self.run_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Stop button
        self.stop_button = ttk.Button(
            control_frame, 
            text="Stop Optimization", 
            command=self.stop_callback,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            control_frame, 
            variable=self.progress_var, 
            mode='indeterminate',
            length=200
        )
        self.progress_bar.pack(side=tk.LEFT, padx=(10, 0))
        
        # Status label
        self.status_var = tk.StringVar(value="Ready to run optimization")
        self.status_label = ttk.Label(control_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Log section
        log_frame = ttk.LabelFrame(main_frame, text="Optimization Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # Log text area with scrollbar
        self.log_text = scrolledtext.ScrolledText(
            log_frame, 
            wrap=tk.WORD, 
            height=20,
            font=('Consolas', 9)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Log control buttons
        log_control_frame = ttk.Frame(log_frame)
        log_control_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(log_control_frame, text="Clear Log", command=self._clear_log).pack(side=tk.LEFT)
        ttk.Button(log_control_frame, text="Save Log...", command=self._save_log).pack(side=tk.LEFT, padx=(10, 0))
        
        # Auto-scroll checkbox
        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            log_control_frame, 
            text="Auto-scroll", 
            variable=self.auto_scroll_var
        ).pack(side=tk.RIGHT)
        
        # Information section
        info_frame = ttk.LabelFrame(main_frame, text="Quick Start Guide", padding=10)
        info_frame.pack(fill=tk.X, pady=(10, 0))
        
        info_text = """1. Configure your settings in the Configuration tab
2. Select a player pool CSV file
3. Choose single lineup or multilineup optimization
4. Click 'Start Optimization' to begin
5. Monitor progress in the log below
6. View results in the Results tab when complete"""
        
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT).pack(anchor=tk.W)
    
    def _setup_logging(self):
        """Set up logging to capture optimization output"""
        # Create a custom log handler that writes to our text widget
        self.log_handler = TextWidgetHandler(self.log_text, self.auto_scroll_var)
        self.log_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                    datefmt='%H:%M:%S')
        self.log_handler.setFormatter(formatter)
        
        # Add handler to root logger
        logging.getLogger().addHandler(self.log_handler)
        
        # Initial log message
        self._log_message("Optimization panel ready")
    
    def set_running(self, running: bool):
        """Update UI state for running/stopped"""
        self.is_running = running
        
        if running:
            self.run_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.progress_bar.start()
            self.status_var.set("Optimization running...")
            self._log_message("Starting optimization...")
        else:
            self.run_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.progress_bar.stop()
            self.status_var.set("Ready to run optimization")
            if running is False:  # Explicitly stopped, not just initialized
                self._log_message("Optimization stopped")
    
    @staticmethod
    def _log_message(message: str, level: str = "INFO"):
        """Add a message to the log"""
        # Create a log record and let the handler format it
        logger = logging.getLogger(__name__)
        if level == "ERROR":
            logger.error(message)
        elif level == "WARNING":
            logger.warning(message)
        else:
            logger.info(message)
    
    def _clear_log(self):
        """Clear the log text"""
        self.log_text.delete(1.0, tk.END)
        self._log_message("Log cleared")
    
    def _save_log(self):
        """Save log to file"""
        from tkinter import filedialog
        
        filename = filedialog.asksaveasfilename(
            title="Save Log",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                self._log_message(f"Log saved to {filename}")
            except Exception as e:
                self._log_message(f"Failed to save log: {str(e)}", "ERROR")


class TextWidgetHandler(logging.Handler):
    """Custom logging handler that writes to a Tkinter Text widget"""
    
    def __init__(self, text_widget: tk.Text, auto_scroll_var: tk.BooleanVar):
        super().__init__()
        self.text_widget = text_widget
        self.auto_scroll_var = auto_scroll_var
    
    def emit(self, record):
        """Emit a log record to the text widget"""
        try:
            msg = self.format(record)
            
            # Schedule the text insertion in the main thread
            self.text_widget.after(0, self._insert_text, msg)
            
        except Exception:
            self.handleError(record)
    
    def _insert_text(self, msg: str):
        """Insert text into the widget (must be called from main thread)"""
        try:
            # Insert the message
            self.text_widget.insert(tk.END, msg + '\n')
            
            # Auto-scroll if enabled
            if self.auto_scroll_var.get():
                self.text_widget.see(tk.END)
            
            # Limit the number of lines to prevent memory issues
            lines = int(self.text_widget.index('end-1c').split('.')[0])
            if lines > 1000:
                # Remove the first 100 lines
                self.text_widget.delete(1.0, '101.0')
                
        except tk.TclError:
            # Widget might be destroyed
            pass
