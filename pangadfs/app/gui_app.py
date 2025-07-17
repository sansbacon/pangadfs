# pangadfs/app/gui_app.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

"""
GUI Application entry point for pangadfs
"""

import sys
import tkinter as tk
from tkinter import ttk, messagebox
import logging
from pathlib import Path

# Add the parent directory to the path so we can import pangadfs modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pangadfs.gui.main_window import MainWindow


def setup_logging():
    """Set up logging for the GUI application"""
    # Create logs directory if it doesn't exist
    log_dir = Path.home() / '.pangadfs' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_file = log_dir / 'pangadfs_gui.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce noise from some libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []
    
    try:
        import pandas
    except ImportError:
        missing_deps.append('pandas')
    
    try:
        import numpy
    except ImportError:
        missing_deps.append('numpy')
    
    try:
        import stevedore
    except ImportError:
        missing_deps.append('stevedore')
    
    # Optional dependencies
    optional_deps = []
    
    try:
        import openpyxl
    except ImportError:
        optional_deps.append('openpyxl (for Excel export)')
    
    if missing_deps:
        error_msg = f"Missing required dependencies: {', '.join(missing_deps)}\n\n"
        error_msg += "Please install them using:\n"
        error_msg += f"pip install {' '.join(missing_deps)}"
        
        # Show error in GUI if possible, otherwise print
        try:
            root = tk.Tk()
            root.withdraw()  # Hide the root window
            messagebox.showerror("Missing Dependencies", error_msg)
            root.destroy()
        except:
            print(f"ERROR: {error_msg}")
        
        return False
    
    if optional_deps:
        logging.info(f"Optional dependencies not found: {', '.join(optional_deps)}")
    
    return True


def main():
    """Main entry point for the GUI application"""
    try:
        # Set up logging
        setup_logging()
        
        logger = logging.getLogger(__name__)
        logger.info("Starting PangaDFS GUI Application")
        
        # Check dependencies
        if not check_dependencies():
            sys.exit(1)
        
        # Import pangadfs modules to verify they're working
        try:
            from pangadfs.ga import GeneticAlgorithm
            logger.info("PangaDFS modules loaded successfully")
        except ImportError as e:
            error_msg = f"Failed to import pangadfs modules: {e}\n\n"
            error_msg += "Please ensure pangadfs is properly installed."
            
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Import Error", error_msg)
            root.destroy()
            sys.exit(1)
        
        # Create and run the main application
        logger.info("Creating main window")
        app = MainWindow()
        
        logger.info("Starting GUI main loop")
        app.run()
        
        logger.info("GUI application closed")
        
    except KeyboardInterrupt:
        logging.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logging.error(error_msg, exc_info=True)
        
        # Try to show error in GUI
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Application Error", error_msg)
            root.destroy()
        except:
            print(f"FATAL ERROR: {error_msg}")
        
        sys.exit(1)


if __name__ == '__main__':
    main()
