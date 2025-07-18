#!/usr/bin/env python3
"""
Test script to demonstrate the PangaDFS GUI theme functionality
"""

import tkinter as tk
from tkinter import ttk
import sys
import os

# Add pangadfs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pangadfs'))

from pangadfs.gui.theme_manager import ThemeManager, create_primary_button, create_success_button, create_danger_button


def demo_themes():
    """Demonstrate the theme system"""
    
    # Create main window
    root = tk.Tk()
    root.title("PangaDFS Theme Demo")
    root.geometry("600x400")
    
    # Initialize theme manager
    theme_manager = ThemeManager(root, 'light')
    
    # Create demo content
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
    # Title
    title_label = ttk.Label(main_frame, text="PangaDFS Theme Demo", style='Heading.TLabel')
    title_label.pack(pady=(0, 20))
    
    # Theme selection
    theme_frame = ttk.LabelFrame(main_frame, text="Theme Selection", padding=10)
    theme_frame.pack(fill=tk.X, pady=(0, 20))
    
    def switch_theme(theme_name):
        theme_manager.switch_theme(theme_name)
        status_label.config(text=f"Current theme: {theme_manager.current_theme}")
    
    ttk.Button(theme_frame, text="Light Theme", command=lambda: switch_theme('light')).pack(side=tk.LEFT, padx=5)
    ttk.Button(theme_frame, text="Dark Theme", command=lambda: switch_theme('dark')).pack(side=tk.LEFT, padx=5)
    ttk.Button(theme_frame, text="Auto Theme", command=lambda: switch_theme('auto')).pack(side=tk.LEFT, padx=5)
    
    # Sample widgets
    widgets_frame = ttk.LabelFrame(main_frame, text="Sample Widgets", padding=10)
    widgets_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
    
    # Buttons
    button_frame = ttk.Frame(widgets_frame)
    button_frame.pack(fill=tk.X, pady=5)
    
    create_primary_button(button_frame, theme_manager, text="Primary Button").pack(side=tk.LEFT, padx=5)
    create_success_button(button_frame, theme_manager, text="Success Button").pack(side=tk.LEFT, padx=5)
    create_danger_button(button_frame, theme_manager, text="Danger Button").pack(side=tk.LEFT, padx=5)
    
    # Entry and combobox
    entry_frame = ttk.Frame(widgets_frame)
    entry_frame.pack(fill=tk.X, pady=5)
    
    ttk.Label(entry_frame, text="Sample Entry:").pack(side=tk.LEFT)
    ttk.Entry(entry_frame, width=20).pack(side=tk.LEFT, padx=5)
    
    ttk.Label(entry_frame, text="Sample Combobox:").pack(side=tk.LEFT, padx=(20, 0))
    ttk.Combobox(entry_frame, values=["Option 1", "Option 2", "Option 3"], width=15).pack(side=tk.LEFT, padx=5)
    
    # Treeview
    tree_frame = ttk.Frame(widgets_frame)
    tree_frame.pack(fill=tk.BOTH, expand=True, pady=5)
    
    tree = ttk.Treeview(tree_frame, columns=('col1', 'col2'), show='tree headings', height=6)
    tree.heading('#0', text='Player')
    tree.heading('col1', text='Position')
    tree.heading('col2', text='Salary')
    
    # Add sample data
    tree.insert('', 'end', text='Patrick Mahomes', values=('QB', '$8,500'))
    tree.insert('', 'end', text='Christian McCaffrey', values=('RB', '$9,200'))
    tree.insert('', 'end', text='Davante Adams', values=('WR', '$8,800'))
    tree.insert('', 'end', text='Travis Kelce', values=('TE', '$7,600'))
    
    tree.pack(fill=tk.BOTH, expand=True)
    
    # Status
    status_label = ttk.Label(main_frame, text=f"Current theme: {theme_manager.current_theme}")
    status_label.pack(pady=5)
    
    # Instructions
    instructions = ttk.Label(main_frame, 
                           text="Click the theme buttons above to see the visual changes in real-time!",
                           style='Small.TLabel')
    instructions.pack(pady=5)
    
    print("Theme demo window created. Try switching themes to see the visual changes!")
    root.mainloop()


if __name__ == "__main__":
    demo_themes()
