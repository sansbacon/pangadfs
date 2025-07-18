# pangadfs/gui/theme_manager.py
# -*- coding: utf-8 -*-
# Copyright (C) 2020 Eric Truett
# Licensed under the MIT License

import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, Optional
import sys


class ThemeManager:
    """Manages themes and styling for the PangaDFS GUI"""
    
    # Color palettes
    THEMES = {
        'light': {
            'bg_primary': '#ffffff',
            'bg_secondary': '#f8f9fa',
            'bg_tertiary': '#e9ecef',
            'bg_accent': '#e3f2fd',
            'fg_primary': '#212529',
            'fg_secondary': '#6c757d',
            'fg_accent': '#1976d2',
            'border': '#dee2e6',
            'border_focus': '#80bdff',
            'success': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545',
            'info': '#17a2b8',
            'button_bg': '#007bff',
            'button_fg': '#ffffff',
            'button_hover': '#0056b3',
            'button_active': '#004085',
            'entry_bg': '#ffffff',
            'entry_fg': '#495057',
            'selection_bg': '#007bff',
            'selection_fg': '#ffffff'
        },
        'dark': {
            'bg_primary': '#3a3a3a',
            'bg_secondary': '#4a4a4a',
            'bg_tertiary': '#5a5a5a',
            'bg_accent': '#2d4a6b',
            'fg_primary': '#f0f0f0',
            'fg_secondary': '#c0c0c0',
            'fg_accent': '#7bb3f0',
            'border': '#666666',
            'border_focus': '#7bb3f0',
            'success': '#5cbf60',
            'warning': '#ffb74d',
            'danger': '#f66',
            'info': '#42a5f5',
            'button_bg': '#2196f3',
            'button_fg': '#ffffff',
            'button_hover': '#1976d2',
            'button_active': '#1565c0',
            'entry_bg': '#505050',
            'entry_fg': '#f0f0f0',
            'selection_bg': '#2196f3',
            'selection_fg': '#ffffff'
        },
        'blue': {
            'bg_primary': '#f0f4f8',
            'bg_secondary': '#e2e8f0',
            'bg_tertiary': '#cbd5e0',
            'bg_accent': '#bee3f8',
            'fg_primary': '#1a202c',
            'fg_secondary': '#4a5568',
            'fg_accent': '#2b6cb0',
            'border': '#cbd5e0',
            'border_focus': '#4299e1',
            'success': '#38a169',
            'warning': '#d69e2e',
            'danger': '#e53e3e',
            'info': '#3182ce',
            'button_bg': '#4299e1',
            'button_fg': '#ffffff',
            'button_hover': '#3182ce',
            'button_active': '#2c5282',
            'entry_bg': '#ffffff',
            'entry_fg': '#2d3748',
            'selection_bg': '#4299e1',
            'selection_fg': '#ffffff'
        },
        'green': {
            'bg_primary': '#f0fff4',
            'bg_secondary': '#e6fffa',
            'bg_tertiary': '#c6f6d5',
            'bg_accent': '#9ae6b4',
            'fg_primary': '#1a202c',
            'fg_secondary': '#4a5568',
            'fg_accent': '#2f855a',
            'border': '#c6f6d5',
            'border_focus': '#48bb78',
            'success': '#38a169',
            'warning': '#d69e2e',
            'danger': '#e53e3e',
            'info': '#3182ce',
            'button_bg': '#48bb78',
            'button_fg': '#ffffff',
            'button_hover': '#38a169',
            'button_active': '#2f855a',
            'entry_bg': '#ffffff',
            'entry_fg': '#2d3748',
            'selection_bg': '#48bb78',
            'selection_fg': '#ffffff'
        },
        'purple': {
            'bg_primary': '#faf5ff',
            'bg_secondary': '#f3e8ff',
            'bg_tertiary': '#e9d8fd',
            'bg_accent': '#d6bcfa',
            'fg_primary': '#1a202c',
            'fg_secondary': '#4a5568',
            'fg_accent': '#6b46c1',
            'border': '#e9d8fd',
            'border_focus': '#9f7aea',
            'success': '#38a169',
            'warning': '#d69e2e',
            'danger': '#e53e3e',
            'info': '#3182ce',
            'button_bg': '#9f7aea',
            'button_fg': '#ffffff',
            'button_hover': '#805ad5',
            'button_active': '#6b46c1',
            'entry_bg': '#ffffff',
            'entry_fg': '#2d3748',
            'selection_bg': '#9f7aea',
            'selection_fg': '#ffffff'
        },
        'warm': {
            'bg_primary': '#fffbf0',
            'bg_secondary': '#fef5e7',
            'bg_tertiary': '#fed7aa',
            'bg_accent': '#fbb6ce',
            'fg_primary': '#1a202c',
            'fg_secondary': '#4a5568',
            'fg_accent': '#c05621',
            'border': '#fed7aa',
            'border_focus': '#ed8936',
            'success': '#38a169',
            'warning': '#d69e2e',
            'danger': '#e53e3e',
            'info': '#3182ce',
            'button_bg': '#ed8936',
            'button_fg': '#ffffff',
            'button_hover': '#dd6b20',
            'button_active': '#c05621',
            'entry_bg': '#ffffff',
            'entry_fg': '#2d3748',
            'selection_bg': '#ed8936',
            'selection_fg': '#ffffff'
        }
    }
    
    def __init__(self, root: tk.Tk, theme_name: str = 'light'):
        self.root = root
        self.style = ttk.Style()
        
        # Detect system theme preference if auto is requested
        if theme_name == 'auto':
            self.current_theme = self._detect_system_theme()
        else:
            self.current_theme = theme_name
            
        self.colors = self.THEMES[self.current_theme].copy()
        
        self._setup_fonts()
        self._apply_theme()
    
    @staticmethod
    def _detect_system_theme() -> str:
        """Detect system theme preference"""
        try:
            # Try to detect dark mode on Windows 10/11
            if sys.platform == "win32":
                import winreg
                try:
                    registry = winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER)
                    key = winreg.OpenKey(registry, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Themes\Personalize")
                    value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
                    winreg.CloseKey(key)
                    return 'light' if value else 'dark'
                except:
                    pass
            
            # Try to detect dark mode on macOS
            elif sys.platform == "darwin":
                import subprocess
                try:
                    result = subprocess.run(['defaults', 'read', '-g', 'AppleInterfaceStyle'], 
                                          capture_output=True, text=True, check=True)
                    return 'dark' if 'Dark' in result.stdout else 'light'
                except:
                    pass
        except:
            pass
        
        # Default to light theme
        return 'light'
    
    def _setup_fonts(self):
        """Set up custom fonts"""
        # Define font families with fallbacks
        if sys.platform == "win32":
            self.fonts = {
                'default': ('Segoe UI', 9),
                'heading': ('Segoe UI', 11, 'bold'),
                'subheading': ('Segoe UI', 10, 'bold'),
                'small': ('Segoe UI', 8),
                'monospace': ('Consolas', 9)
            }
        elif sys.platform == "darwin":
            self.fonts = {
                'default': ('SF Pro Text', 13),
                'heading': ('SF Pro Text', 15, 'bold'),
                'subheading': ('SF Pro Text', 14, 'bold'),
                'small': ('SF Pro Text', 11),
                'monospace': ('SF Mono', 13)
            }
        else:  # Linux
            self.fonts = {
                'default': ('Ubuntu', 10),
                'heading': ('Ubuntu', 12, 'bold'),
                'subheading': ('Ubuntu', 11, 'bold'),
                'small': ('Ubuntu', 9),
                'monospace': ('Ubuntu Mono', 10)
            }
    
    def _apply_theme(self):
        """Apply the current theme to all widgets"""
        # Configure root window
        self.root.configure(bg=self.colors['bg_primary'])
        
        # Configure ttk styles
        self._configure_frame_styles()
        self._configure_label_styles()
        self._configure_button_styles()
        self._configure_entry_styles()
        self._configure_treeview_styles()
        self._configure_notebook_styles()
        self._configure_progressbar_styles()
        self._configure_checkbutton_styles()
        self._configure_combobox_styles()
        self._configure_scrollbar_styles()
    
    def _configure_frame_styles(self):
        """Configure frame styles"""
        # Main frame
        self.style.configure('TFrame',
                           background=self.colors['bg_primary'],
                           borderwidth=0)
        
        # Secondary frame
        self.style.configure('Secondary.TFrame',
                           background=self.colors['bg_secondary'],
                           borderwidth=1,
                           relief='solid')
        
        # LabelFrame - more compact
        self.style.configure('TLabelframe',
                           background=self.colors['bg_primary'],
                           borderwidth=1,
                           relief='solid')
        
        self.style.configure('TLabelframe.Label',
                           background=self.colors['bg_primary'],
                           foreground=self.colors['fg_accent'],
                           font=self.fonts['subheading'])
    
    def _configure_label_styles(self):
        """Configure label styles"""
        # Default label
        self.style.configure('TLabel',
                           background=self.colors['bg_primary'],
                           foreground=self.colors['fg_primary'],
                           font=self.fonts['default'])
        
        # Heading label
        self.style.configure('Heading.TLabel',
                           background=self.colors['bg_primary'],
                           foreground=self.colors['fg_accent'],
                           font=self.fonts['heading'])
        
        # Subheading label
        self.style.configure('Subheading.TLabel',
                           background=self.colors['bg_primary'],
                           foreground=self.colors['fg_primary'],
                           font=self.fonts['subheading'])
        
        # Small label
        self.style.configure('Small.TLabel',
                           background=self.colors['bg_primary'],
                           foreground=self.colors['fg_secondary'],
                           font=self.fonts['small'])
        
        # Success label
        self.style.configure('Success.TLabel',
                           background=self.colors['bg_primary'],
                           foreground=self.colors['success'],
                           font=self.fonts['default'])
        
        # Warning label
        self.style.configure('Warning.TLabel',
                           background=self.colors['bg_primary'],
                           foreground=self.colors['warning'],
                           font=self.fonts['default'])
        
        # Error label
        self.style.configure('Error.TLabel',
                           background=self.colors['bg_primary'],
                           foreground=self.colors['danger'],
                           font=self.fonts['default'])
    
    def _configure_button_styles(self):
        """Configure button styles"""
        # Primary button
        self.style.configure('Primary.TButton',
                           background=self.colors['button_bg'],
                           foreground=self.colors['button_fg'],
                           borderwidth=0,
                           focuscolor='none',
                           font=self.fonts['default'],
                           padding=(12, 8))
        
        self.style.map('Primary.TButton',
                      background=[('active', self.colors['button_hover']),
                                ('pressed', self.colors['button_active'])])
        
        # Success button
        self.style.configure('Success.TButton',
                           background=self.colors['success'],
                           foreground='white',
                           borderwidth=0,
                           focuscolor='none',
                           font=self.fonts['default'],
                           padding=(12, 8))
        
        self.style.map('Success.TButton',
                      background=[('active', '#218838'),
                                ('pressed', '#1e7e34')])
        
        # Danger button
        self.style.configure('Danger.TButton',
                           background=self.colors['danger'],
                           foreground='white',
                           borderwidth=0,
                           focuscolor='none',
                           font=self.fonts['default'],
                           padding=(12, 8))
        
        self.style.map('Danger.TButton',
                      background=[('active', '#c82333'),
                                ('pressed', '#bd2130')])
        
        # Secondary button
        self.style.configure('Secondary.TButton',
                           background=self.colors['bg_tertiary'],
                           foreground=self.colors['fg_primary'],
                           borderwidth=1,
                           focuscolor='none',
                           font=self.fonts['default'],
                           padding=(12, 8))
        
        self.style.map('Secondary.TButton',
                      background=[('active', self.colors['bg_secondary']),
                                ('pressed', self.colors['border'])])
    
    def _configure_entry_styles(self):
        """Configure entry styles"""
        self.style.configure('TEntry',
                           fieldbackground=self.colors['entry_bg'],
                           foreground=self.colors['entry_fg'],
                           borderwidth=2,
                           insertcolor=self.colors['fg_primary'],
                           font=self.fonts['default'],
                           padding=8)
        
        self.style.map('TEntry',
                      focuscolor=[('!focus', self.colors['border']),
                                ('focus', self.colors['border_focus'])])
    
    def _configure_treeview_styles(self):
        """Configure treeview styles"""
        self.style.configure('Treeview',
                           background=self.colors['entry_bg'],
                           foreground=self.colors['entry_fg'],
                           fieldbackground=self.colors['entry_bg'],
                           borderwidth=1,
                           font=self.fonts['default'])
        
        self.style.configure('Treeview.Heading',
                           background=self.colors['bg_tertiary'],
                           foreground=self.colors['fg_primary'],
                           borderwidth=1,
                           font=self.fonts['subheading'])
        
        self.style.map('Treeview',
                      background=[('selected', self.colors['selection_bg'])],
                      foreground=[('selected', self.colors['selection_fg'])])
        
        self.style.map('Treeview.Heading',
                      background=[('active', self.colors['bg_secondary'])])
    
    def _configure_notebook_styles(self):
        """Configure notebook (tab) styles"""
        self.style.configure('TNotebook',
                           background=self.colors['bg_primary'],
                           borderwidth=0)
        
        self.style.configure('TNotebook.Tab',
                           background=self.colors['bg_tertiary'],
                           foreground=self.colors['fg_primary'],
                           borderwidth=1,
                           padding=(12, 8),
                           font=self.fonts['default'])
        
        self.style.map('TNotebook.Tab',
                      background=[('selected', self.colors['bg_primary']),
                                ('active', self.colors['bg_secondary'])],
                      foreground=[('selected', self.colors['fg_accent'])])
    
    def _configure_progressbar_styles(self):
        """Configure progressbar styles"""
        self.style.configure('TProgressbar',
                           background=self.colors['button_bg'],
                           borderwidth=0,
                           lightcolor=self.colors['button_bg'],
                           darkcolor=self.colors['button_bg'])
    
    def _configure_checkbutton_styles(self):
        """Configure checkbutton styles"""
        self.style.configure('TCheckbutton',
                           background=self.colors['bg_primary'],
                           foreground=self.colors['fg_primary'],
                           focuscolor='none',
                           font=self.fonts['default'])
        
        self.style.map('TCheckbutton',
                      background=[('active', self.colors['bg_primary'])])
    
    def _configure_combobox_styles(self):
        """Configure combobox styles"""
        self.style.configure('TCombobox',
                           fieldbackground=self.colors['entry_bg'],
                           foreground=self.colors['entry_fg'],
                           background=self.colors['entry_bg'],
                           borderwidth=2,
                           font=self.fonts['default'],
                           padding=8)
        
        self.style.map('TCombobox',
                      focuscolor=[('!focus', self.colors['border']),
                                ('focus', self.colors['border_focus'])])
    
    def _configure_scrollbar_styles(self):
        """Configure scrollbar styles"""
        self.style.configure('TScrollbar',
                           background=self.colors['bg_tertiary'],
                           borderwidth=0,
                           arrowcolor=self.colors['fg_secondary'],
                           troughcolor=self.colors['bg_secondary'])
        
        self.style.map('TScrollbar',
                      background=[('active', self.colors['bg_secondary'])])
    
    def switch_theme(self, theme_name: str):
        """Switch to a different theme"""
        if theme_name == 'auto':
            self.current_theme = self._detect_system_theme()
        elif theme_name in self.THEMES:
            self.current_theme = theme_name
        else:
            # Fallback to light theme if invalid theme name
            self.current_theme = 'light'
            
        self.colors = self.THEMES[self.current_theme].copy()
        self._apply_theme()
    
    def get_color(self, color_name: str) -> str:
        """Get a color from the current theme"""
        return self.colors.get(color_name, '#000000')
    
    def get_font(self, font_name: str) -> tuple:
        """Get a font from the current theme"""
        return self.fonts.get(font_name, self.fonts['default'])
    
    @staticmethod
    def create_styled_widget(widget_class, parent, style_name: str = None, **kwargs):
        """Create a widget with the specified style"""
        if style_name:
            kwargs['style'] = style_name
        return widget_class(parent, **kwargs)


# Utility functions for common styled widgets
def create_primary_button(parent, theme_manager: ThemeManager, **kwargs):
    """Create a primary styled button"""
    return theme_manager.create_styled_widget(ttk.Button, parent, 'Primary.TButton', **kwargs)

def create_success_button(parent, theme_manager: ThemeManager, **kwargs):
    """Create a success styled button"""
    return theme_manager.create_styled_widget(ttk.Button, parent, 'Success.TButton', **kwargs)

def create_danger_button(parent, theme_manager: ThemeManager, **kwargs):
    """Create a danger styled button"""
    return theme_manager.create_styled_widget(ttk.Button, parent, 'Danger.TButton', **kwargs)

def create_secondary_button(parent, theme_manager: ThemeManager, **kwargs):
    """Create a secondary styled button"""
    return theme_manager.create_styled_widget(ttk.Button, parent, 'Secondary.TButton', **kwargs)

def create_heading_label(parent, theme_manager: ThemeManager, **kwargs):
    """Create a heading styled label"""
    return theme_manager.create_styled_widget(ttk.Label, parent, 'Heading.TLabel', **kwargs)

def create_subheading_label(parent, theme_manager: ThemeManager, **kwargs):
    """Create a subheading styled label"""
    return theme_manager.create_styled_widget(ttk.Label, parent, 'Subheading.TLabel', **kwargs)
