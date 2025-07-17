# PangaDFS GUI Application

A comprehensive graphical user interface for the PangaDFS fantasy sports lineup optimization framework, supporting both single lineup and multilineup optimization up to 150 lineups.

## Features

### 🎯 **Complete Configuration Management**
- **Genetic Algorithm Settings**: Population size, generations, mutation rate, crossover methods, and more
- **Multilineup Optimization**: Support for up to 150 lineups with diversity controls
- **Site Settings**: Salary cap, position requirements, flex positions
- **Configuration Presets**: Save and load optimization configurations
- **Auto-detection**: Automatically detect CSV column mappings

### 🚀 **Optimization Execution**
- **Background Processing**: Run optimizations without freezing the GUI
- **Real-time Logging**: Monitor optimization progress with detailed logs
- **Progress Tracking**: Visual progress indicators and status updates
- **Stop/Resume**: Control optimization execution

### 📊 **Comprehensive Results Display**
- **Summary Dashboard**: Key metrics, best scores, diversity statistics
- **All Lineups View**: Sortable table of all generated lineups
- **Individual Lineup Details**: Detailed view of any specific lineup
- **Diversity Analysis**: Overlap matrices and diversity metrics
- **Performance Profiling**: Optimization performance statistics

### 💾 **Flexible Export Options**
- **Multiple Formats**: CSV, Excel, JSON export support
- **Selective Export**: Choose specific lineups or ranges
- **Content Control**: Select which data to include (lineups, summary, diversity, profiling)
- **Auto-naming**: Automatic filename generation with timestamps

## Installation

1. **Install PangaDFS** (if not already installed):
   ```bash
   pip install -e .
   ```

2. **Install Optional Dependencies** (for full functionality):
   ```bash
   pip install openpyxl  # For Excel export support
   ```

3. **Launch the GUI**:
   ```bash
   pangadfs-gui
   ```

   Or run directly:
   ```bash
   python pangadfs/app/gui_app.py
   ```

## Quick Start Guide

### 1. **Configure Your Optimization**
- Go to the **Configuration** tab
- Select your player pool CSV file using "Browse..."
- Verify column mappings (auto-detected)
- Adjust GA settings as needed
- For multilineup optimization:
  - Check "Enable Multilineup Optimization"
  - Set target number of lineups (up to 150)
  - Configure diversity settings

### 2. **Run Optimization**
- Switch to the **Run Optimization** tab
- Click "Start Optimization"
- Monitor progress in the log area
- Wait for completion (results tabs will be enabled)

### 3. **View Results**
- **Summary Tab**: Overview of key metrics and best lineup
- **All Lineups Tab**: Browse all generated lineups
- **Lineup Details Tab**: Examine individual lineups
- **Diversity Analysis Tab**: View lineup diversity metrics

### 4. **Export Results**
- Go to the **Export** tab
- Choose export format (CSV, Excel, JSON)
- Select content to include
- Choose lineup selection (all, top N, or range)
- Click "Export to File..." or "Quick Export"

## Configuration Options

### Genetic Algorithm Settings
- **Population Size**: Number of individuals per generation (default: 30,000)
- **Generations**: Maximum number of generations (default: 20)
- **Mutation Rate**: Probability of mutation (default: 0.05)
- **Elite Divisor**: Elite selection parameter (default: 5)
- **Stop Criteria**: Early stopping threshold (default: 10)

### Multilineup Settings
- **Target Lineups**: Number of lineups to generate (1-150)
- **Diversity Weight**: Balance between score and diversity (0-1)
- **Min Overlap Threshold**: Minimum diversity between lineups (0-1)
- **Diversity Method**: Algorithm for calculating diversity (Jaccard/Hamming)

### Site Settings
- **Salary Cap**: Maximum total salary per lineup
- **Position Map**: Number of players per position
- **Flex Positions**: Positions eligible for FLEX slots

## File Formats

### CSV Input Format
Your player pool CSV should contain at minimum:
- **Player names** (any column name)
- **Positions** (default: 'pos')
- **Salaries** (default: 'salary')
- **Projections** (default: 'proj')

Example:
```csv
player,pos,salary,proj,team
Patrick Mahomes,QB,8000,24.5,KC
Derrick Henry,RB,7500,18.2,TEN
Tyreek Hill,WR,8500,16.8,MIA
```

### Export Formats

#### CSV Export
- One row per player per lineup
- Includes lineup ID, scores, and player details

#### Excel Export
- **Lineups Sheet**: Player data with lineup groupings
- **Summary Sheet**: Key statistics and metrics
- **Diversity Sheet**: Diversity analysis (if enabled)

#### JSON Export
- Structured data with nested lineup objects
- Includes all metadata and statistics
- Suitable for programmatic processing

## Advanced Features

### Configuration Management
- **Auto-save**: Configurations are automatically saved as defaults
- **Presets**: Save multiple named configurations
- **Validation**: Real-time validation with helpful error messages

### Logging and Debugging
- **Detailed Logs**: All optimization steps are logged
- **Log Export**: Save logs for troubleshooting
- **Performance Metrics**: Track optimization performance

### Multilineup Optimization
- **Diversity Control**: Ensure lineups are sufficiently different
- **Score Optimization**: Balance high scores with diversity
- **Scalable**: Efficiently generate up to 150 unique lineups

## Troubleshooting

### Common Issues

**"No optimization results available"**
- Ensure optimization completed successfully
- Check the log for error messages
- Verify your CSV file format and column mappings

**"Configuration errors found"**
- Review the Configuration tab for validation errors
- Ensure all required fields are filled
- Check that CSV file exists and is accessible

**"Failed to import pangadfs modules"**
- Ensure PangaDFS is properly installed: `pip install -e .`
- Check that all dependencies are installed
- Verify Python path includes the project directory

### Performance Tips

**For Large Player Pools (>1000 players):**
- Reduce population size to 10,000-20,000
- Increase elite divisor to 10
- Consider fewer generations (10-15)

**For Many Lineups (>50):**
- Increase diversity weight to 0.4-0.6
- Use Jaccard diversity method
- Monitor memory usage during optimization

## Technical Details

### Architecture
- **Tkinter GUI**: Cross-platform native interface
- **Threading**: Background optimization processing
- **Plugin System**: Uses PangaDFS's extensible plugin architecture
- **Configuration**: JSON-based settings management

### Dependencies
- **Required**: pandas, numpy, stevedore, tkinter (built-in)
- **Optional**: openpyxl (Excel export)

### File Locations
- **Configurations**: `~/.pangadfs/`
- **Logs**: `~/.pangadfs/logs/`
- **Exports**: Current working directory (or user-selected)

## Support

For issues, questions, or feature requests:
1. Check the logs in the GUI for detailed error information
2. Review this documentation and the main PangaDFS documentation
3. Create an issue in the project repository

## License

This GUI application is part of PangaDFS and is licensed under the MIT License.
