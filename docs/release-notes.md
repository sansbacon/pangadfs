# Release Notes

## 0.2.0

**Major Changes:**

* **GUI Separation**: The GUI application has been moved to a separate repository ([pangadfs-gui](https://github.com/sansbacon/pangadfs-gui)) for better modularity and maintenance
* **Reduced Dependencies**: Core library now has minimal dependencies, with GUI-specific dependencies moved to the separate package
* **Focused Core**: The main pangadfs package now focuses purely on the genetic algorithm optimization engine

**Breaking Changes:**

* Removed `pangadfs-gui` console command from core package
* Removed GUI-related dependencies from core requirements
* GUI functionality now requires separate installation: `pip install pangadfs-gui`

**Installation Changes:**

* Core library: `pip install pangadfs`
* GUI application: `pip install pangadfs-gui` (requires core library)

## 0.1.1

Updated documentation structure

## 0.1.0

* Feature Description

Working version of basic application
