# FS FilterLab

A web-based tool for analyzing and visualizing optical filter stacks, quantum efficiency curves, and illuminant spectra. Built with focus on Full-Spectrum photography. 

## Features

- Combine multiple filters and see the resulting transmission
- View and compare RGB channel responses
- Analyze how filters affect different cameras and light sources
- Import your own filter, QE, or illuminant data (TSV format)
- Export analysis as PNG images
- Search and filter by manufacturer, color, or wavelength
- Simple caching for faster data loading

## Quick Start

### Requirements
- Python 3.8 or newer
- pip

### Install

1. Clone this repository,
   Or download the latest Release   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Or use `install.bat` (Windows) or `install.sh` (Linux/macOS).
3. Run the app:
   ```bash
   streamlit run app.py
   ```
   Or use `start.bat` (Windows) or `run.sh` (Linux/macOS).
4. Open your browser to [http://localhost:8501](http://localhost:8501)

## How to Use

1. Select filters from the sidebar
2. Adjust stack counts if needed
3. Pick a camera QE profile and an illuminant
4. See the results in the main area (charts, numbers, etc.)
5. Download a PNG report if you want

### Advanced
- Use "Advanced Search" to filter by manufacturer, color, or transmission at a specific wavelength
- Import your own data in the sidebar (TSV files)
- Rebuild the cache if you add new data files


## Project Structure

```
FS-FilterLab/
├── app.py
├── requirements.txt
├── install.bat / install.sh
├── start.bat / run.sh
├── models/         # Data models
├── services/       # Data processing and logic
├── views/          # UI components
├── data/           # Your spectral data files
└── cache/          # Auto-generated cache
```

## Basic Troubleshooting

- Delete .venv, then run install.bat/.sh to re-install dependencies
- Use "Rebuild Cache" in the sidebar if you add or change data files. Alternatively, manually delete the /cache folder

## License

MIT License. See LICENSE file.

---
