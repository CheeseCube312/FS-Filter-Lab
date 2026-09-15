# FS FilterLab

A web-based tool for analyzing and visualizing optical filter stacks, quantum efficiency curves, and illuminant spectra. Built with NiceGUI for Full-Spectrum photography analysis.

Credit: 21.09.2025 Refactor based on 01luna's fork. Contains Vegetation Color Preview feature she created

## Features

- Combine multiple filters and see the resulting transmission
- View and compare RGB channel responses
- Analyze how filters affect different cameras, film stocks, and light sources
- Film emulation targets: slide, color negative, and B&W film profiles
- Import your own filter, QE, film, or illuminant data (TSV format)
- Export analysis as PNG reports
- Advanced search: filter by manufacturer, color, or transmission at specific wavelengths
- Surface reflector library with default curated lists
- Fast caching system for instant data loading
- Interactive Plotly charts with zoom and hover details

## Quick Start

### Requirements
- Python 3.10 or newer
- pip

### Install

1. Clone this repository (with submodules for filter data):
   ```bash
   git clone --recursive https://github.com/YourUsername/FS-Filter-Lab.git
   ```
   Or download the latest Release
2. Install dependencies:
   ```bash
   pip install -r program/requirements.txt
   ```
   Or use `install.bat` (Windows) to auto-create a virtual environment.
3. Run the app:
   ```bash
   python program/app.py
   ```
   Or use `start.bat` (Windows).
4. Open your browser to [http://127.0.0.1:8080](http://127.0.0.1:8080) (opens automatically)

## How to Use

1. **Select filters** from the sidebar dropdown
2. **Adjust stack counts** if you want multiple instances of a filter
3. **Pick a camera QE profile** and an **illuminant** (light source)
4. **View results**: transmission charts, RGB response curves, color metrics
5. **Add reflectors** to see how surfaces appear under your filter stack
6. **Export PNG reports** for documentation

### Advanced Features
- **Advanced Filter Search**: Filter by manufacturer, hex color, or transmission at specific wavelengths with live sparkline previews
- **Advanced Reflector Search**: Filter surfaces by organization or target type (applies to imported files carrying that metadata)
- **My Filters Manager**: Curate your own filter collection with a "My Filters only" toggle
- **White Balance from Surface Reference**: Use any loaded reflector as the white balance source
- **Channel Mixer**: Apply custom RGB channel mixing matrices
- **Import Data**: Add your own filters, illuminants, camera QE curves, film profiles, and reflector data from the import cards on the Data tab
- **Automatic Cache**: Cached data rebuilds silently whenever data files change — nothing to manage

## Project Structure

```
FS-FilterLab/
├── install.bat           # Windows installer script
├── start.bat             # Windows launcher
├── README.md             # This file
├── USAGE.md              # Detailed usage guide
└── program/
    ├── app.py            # NiceGUI application entry point
    ├── requirements.txt  # Python dependencies
    ├── TECHNICAL.md      # Developer technical overview
    ├── models/           # Data structures and constants
    ├── services/         # Business logic (calculations, presenter, data loading, state)
    ├── views/            # NiceGUI UI components
    ├── tests/            # unittest suite
    ├── data/             # Spectral data files (git submodule)
    │   ├── filters_data/
    │   ├── QE_data/
    │   ├── film_data/
    │   ├── illuminants/
    │   └── reflectors/
    ├── cache/            # Auto-generated (gitignored)
    └── output/           # Generated reports (gitignored)
```

## Basic Troubleshooting

- **Missing dependencies**: Delete `.venv/`, then run `install.bat` to reinstall
- **Data not loading**: Delete the `program/cache/` folder — it rebuilds automatically on the next start
- **Port already in use**: Edit `app.py` and change the port in `ui.run(port=8080)`
- **App still running after closing the tab**: Stop the server with `Ctrl+C` in its console (or close the console window)
- **Submodule data missing**: Run `git submodule update --init --recursive`

## License

MIT License. See `program/LICENSE` file.

---
