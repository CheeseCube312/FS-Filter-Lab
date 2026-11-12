# FS FilterLab - Usage Guide

This guide walks you through using FS FilterLab for optical analysis.

## Getting Started

### 1. Launch the Application
- **Windows**: Double-click `start.bat`
- **Manual**: Run `python program/app.py` (on macOS/Linux: `python3 program/app.py`)
- The app will open automatically at [http://127.0.0.1:8080](http://127.0.0.1:8080)
- **Sanity check**: the console prints `Loaded N filters from M manufacturer(s)` at startup. If the Filters shelf shows far fewer filters than expected (e.g. a single brand), this copy's `program/data/` folder is incomplete or its cache is stale: re-copy the full `program/data/` tree, delete `program/cache/`, and relaunch

### 2. Basic Interface Overview
- **Left Sidebar (Setup rail)**: Receiver picker (digital sensors + film), illuminant picker, and the filter stack editor
- **Right Drawer (Shelves)**: **Filters** shelf (search + My Filters) and **Data** tab (import cards)
- **Main Content Area**: transmission chart + metrics, sensor response curves, and a collapsible raw-curve Reference section at the bottom
- **Header**: status line (receiver · illuminant · stack · WB) plus **Report** and **Reset** buttons
- **Notifications**: Success/error toasts appear in the top-right corner

## Basic Workflow

### Step 1: Add Filters to the Stack
1. On the **Filters** shelf (right drawer), use the **search box** to find filters
2. Click **+** on a result row to add it to the stack (click the **info** icon for a full-size transmission curve)
3. Click **★** on results you own to build your My Filters collection

### Step 2: Configure Filter Stack
1. Stacked filters multiply; the sidebar **Filter stack** editor lists each one with its color chip
2. Use the row's count dropdown (1–5) for multiple instances — each pass multiplies the filter's transmission (T^N; hover the dropdown for the current factor)
3. Remove a filter with the row's ✕ button, or clear the whole stack

### Step 3: Choose Analysis Parameters
1. **Receiver**: pick a digital camera QE profile or a film stock (slide, color negative, or B&W) — one dropdown; film stocks carry a "(film)" label suffix
2. **Illuminant**: choose light source (AM1.5 Global is typical daylight)
3. **Display Options**: toggle individual RGB channels and white balance (switches on the sensor response card)

### Step 4: Analyze Results
1. **Transmission chart** shows each stacked filter as a faint trace of its own (multiplied) curve beneath the bold combined line
2. **RGB sensor response** shows how the filtered light affects camera channels
3. **Metrics row** shows quantitative analysis (average transmission, effective stops)

## Advanced Features

### Advanced Filter Search
1. On the **Filters** shelf, open the **"Advanced search"** expansion
2. **Manufacturers**: select brands to narrow choices
3. **Transmission criteria**: constrain filters by min/max transmission at target wavelengths — all criteria must match
4. **Sorting**: order results by filter number, name, transmission at λ, or hex-rainbow color
5. Results update live as you change criteria — no Apply step; **"Owned only"** applies here too
6. If there are more than 60 matches, use **Show more** to reveal the next batch

### Sensor and Film Profiles
- The **Receiver** picker lists both digital sensors and film stocks (films labelled with a "(film)" suffix)
- **Slide film** behaves like a positive sensor; **color negative** film is inverted for display; **B&W film** shows a single grey "Luminance" trace

### My Filters Manager
1. On the **Filters** shelf, star result rows (★) to mark filters you own
2. Open the **My Filters** expansion for bulk controls: mark/unmark whole manufacturers, remove single entries, or clear all
3. Toggle **"Owned only"** above the search box to restrict results to your collection (quick and advanced search both honor it)
4. Your collection persists in `program/data/filters_data/my_filters.json`

### Custom Data Import
1. Open the **Data** tab — four import cards. Each card walks top-to-bottom: choose the CSV (staged, nothing saved yet), fill in the metadata, then click the **Import** button — it activates once a file is staged:
2. **Filters**: import custom filter CSV files (wavelength, transmission, and metadata)
3. **Illuminants**: add custom light sources
4. **Camera QE**: add digital camera sensor profiles
5. **Film**: add film profiles (slide, color negative, or B&W) — TSV files with `# Key: Value` headers: `Manufacturer`, `Name`, `Type` required; `ISO`, `Process`, `Notes` optional

### Report Generation
1. Select your desired filter configuration
2. Choose the receiver profile for analysis
3. Click **Report** in the page header — the PNG downloads automatically

## Understanding the Charts

### Transmission Chart
- **X-axis**: Wavelength (300-1100 nm)
- **Y-axis**: Transmission (0-100%)
- **Faint colored traces**: each stacked filter's own curve (labeled ×N when stacked multiple times)
- **Bold black line**: combined result of all filters

### RGB Response Chart
- **Red/Green/Blue Lines**: How each camera channel responds
- **Combined Effect**: Shows color shifts and intensity changes
- **White Balance**: Correction factors for neutral response

### Sparkline Plots
- **Miniature Charts**: Quick visual summary in selection lists
- **Filter Overview**: Rapid identification of filter characteristics
- **Comparison Tool**: Easy visual comparison between options


## Settings and Preferences

### Display Options
- **RGB Channels**: Show/hide individual red, green, blue responses (per-channel checkboxes on the sensor response card)
- **WB**: Enable automatic white balance correction (switch on the sensor response card)
