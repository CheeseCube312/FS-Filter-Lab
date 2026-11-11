# FS FilterLab - Usage Guide

This guide walks you through using FS FilterLab for optical analysis.

## Getting Started

### 1. Launch the Application
- **Windows**: Double-click `start.bat`
- **Manual**: Run `python program/app.py`
- The app will open automatically at [http://127.0.0.1:8080](http://127.0.0.1:8080)

### 2. Basic Interface Overview
- **Left Sidebar (Drawer)**: Filter selection, QE/illuminant pickers, settings, and actions
- **Main Content Area**: transmission metrics, sensor response curves, Scene Preview swatches, and metrics tables — raw sensitivity & illuminant curves live in a collapsible section at the bottom
- **Notifications**: Success/error toasts appear in the top-right corner

## Basic Workflow

### Step 1: Select Filters
1. In the sidebar, use the **"Select filters to plot"** dropdown
2. Choose one or more filters from the list
3. Selected filters appear as colored tags
4. Use the **search box** to quickly find specific filters

### Step 2: Configure Filter Stack
1. If multiple filters are selected, they automatically stack (multiply)
2. Use **"Set Filter Stack Counts"** to specify how many of each filter
3. Example: 2x UV Filter + 1x Polarizer = specific transmission curve

### Step 3: Choose Analysis Parameters
1. **Sensor / Film Profile**: Select a digital camera QE profile or a film stock (slide, color negative, or B&W)
2. **Illuminant**: Choose light source (AM1.5 Global is typical daylight)
3. **Display Options**: Toggle individual RGB channels and white balance (switches on the sensor response card)

### Step 4: Analyze Results
1. **Main transmission chart** shows the combined filter response
2. **RGB sensor response** shows how the filtered light affects camera channels
3. **Metrics panels** show quantitative analysis (effective stops, white balance)
4. **Deviation metrics** compare to target profiles (if loaded)

## Advanced Features

### Scene Preview
- The **Scene Preview** card sits below the Sensor response card: every selected surface gets one render swatch (name + reflectance sparkline) that follows the camera-look toggle
- The **camera icon** on the card toggles the "camera look" (filmic tone curve + chroma boost, on by default); with a color-negative film profile the render is inverted like a print
- **Click a swatch** to open its popover: the active render's linear R/G/B, WB-from-surface, and remove
- **Partial data** badge: more than 10% of the chain's spectral weight lands on wavelengths without reflectance data (e.g. UV stacks over leaf data) — the popover lists the missing bands
- No surfaces yet? Click **"Load default scene set"** to add the curated leaf set, or add reflectors from the Reflectors shelf

### Advanced Filter Search
1. On the **Filters** shelf, open the **"Advanced search"** expansion
2. **Manufacturers**: Select brands to narrow choices
3. **Transmission criteria**: Constrain filters by min/max transmission at target wavelengths — all criteria must match
4. **Sorting**: Order results by filter number, name, transmission at λ, or hex-rainbow color

### Sensor and Film Profiles
- The **"Sensor / Film Profile"** picker lists both digital sensors and film stocks
- **Slide film** behaves like a positive sensor; **color negative** film is inverted for display; **B&W film** shows a single grey "Luminance" trace
- Use the **"Film Emulation Target"** picker to overlay a film's response as a target, optionally with its own filter stack ("Filters on Film Target") and independent white balance

### My Filters Manager
1. Click the **"Manage Owned Filters"** toggle in the sidebar
2. Use **"Search and select filter to add"** to mark filters you own; remove them the same way or via the manufacturer bulk controls
3. Toggle **"My Filters only"** in the sidebar to restrict the filter list to your collection
4. Your collection persists in `program/data/filters_data/my_filters.json`

### White Balance from a Surface Reference
1. Load a **Sensor / Film Profile** and an **Illuminant** (required for white balance)
2. Click a **swatch** in the **Scene Preview** card and choose **WB** to use that surface as the white balance reference — it shows **● WB** while active; click again to reset
3. The white balance panel shows **"(from surface reference)"** while active

### Custom Data Import
1. Open the **Data** tab — five import cards, one per data type. Each card walks top-to-bottom: choose the CSV (staged, nothing saved yet), fill in the metadata, then click the **Import** button — it activates once a file is staged:
2. **Filters**: Import custom TSV filter files
3. **Illuminants**: Add custom light sources
4. **Camera QE**: Add digital camera sensor profiles
5. **Film**: Add film profiles (slide, color negative, or B&W) — TSV files with `# Key: Value` headers: `Manufacturer`, `Name`, `Type` required; `ISO`, `Process`, `Notes` optional
6. **Reflectance**: Add surface material data (single-spectrum CSV)

### Report Generation
1. Select your desired filter configuration
2. Choose the sensor / film profile for analysis
3. Click **"Generate Report"** in the sidebar
4. Download the generated PNG report with **"Download Last Report"**

## Understanding the Charts

### Transmission Chart
- **X-axis**: Wavelength (300-1100 nm)
- **Y-axis**: Transmission (0-100%)
- **Multiple Lines**: Individual filters in stack
- **Combined Line**: Final result of all filters

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
