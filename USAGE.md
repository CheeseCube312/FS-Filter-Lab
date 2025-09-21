# FS FilterLab - Usage Guide

This guide walks you through using FS FilterLab for optical analysis.

## 🚀 Getting Started

### 1. Launch the Application
- **Windows**: Double-click `start.bat`
- **Linux/macOS**: Run `./run.sh`
- **Manual**: Run `streamlit run app.py`

### 2. Basic Interface Overview
- **Sidebar**: Filter selection, settings, and controls
- **Main Area**: Charts, analysis results, and data displays
- **Status Messages**: Feedback and error information at the top

## 📊 Basic Workflow

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
1. **Camera QE**: Select quantum efficiency profile (or use Default)
2. **Illuminant**: Choose light source (AM1.5 Global is typical daylight)
3. **Display Options**: Toggle RGB channels, log scale, white balance

### Step 4: Analyze Results
1. **Main transmission chart** shows the combined filter response
2. **RGB sensor response** shows how the filtered light affects camera channels
3. **Metrics panels** show quantitative analysis (effective stops, white balance)
4. **Deviation metrics** compare to target profiles (if loaded)

## 🔍 Advanced Features

### Advanced Filter Search
1. Click **"Show Advanced Search"** in the sidebar
2. **Filter by Manufacturer**: Select brands to narrow choices
3. **Transmission at Wavelength**: Find filters with specific properties at target wavelengths
4. **Color Sorting**: Sort by rainbow color for visual selection

### Custom Data Import
1. Click **"Show Import Data"** in the sidebar
2. **Upload Filter Data**: Import custom TSV filter files
3. **Upload QE Data**: Add camera sensor profiles
4. **Upload Illuminant**: Add custom light sources
5. **Upload Reflectance**: Add surface material data

### Report Generation
1. Select your desired filter configuration
2. Choose the camera profile for analysis
3. Click **"📊 Generate Report"** in the sidebar
4. Download the generated PNG report with **"⬇️ Download Last Report"**

## 📈 Understanding the Charts

### Transmission Chart
- **X-axis**: Wavelength (300-1100 nm)
- **Y-axis**: Transmission (0-100% or logarithmic)
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

## 🎯 Common Use Cases

### Photography Filter Analysis
1. Select camera QE profile matching your camera
2. Choose daylight illuminant (AM1.5 Global)
3. Add ND, UV, or polarizer filters as needed
4. Check effective stops for exposure compensation
5. Review white balance for color correction

### Scientific Optical Design
1. Load custom illuminant matching your source
2. Select multiple bandpass or blocking filters
3. Use target profile for optimization goals
4. Generate reports for documentation
5. Export results for further analysis

### Educational Demonstrations
1. Start with simple single filters
2. Show how multiple filters combine
3. Demonstrate RGB channel separation
4. Explain effective stops concept
5. Compare different illuminant effects

## ⚙️ Settings and Preferences

### Display Options
- **Log View**: Toggle between linear and logarithmic transmission scales
- **RGB Channels**: Show/hide individual red, green, blue responses
- **Apply White Balance**: Enable automatic white balance correction

### Performance Options
- **Rebuild Cache**: Clear cached data when adding new files
- **Advanced Search**: Enable multi-criteria filter search
- **Import Data**: Enable file upload interfaces

## 🛠️ Troubleshooting

### Common Issues

**No filters appear:**
- Check that TSV files are in `data/filters_data/` folder
- Verify files have correct column headers
- Try rebuilding cache from settings

**Charts not updating:**
- Ensure filters are actually selected (check sidebar)
- Try refreshing the browser page
- Check browser console for JavaScript errors

**Slow performance:**
- Reduce number of selected filters
- Use advanced search to filter dataset
- Close other browser tabs using memory

**Import fails:**
- Verify file format matches expected TSV structure
- Check for special characters in data
- Ensure wavelength range covers 300-1100nm

### Getting Better Results

**Filter Selection:**
- Start with single filters to understand behavior
- Combine filters logically (UV + ND, not competing bandpass)
- Use manufacturer data when available

**Analysis Setup:**
- Match camera QE to your actual equipment
- Choose appropriate illuminant for your scenario
- Consider target profiles for specific goals

**Report Quality:**
- Select meaningful filter combinations
- Include relevant analysis parameters
- Add descriptive notes for documentation

## 📚 Tips and Best Practices

### Efficient Workflow
1. **Bookmark Configurations**: Use browser bookmarks for common setups
2. **Batch Analysis**: Compare multiple configurations systematically  
3. **Save Reports**: Download results before changing configurations
4. **Document Parameters**: Note illuminant and camera choices in reports

### Data Management
1. **Organize Filters**: Use clear naming conventions in data files
2. **Regular Backups**: Keep copies of custom data files
3. **Cache Maintenance**: Rebuild cache when adding new data
4. **Version Control**: Track changes to custom data sets

### Analysis Quality
1. **Validate Results**: Cross-check with known references
2. **Consider Uncertainties**: Remember extrapolated data limitations
3. **Document Assumptions**: Note analysis conditions and choices
4. **Peer Review**: Have colleagues review critical analyses

---

Need more help? Check the detailed README.md or create an issue on GitHub!
