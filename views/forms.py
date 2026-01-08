"""
Form-based UI components for FS FilterLab.

This module provides UI components for forms, including search and import interfaces.
"""
# Third-party imports
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Local imports
from models.constants import INTERP_GRID, UI_BUTTONS, UI_WARNING_MESSAGES
from views.ui_utils import is_dark_color, is_valid_hex_color, handle_error
from services.visualization import create_sparkline_plot

# Cache the sparkline plot to improve performance when toggling filter details
@st.cache_data
def cached_create_sparkline_plot(wavelengths, transmission, color):
    """Cached version of create_sparkline_plot to improve performance."""
    return create_sparkline_plot(wavelengths, transmission, color=color)


# -- Filter & Sort Utilities -----------------------------------------

def filter_by_manufacturer(df: pd.DataFrame, manufacturers: List[str]) -> pd.DataFrame:
    """
    Filter DataFrame by manufacturer names.
    
    Args:
        df: DataFrame to filter
        manufacturers: List of manufacturer names
        
    Returns:
        Filtered DataFrame
    """
    return df if not manufacturers else df[df["Manufacturer"].isin(manufacturers)]


def filter_by_trans_at_wavelength(
    df: pd.DataFrame,
    interp_grid: np.ndarray,
    matrix: np.ndarray,
    wavelength: int,
    min_t: float = 0.0,
    max_t: float = 1.0,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Filter DataFrame by transmission at a specific wavelength.
    
    Args:
        df: DataFrame of filters
        interp_grid: Wavelength grid
        matrix: Transmission matrix
        wavelength: Target wavelength
        min_t: Minimum transmission (0-1)
        max_t: Maximum transmission (0-1)
        
    Returns:
        Tuple of (filtered DataFrame, transmission values)
    """
    idx = np.where(interp_grid == wavelength)[0]
    if idx.size == 0:
        handle_error(f"Wavelength {wavelength} nm not in interpolation grid")
        return df, np.zeros(len(df))
    selected_index = idx[0]

    # Make sure we're filtering the correct rows in the matrix
    df_indices = df.index.to_numpy()
    transmission_values = matrix[df_indices, selected_index]

    transmission_mask = (transmission_values >= min_t) & (transmission_values <= max_t)

    return df.iloc[transmission_mask], transmission_values[transmission_mask]


def sort_by_hex_rainbow(df: pd.DataFrame, hex_col: str = "Hex Color") -> pd.DataFrame:
    """
    Sort DataFrame by hex color in rainbow order.
    
    Args:
        df: DataFrame to sort
        hex_col: Name of the hex color column
        
    Returns:
        Sorted DataFrame
    """
    import colorsys
    
    def hex_to_hsl(hex_str):
        hex_str = hex_str.lstrip('#')
        try:
            r, g, b = (int(hex_str[i:i+2], 16)/255.0 for i in (0, 2, 4))
            h, l, s = colorsys.rgb_to_hls(r, g, b)
            return (h, s, l)
        except (ValueError, IndexError):
            return (0, 0, 0)

    valid_mask = df[hex_col].apply(is_valid_hex_color)
    invalid_rows = df[~valid_mask]
    if not invalid_rows.empty:
        st.warning(UI_WARNING_MESSAGES['invalid_hex_colors'].format(count=len(invalid_rows)))
        st.dataframe(invalid_rows[[hex_col, "Filter Number", "Filter Name", "Manufacturer"]])

    hsl_df = df[hex_col].apply(hex_to_hsl).apply(pd.Series)
    hsl_df.columns = ["_hue", "_sat", "_lit"]
    df_sorted = pd.concat([df, hsl_df], axis=1).sort_values(by=["_hue", "_sat", "_lit"])
    return df_sorted.drop(columns=["_hue", "_sat", "_lit"])


def sort_by_trans_at_wavelength(df: pd.DataFrame, trans_vals: np.ndarray, ascending: bool = False) -> pd.DataFrame:
    """
    Sort DataFrame by transmission values.
    
    Args:
        df: DataFrame to sort
        trans_vals: Transmission values
        ascending: Whether to sort in ascending order
        
    Returns:
        Sorted DataFrame
    """
    temp = df.copy()
    temp["_t"] = trans_vals
    return temp.sort_values(by="_t", ascending=ascending).drop(columns=["_t"])


# -- Advanced Search UI -----------------------------------------

def advanced_filter_search(df: pd.DataFrame, filter_matrix: np.ndarray) -> None:
    """
    Display advanced filter search interface with multiple filter criteria.
    
    Args:
        df: DataFrame containing filter metadata
        filter_matrix: Matrix of filter transmission data
    """
    # Check if advanced search is enabled (controlled by sidebar toggle)
    if not st.session_state.get("show_advanced_search", False):
        return

    st.markdown("### Advanced Filter Search")
    st.markdown("Use the controls below to search by manufacturer, color, or spectral transmittance.")

    with st.form("adv_search_form"):
        cols = st.columns([2, 1, 2, 2, 1])
        manufs = cols[0].multiselect("Manufacturer", df["Manufacturer"].unique())
        wl = cols[1].number_input("λ (nm)", 300, 1100, 550, 5)
        tmin, tmax = cols[2].slider("Transmittance range (%)", 0, 100, (0, 100), step=1)
        sort_choice = cols[3].selectbox("Sort by", [
            "Filter Number", "Filter Name", "Hex‑Rainbow", f"Trans @ {wl} nm"
        ])
        with cols[4]:
            st.markdown("<div style='margin-top: 28px'></div>", unsafe_allow_html=True)
            apply_clicked = st.form_submit_button(UI_BUTTONS['apply'])

    if apply_clicked:
        st.session_state.update({
            "filters_applied": True,
            "manufs": manufs,
            "wl": wl,
            "tmin": tmin,
            "tmax": tmax,
            "sort_choice": sort_choice
        })

    manufs = st.session_state.get("manufs", [])
    wl = st.session_state.get("wl", 550)
    tmin = st.session_state.get("tmin", 0)
    tmax = st.session_state.get("tmax", 100)
    sort_choice = st.session_state.get("sort_choice", "Filter Number")

    filters_by_manufacturer = filter_by_manufacturer(df, manufs)
    filtered_results, transmission_values = filter_by_trans_at_wavelength(
        filters_by_manufacturer, INTERP_GRID, filter_matrix, wl, tmin / 100, tmax / 100
    )

    if sort_choice == "Hex‑Rainbow":
        sorted_filters = sort_by_hex_rainbow(filtered_results)
    elif sort_choice.startswith("Trans @"):
        sorted_filters = sort_by_trans_at_wavelength(filtered_results, transmission_values)
    elif sort_choice == "Filter Name":
        sorted_filters = filtered_results.sort_values("Filter Name")
    else:
        sorted_filters = filtered_results.sort_values("Filter Number")

    st.markdown("---")
    st.write(f"**{len(sorted_filters)} filters found:**")

    for idx, row in sorted_filters.iterrows():
        hex_color = row["Hex Color"]
        if not is_valid_hex_color(hex_color):
            hex_color = "#888888"

        number = row["Filter Number"]
        name = row["Filter Name"]
        brand = row["Manufacturer"]
        text_color = "#FFF" if is_dark_color(hex_color) else "#000"

        with st.container():
            cols = st.columns([6, 1])
            with cols[0]:
                st.markdown(f"""
                    <div style="
                        background-color: {hex_color};
                        color: {text_color};
                        padding: 8px 12px;
                        border-radius: 6px;
                        font-weight: 600;
                        font-size: 1rem;
                        margin-bottom: 0;
                    ">
                        {number} — {name} — {brand} 
                    </div>
                """, unsafe_allow_html=True)

            toggle_key = f"filter_toggle_{idx}"
            with cols[1]:
                show_details = st.toggle("Details", key=toggle_key, label_visibility="collapsed")

            if show_details:
                # Use cached version to prevent regenerating the plot on every rerun
                fig = cached_create_sparkline_plot(INTERP_GRID, filter_matrix[idx, :], color=hex_color)
                st.plotly_chart(fig, width='content')
                st.checkbox("Select this filter", key=f"adv_sel_{idx}")

    st.markdown("---")
    col_done, col_cancel = st.columns([1, 1])
    with col_done:
        if st.button(UI_BUTTONS['done']):
            selected_idxs = [
                idx for idx in sorted_filters.index
                if st.session_state.get(f"adv_sel_{idx}", False)
            ]
            selected_display = [
                f"{sorted_filters.loc[idx, 'Filter Name']} "
                f"({sorted_filters.loc[idx, 'Filter Number']}, "
                f"{sorted_filters.loc[idx, 'Manufacturer']})"
                for idx in selected_idxs
            ]

            st.session_state["_pending_selected_filters"] = selected_display
            st.session_state["_close_advanced_search"] = True
            
            # Clean up selection checkboxes after processing
            for idx in sorted_filters.index:
                st.session_state.pop(f"adv_sel_{idx}", None)
                st.session_state.pop(f"filter_toggle_{idx}", None)
                
            st.rerun()

    with col_cancel:
        if st.button(UI_BUTTONS['cancel']):
            st.session_state["_close_advanced_search"] = True
            
            # Clean up selection checkboxes on cancel
            for idx in sorted_filters.index:
                st.session_state.pop(f"adv_sel_{idx}", None)
                st.session_state.pop(f"filter_toggle_{idx}", None)
                
            st.rerun()


# -- Import UI -----------------------------------------

def import_data_form() -> None:
    """Display the data import form with tabs for different data types."""
    """
    Display data import UI with all import options.
    """
    st.markdown("---")
    st.subheader("Import Data")
    
    # Create tabs for different import types
    tab1, tab2, tab3, tab4 = st.tabs(["Filters", "Illuminants", "Camera QE", "Reflectance/ECOSIS"])
    
    with tab1:
        import_filter_tab()
    
    with tab2:
        import_illuminant_tab()
    
    with tab3:
        import_qe_tab()
    
    with tab4:
        import_reflectance_tab()


def import_filter_tab():
    """Display the filter data import interface."""
    """Filter import interface."""
    from services.importing import import_filter_from_csv
    
    uploaded_file = st.file_uploader(
        "Upload CSV (Wavelength, Transmittance)", 
        type="csv", 
        key="filter_upload"
    )
    
    if uploaded_file is not None:
        with st.form("filter_import_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                manufacturer = st.text_input("Manufacturer", value="Custom")
                filter_name = st.text_input("Filter Name", value="Custom Filter")
            
            with col2:
                filter_number = st.text_input("Filter Number", value="001")
                hex_color = st.color_picker("Color", value="#808080")
            
            # Always extrapolate to full range for consistency
            extrap_lower = True
            extrap_upper = True
            
            submitted = st.form_submit_button("Import Filter", type="primary")
            
            if submitted:
                # Basic validation
                if not all([manufacturer.strip(), filter_name.strip(), filter_number.strip()]):
                    st.error("All fields are required")
                else:
                    with st.spinner("Importing..."):
                        meta = {
                            "manufacturer": manufacturer.strip(),
                            "filter_name": filter_name.strip(),
                            "filter_number": filter_number.strip(),
                            "hex_color": hex_color
                        }
                        
                        success, message = import_filter_from_csv(uploaded_file, meta, extrap_lower, extrap_upper)
                        
                        if success:
                            st.success("Filter imported successfully!")
                            st.rerun()  # Refresh to show new data
                        else:
                            st.error(f"Import failed: {message}")


def import_illuminant_tab():
    """Illuminant import interface."""
    from services.importing import import_illuminant_from_csv
    
    uploaded_file = st.file_uploader(
        "Upload CSV (Wavelength, Power)", 
        type="csv", 
        key="illuminant_upload"
    )
    
    if uploaded_file is not None:
        with st.form("illuminant_import_form"):
            description = st.text_input(
                "Illuminant Name", 
                value="Custom Illuminant",
                max_chars=50
            )
            
            submitted = st.form_submit_button("Import Illuminant", type="primary")
            
            if submitted:
                if not description.strip():
                    st.error("Illuminant name is required")
                else:
                    with st.spinner("Importing..."):
                        success, message = import_illuminant_from_csv(uploaded_file, description.strip())
                        
                        if success:
                            st.success("Illuminant imported successfully!")
                            st.rerun()
                        else:
                            st.error(f"Import failed: {message}")


def import_qe_tab():
    """Camera QE import interface."""
    from services.importing import import_qe_from_csv
    
    uploaded_file = st.file_uploader(
        "Upload CSV (Wavelength, R, G, B)", 
        type="csv", 
        key="qe_upload"
    )
    
    if uploaded_file is not None:
        with st.form("qe_import_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                brand = st.text_input("Brand", value="Custom")
            with col2:
                model = st.text_input("Model", value="Custom Model")
            
            submitted = st.form_submit_button("Import Camera QE", type="primary")
            
            if submitted:
                if not all([brand.strip(), model.strip()]):
                    st.error("Both brand and model are required")
                else:
                    with st.spinner("Importing..."):
                        success, message = import_qe_from_csv(uploaded_file, brand.strip(), model.strip())
                        
                        if success:
                            st.success("Camera QE imported successfully!")
                            st.rerun()
                        else:
                            st.error(f"Import failed: {message}")


def import_reflectance_tab():
    """Reflectance/absorption import interface."""
    from services.importing import import_reflectance_absorption_from_csv, import_ecosis_csv
    
    st.subheader("Import Reflectance/Absorption Data")
    
    # File type selection
    file_type = st.radio(
        "Select file type:",
        ["Single Spectrum CSV", "ECOSIS Multi-Spectrum CSV"],
        help="Choose between importing a single spectrum or multiple spectra from ECOSIS database"
    )
    
    if file_type == "Single Spectrum CSV":
        st.info("Upload a CSV file with two columns: Wavelength, Reflectance/Absorption")
        
        uploaded_file = st.file_uploader(
            "Upload CSV File", 
            type="csv", 
            key="single_spectrum_upload"
        )
        
        if uploaded_file is not None:
            with st.form("single_spectrum_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    name = st.text_input("Spectrum Name", value="Custom Spectrum", max_chars=50)
                    data_type = st.selectbox("Data Type", ["Reflectance", "Absorption"])
                
                with col2:
                    category = st.selectbox("Category", ["Plant", "Other"])
                    description = st.text_area("Description (optional)", height=70)
                
                # Always extrapolate to full range for consistency
                extrap_lower = True
                extrap_upper = True
                
                submitted = st.form_submit_button("Import Single Spectrum", type="primary")
                
                if submitted:
                    if not name.strip():
                        st.error("Spectrum name is required")
                    else:
                        with st.spinner("Importing spectrum..."):
                            meta = {
                                "name": name.strip(),
                                "data_type": data_type,
                                "category": category,
                                "description": description.strip()
                            }
                            
                            success, message = import_reflectance_absorption_from_csv(
                                uploaded_file, meta, extrap_lower, extrap_upper
                            )
                            
                            if success:
                                st.success("Spectrum imported successfully!")
                                st.rerun()
                            else:
                                st.error(f"Import failed: {message}")
    
    elif file_type == "ECOSIS Multi-Spectrum CSV":
        st.info("Upload an ECOSIS database CSV file containing multiple samples with wavelength columns")
        
        ecosis_file = st.file_uploader(
            "Upload ECOSIS CSV File", 
            type="csv", 
            key="ecosis_upload",
            help="CSV files from the ECOSIS spectral library"
        )
        
        if ecosis_file is not None:
            # Get available metadata columns from the uploaded file
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(ecosis_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Import the helper function
            from services.importing import get_ecosis_csv_metadata_columns
            metadata_columns = get_ecosis_csv_metadata_columns(tmp_file_path)
            
            with st.form("ecosis_import_form"):
                st.write("**Import Settings**")
                
                # Column selection for naming/search
                name_column = None
                if metadata_columns:
                    st.write("**Name Column Selection:**")
                    name_column = st.selectbox(
                        "Choose column for spectrum names in dropdown menu:",
                        options=["None"] + metadata_columns,
                        index=0,
                        help="Select which CSV column should be used to name spectra in the application's dropdown menus. This will be stored as 'name_for_search' metadata."
                    )
                    if name_column == "None":
                        name_column = None
                
                # API URL input for enhanced metadata
                api_url = st.text_input(
                    "ECOSIS API URL (Optional)",
                    placeholder="https://ecosis.org/api/package/package-name",
                    help="Optional: Provide the ECOSIS API URL to include standardized metadata and attribution information in the output files. This ensures proper scientific attribution to the original dataset."
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Output Location:** Reflectors/Ecosis/{filename}")
                    st.write("**Processing:** Individual .tsv files per spectrum")
                
                with col2:
                    st.write("**Wavelength Range:** 300-1100nm (extrapolated)")
                    st.write("**Format:** Standard interpolated grid")
                    
                if api_url:
                    st.write("**Enhanced:** API metadata and attribution will be included")
                
                submitted = st.form_submit_button("Import ECOSIS File", type="primary")
                
                if submitted:
                    with st.spinner("Processing ECOSIS CSV file..."):
                        try:
                            # Save uploaded file temporarily (reuse the same temp file)
                            # tmp_file_path is already available from column detection above
                            
                            # Create Ecosis subfolder structure with CSV filename as subfolder
                            csv_filename = os.path.splitext(ecosis_file.name)[0]  # Remove .csv extension
                            csv_filename = ''.join(c for c in csv_filename if c.isalnum() or c in (' ', '_', '-')).rstrip().replace(' ', '_')
                            
                            output_dir = os.path.join("data", "reflectors", "Ecosis", csv_filename)
                            os.makedirs(output_dir, exist_ok=True)
                            
                            created_files = import_ecosis_csv(
                                tmp_file_path, 
                                output_dir, 
                                api_url.strip() if api_url.strip() else None,
                                name_column
                            )
                            
                            # Clean up temp file
                            os.unlink(tmp_file_path)

                            st.success(f"Successfully imported {len(created_files)} spectra from ECOSIS file!")

                            # Show imported file names in an expander
                            if len(created_files) > 0:
                                with st.expander("View imported files", expanded=True):
                                    for i, filepath in enumerate(created_files[:8]):
                                        filename = os.path.basename(filepath)
                                        st.write(f"- {filename}")

                                    if len(created_files) > 8:
                                        st.write(f"... and {len(created_files) - 8} more files")

                            st.rerun()

                        except Exception as e:
                            st.error(f"ECOSIS import failed: {str(e)}")
