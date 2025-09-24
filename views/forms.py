"""
Form-based UI components for FS FilterLab.

This module provides UI components for forms, including search and import interfaces.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from views.ui_utils import is_dark_color, is_valid_hex_color, handle_error
from models.constants import INTERP_GRID
from services.visualization import create_sparkline_plot


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
    i = idx[0]

    # Make sure we're filtering the correct rows in the matrix
    df_indices = df.index.to_numpy()
    vals = matrix[df_indices, i]

    mask = (vals >= min_t) & (vals <= max_t)

    return df.iloc[mask], vals[mask]


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
        st.warning(f"⚠ Found {len(invalid_rows)} filters with invalid hex color codes:")
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
    Render advanced filter search UI.
    
    Args:
        df: DataFrame of filters
        filter_matrix: Matrix of filter transmission values
    """
    if not st.session_state.get("advanced", False):
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
            apply_clicked = st.form_submit_button("🔄 Apply")

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

    df_man = filter_by_manufacturer(df, manufs)
    df_filt, trans_vals = filter_by_trans_at_wavelength(
        df_man, INTERP_GRID, filter_matrix, wl, tmin / 100, tmax / 100
    )

    if sort_choice == "Hex‑Rainbow":
        df_sorted = sort_by_hex_rainbow(df_filt)
    elif sort_choice.startswith("Trans @"):
        df_sorted = sort_by_trans_at_wavelength(df_filt, trans_vals)
    elif sort_choice == "Filter Name":
        df_sorted = df_filt.sort_values("Filter Name")
    else:
        df_sorted = df_filt.sort_values("Filter Number")

    st.markdown("---")
    st.write(f"**{len(df_sorted)} filters found:**")

    for idx, row in df_sorted.iterrows():
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
                fig = create_sparkline_plot(INTERP_GRID, filter_matrix[idx, :], line_color=hex_color)
                st.plotly_chart(fig, use_container_width=False)
                st.checkbox("Select this filter", key=f"adv_sel_{idx}")

    st.markdown("---")
    col_done, col_cancel = st.columns([1, 1])
    with col_done:
        if st.button("✅ Done"):
            selected_idxs = [
                idx for idx in df_sorted.index
                if st.session_state.get(f"adv_sel_{idx}", False)
            ]
            selected_display = [
                f"{df_sorted.loc[idx, 'Filter Name']} "
                f"({df_sorted.loc[idx, 'Filter Number']}, "
                f"{df_sorted.loc[idx, 'Manufacturer']})"
                for idx in selected_idxs
            ]

            st.session_state["_pending_selected_filters"] = selected_display
            st.session_state.advanced = False
            st.rerun()

    for idx in df_sorted.index:
        st.session_state.pop(f"adv_sel_{idx}", None)
        st.session_state.pop(f"filter_toggle_{idx}", None)

    with col_cancel:
        if st.button("✖ Cancel"):
            st.session_state.advanced = False
            st.rerun()


# -- Import UI -----------------------------------------

def import_data_form() -> None:
    """
    Display data import UI with all import options.
    """
    st.markdown("---")
    st.header("Import Data")
    
    # Create tabs for different import types
    tab1, tab2, tab3, tab4 = st.tabs(["Filters", "Illuminants", "Camera QE", "Reflectance"])
    
    with tab1:
        import_filter_tab()
    
    with tab2:
        import_illuminant_tab()
    
    with tab3:
        import_qe_tab()
    
    with tab4:
        import_reflectance_tab()


def import_filter_tab():
    """Filter import interface."""
    from services.importing import import_filter_from_csv
    
    st.subheader("Import Filter Data")
    st.write("Upload CSV files with Wavelength (nm) and Transmittance columns.")
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv", key="filter_upload")
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            manufacturer = st.text_input("Manufacturer", value="Custom")
            filter_name = st.text_input("Filter Name", value="Custom Filter")
        
        with col2:
            filter_number = st.text_input("Filter Number", value="001")
            hex_color = st.color_picker("Filter Color", value="#808080")
        
        col3, col4 = st.columns(2)
        with col3:
            extrap_lower = st.checkbox("Extrapolate to 300nm", value=False)
        with col4:
            extrap_upper = st.checkbox("Extrapolate to 1100nm", value=False)
        
        if st.button("Import Filter", type="primary"):
            meta = {
                "manufacturer": manufacturer,
                "filter_name": filter_name,
                "filter_number": filter_number,
                "hex_color": hex_color
            }
            
            success, message = import_filter_from_csv(uploaded_file, meta, extrap_lower, extrap_upper)
            
            if success:
                st.success(message)
                st.info("Please refresh the page to see the new filter in the list.")
            else:
                handle_error(message)


def import_illuminant_tab():
    """Illuminant import interface."""
    from services.importing import import_illuminant_from_csv
    
    st.subheader("Import Illuminant Data")
    st.write("Upload CSV files with Wavelength (nm) and Relative Power columns.")
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv", key="illuminant_upload")
    
    if uploaded_file is not None:
        description = st.text_input("Description", value="Custom Illuminant")
        
        if st.button("Import Illuminant", type="primary"):
            success, message = import_illuminant_from_csv(uploaded_file, description)
            
            if success:
                st.success(message)
                st.info("Please refresh the page to see the new illuminant in the list.")
            else:
                handle_error(message)


def import_qe_tab():
    """Camera QE import interface."""
    from services.importing import import_qe_from_csv
    
    st.subheader("Import Camera QE Data")
    st.write("Upload CSV files with Wavelength (nm), R, G, B columns.")
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv", key="qe_upload")
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            brand = st.text_input("Camera Brand", value="Custom")
        with col2:
            model = st.text_input("Camera Model", value="Custom Model")
        
        if st.button("Import Camera QE", type="primary"):
            success, message = import_qe_from_csv(uploaded_file, brand, model)
            
            if success:
                st.success(message)
                st.info("Please refresh the page to see the new camera in the list.")
            else:
                handle_error(message)


def import_reflectance_tab():
    """Reflectance/absorption import interface."""
    from services.importing import import_reflectance_absorption_from_csv
    
    st.subheader("Import Reflectance/Absorption Data")
    st.write("Upload CSV files with Wavelength (nm) and Reflectance/Absorption columns.")
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv", key="reflectance_upload")
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Spectrum Name", value="Custom Spectrum")
            data_type = st.selectbox("Data Type", ["Reflectance", "Absorption"])
        
        with col2:
            category = st.selectbox("Category", ["Plant", "Other"])
            description = st.text_area("Description", value="")
        
        col3, col4 = st.columns(2)
        with col3:
            extrap_lower = st.checkbox("Extrapolate to 300nm", value=False, key="refl_extrap_lower")
        with col4:
            extrap_upper = st.checkbox("Extrapolate to 1100nm", value=False, key="refl_extrap_upper")
        
        if st.button("Import Spectrum", type="primary"):
            meta = {
                "name": name,
                "data_type": data_type,
                "category": category,
                "description": description
            }
            
            success, message = import_reflectance_absorption_from_csv(
                uploaded_file, meta, extrap_lower, extrap_upper
            )
            
            if success:
                st.success(message)
                st.info("Please refresh the page to see the new spectrum in the reflector preview.")
            else:
                handle_error(message)
