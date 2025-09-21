"""
frontend_import_interface.py

Frontend tying together the .csv importers

Allows users to upload three types of spectral data generated via WebPlotDigitizer:
- Optical Filters
- Illuminants
- Quantum Efficiency (QE) curves

The UI dynamically adjusts based on the selected import type, collects relevant metadata,
and delegates CSV processing to the corresponding utility functions.

--- Features ---
- Accepts `.csv` uploads from WebPlotDigitizer.
- Prompts user for appropriate metadata based on data type.
- Supports optional extrapolation for filter data.
- Invokes backend parsers:
    - `import_filter_from_csv(...)`
    - `import_illuminant_from_csv(...)`
    - `import_qe_from_csv(...)`
- Displays success or error messages based on import result.

--- Dependencies ---
- Streamlit (for UI)
- Internal utilities from:
    - `import_filter.py`
    - `import_illuminant.py`
    - `import_quantum_efficiency.py`

--- Usage ---
This module is meant to be run inside a Streamlit app context.
It defines a single function `import_data()` that can be called within a Streamlit script
(e.g., inside `main.py`) to present the data import panel.

Example:
    import_interface.import_data()

"""


# /utils/importers/frontend_import_interface.py

import streamlit as st
from .import_filter import import_filter_from_csv
from .import_illuminant import import_illuminant_from_csv
from .import_quantum_efficiency import import_qe_from_csv
from .import_reflectance_absorption import import_reflectance_absorption_from_csv  

def import_data():
    st.header("🛠 Import Data")

    import_type = st.radio(
        "What type of data are you importing?",
        ["Filter", "Illuminant", "Quantum Efficiency", "Reflectance/Absorption"]  # added option
    )

    uploaded_file = st.file_uploader("Upload CSV (WebPlotDigitizer output)", type=["csv"])
    if not uploaded_file:
        return

    if import_type == "Filter":
        st.subheader("🧾 Filter Metadata")
        meta = {
            "filter_number": st.text_input("Filter Number"),
            "filter_name": st.text_input("Filter Name"),
            "manufacturer": st.text_input("Manufacturer"),
            "hex_color": st.text_input("Hex Color (e.g. #FF0000)", value="#1f77b4")
        }
        extrap_lower = st.checkbox("Extrapolate below to 300nm?", value=False)
        extrap_upper = st.checkbox("Extrapolate above to 1100nm?", value=True)

        if st.button("Import Filter"):
            success, msg = import_filter_from_csv(uploaded_file, meta, extrap_lower, extrap_upper)
            st.success(msg) if success else st.error(msg)

    elif import_type == "Illuminant":
        st.subheader("🧾 Illuminant Metadata")
        description = st.text_input("Illuminant Description")

        if st.button("Import Illuminant"):
            success, msg = import_illuminant_from_csv(uploaded_file, description)
            st.success(msg) if success else st.error(msg)

    elif import_type == "Quantum Efficiency":
        st.subheader("🧾 Camera Metadata")
        brand = st.text_input("Camera Brand")
        model = st.text_input("Camera Model")

        if st.button("Import QE"):
            success, msg = import_qe_from_csv(uploaded_file, brand, model)
            st.success(msg) if success else st.error(msg)

    elif import_type == "Reflectance/Absorption":
        st.subheader("🧾 Spectrum Metadata")
        spectrum_name = st.text_input("Spectrum Name")
        description = st.text_area("Description (optional)")
        spectrum_type = st.radio("Spectrum Type", ["Reflectance", "Absorption"])
        hex_color = st.text_input("Hex Color (e.g. #00FF00)", value="#1f77b4")  # <-- Added hex color input

        extrap_lower = st.checkbox("Extrapolate below to 300nm?", value=False)
        extrap_upper = st.checkbox("Extrapolate above to 1100nm?", value=True)

        if st.button("Import Spectrum"):
            meta = {
                "spectrum_name": spectrum_name.strip() or "Unnamed_Spectrum",
                "description": description.strip(),
                "spectrum_type": spectrum_type.lower(),
                "hex_color": hex_color.strip()  # <-- Include hex color in metadata
            }
            success, msg = import_reflectance_absorption_from_csv(uploaded_file, meta, extrap_lower, extrap_upper)
            st.success(msg) if success else st.error(msg)
        return                          
