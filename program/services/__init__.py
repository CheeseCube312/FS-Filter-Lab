"""
Services package for FS FilterLab.

Core business logic and processing services:

- calculations: Mathematical computations for transmission and color analysis
- data: Data loading and caching services with automatic validation
- importing: File import utilities for various spectral data formats
- presenter: Read-only view-model builders bridging services and the UI layer
- reporting: PNG report generation and filename sanitization
- spectra: Pure spectral-physics core (combine, integrate, WB gains)
- state_manager: Plain-dataclass application state
- visualization: Plotly chart generation for UI views

All imports go directly to submodules (e.g., ``from services.calculations import ...``).
"""
