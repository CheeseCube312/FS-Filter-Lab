"""
file_utils.py

Provides utility for sanitizing strings to be used safely in filenames.

Functions:
- sanitize_filename_component: Cleans and optionally normalizes a string for use in filenames.
"""


import re

#sanitize names
def sanitize_filename_component(name: str, lowercase=False, max_len=None) -> str:
    clean = re.sub(r'[<>:"/\\|?*]', "-", name).strip()
    if lowercase:
        clean = clean.lower()
    return clean