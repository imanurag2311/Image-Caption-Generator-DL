"""
app.py
======
Main Streamlit application entry point for the Image Caption Generator.
Launches the optimized frontend communicating with the Flask REST API backend.

Usage:
    streamlit run app.py
    # or
    streamlit run frontend.py

For the standalone version without a backend (direct model loading in Streamlit):
    streamlit run app_standalone.py
"""

import os
import sys
import runpy

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if __name__ == "__main__":
    runpy.run_path(os.path.join(PROJECT_ROOT, "frontend.py"), run_name="__main__")
