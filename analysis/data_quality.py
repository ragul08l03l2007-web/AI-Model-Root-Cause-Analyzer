# analysis/data_quality.py
"""
Data quality audit engine. Provides backward-compatible interface and delegates
to DataProfiler for universal dataset profiling.
"""

from analysis.evidence import safe_primitive
from analysis.profiler import DataProfiler


def _safe_val(value):
    """Convert any NumPy/pandas scalar to native Python types."""
    return safe_primitive(value)


def analyze_data_quality(df):
    """
    Perform a comprehensive, fully data-driven quality and structural audit
    of any uploaded tabular dataset.
    """
    return DataProfiler.profile_dataset(df)