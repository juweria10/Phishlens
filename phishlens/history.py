"""Scan history, kept per visitor in Streamlit's session state.

Each browser session gets its own list, so visitors on a shared deployment never see each other's
scans. The list lasts until the page is refreshed or the session times out. Use the CSV download
on the History page to keep a copy.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from .engine import ScanResult

KEY = "scan_history"
MAX_ROWS = 500
COLUMNS = ["time", "url", "host", "verdict", "p_phishing", "risk", "flags", "reliability", "source"]


def _rows() -> list[dict]:
    if KEY not in st.session_state:
        st.session_state[KEY] = []
    return st.session_state[KEY]


def add(result: ScanResult, source: str = "single") -> None:
    rows = _rows()
    rows.append({
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "url": result.url,
        "host": result.host,
        "verdict": result.verdict,
        "p_phishing": round(result.p_phishing, 4),
        "risk": result.risk,
        "flags": len(result.flags),
        "reliability": result.reliability,
        "source": source,
    })
    del rows[:-MAX_ROWS]


def load() -> pd.DataFrame:
    df = pd.DataFrame(_rows(), columns=COLUMNS)
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.iloc[::-1].reset_index(drop=True)  # newest first


def clear() -> None:
    st.session_state[KEY] = []
