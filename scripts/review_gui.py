#!/usr/bin/env python3
"""
Textbook Q&A Extractor - Web Interface

A Streamlit-based GUI for extracting Q&A pairs from textbooks using LLM.

Workflow:
1. Select source PDF
2. Extract chapters and preview text
3. Extract questions per chapter
4. QC questions with progress tracking
5. Export to Anki deck

Usage:
    streamlit run scripts/review_gui.py
"""

import streamlit as st

from state_management import init_session_state, save_settings
from ui_components import (
    render_sidebar,
    render_source_step,
    render_chapters_step,
    render_questions_step,
    render_format_step,
    render_qc_step,
    render_generate_step,
    render_export_step,
    render_prompts_step
)

# Page config
st.set_page_config(
    page_title="Textbook Q&A Extractor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to fix cursor on dropdowns
st.markdown("""
<style>
    /* Use pointer cursor for selectbox/dropdown elements */
    div[data-baseweb="select"] {
        cursor: pointer !important;
    }
    div[data-baseweb="select"] * {
        cursor: pointer !important;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()

    step = st.session_state.current_step

    if step == "source":
        render_source_step()
    elif step == "chapters":
        render_chapters_step()
    elif step == "questions":
        render_questions_step()
    elif step == "format":
        render_format_step()
    elif step == "qc":
        render_qc_step()
    elif step == "generate":
        render_generate_step()
    elif step == "export":
        render_export_step()
    elif step == "prompts":
        render_prompts_step()


if __name__ == "__main__":
    main()
