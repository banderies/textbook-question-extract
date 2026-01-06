"""
State Management Module

Contains session state management and file persistence functions.
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime

import streamlit as st

from pdf_extraction import assign_chapters_to_images
from llm_extraction import DEFAULT_MODEL_NAME

# =============================================================================
# Path Constants
# =============================================================================

SOURCE_DIR = "source"
BASE_OUTPUT_DIR = "output"


# =============================================================================
# Dynamic Path Functions (per-PDF output directories)
# =============================================================================

def get_pdf_slug(pdf_name: str) -> str:
    """Convert PDF filename to a safe directory slug."""
    slug = Path(pdf_name).stem
    slug = re.sub(r'[^\w\-]', '_', slug)
    slug = re.sub(r'_+', '_', slug)
    return slug.strip('_')


def get_output_dir() -> str:
    """Get output directory for current PDF."""
    if "current_pdf" in st.session_state and st.session_state.current_pdf:
        slug = get_pdf_slug(st.session_state.current_pdf)
        return f"{BASE_OUTPUT_DIR}/{slug}"
    return BASE_OUTPUT_DIR


def get_images_dir() -> str:
    """Get images directory for current PDF."""
    return f"{get_output_dir()}/images"


def get_chapters_file() -> str:
    return f"{get_output_dir()}/chapters.json"


def get_chapter_text_file() -> str:
    return f"{get_output_dir()}/chapter_text.json"


def get_questions_file() -> str:
    return f"{get_output_dir()}/questions_by_chapter.json"


def get_questions_merged_file() -> str:
    return f"{get_output_dir()}/questions_merged.json"


def get_image_assignments_merged_file() -> str:
    return f"{get_output_dir()}/image_assignments_merged.json"


def get_images_file() -> str:
    return f"{get_output_dir()}/images.json"


def get_image_assignments_file() -> str:
    return f"{get_output_dir()}/image_assignments.json"


def get_qc_progress_file() -> str:
    return f"{get_output_dir()}/qc_progress.json"


def get_settings_file() -> str:
    return f"{get_output_dir()}/settings.json"


def get_pages_file() -> str:
    return f"{get_output_dir()}/pages.json"


def get_available_textbooks() -> list[str]:
    """Get list of textbooks that have output data."""
    textbooks = []
    if os.path.exists(BASE_OUTPUT_DIR):
        for item in os.listdir(BASE_OUTPUT_DIR):
            item_path = os.path.join(BASE_OUTPUT_DIR, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "chapters.json")):
                textbooks.append(item)
    return sorted(textbooks)


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize session state variables and auto-load saved data."""
    is_fresh_init = "initialized" not in st.session_state

    if "current_pdf" not in st.session_state:
        st.session_state.current_pdf = None
    if "pages" not in st.session_state:
        st.session_state.pages = None
    if "chapters" not in st.session_state:
        st.session_state.chapters = None
    if "chapter_texts" not in st.session_state:
        st.session_state.chapter_texts = {}
    if "questions" not in st.session_state:
        st.session_state.questions = {}
    if "questions_merged" not in st.session_state:
        st.session_state.questions_merged = {}
    if "images" not in st.session_state:
        st.session_state.images = []
    if "image_assignments" not in st.session_state:
        st.session_state.image_assignments = {}
    if "image_assignments_merged" not in st.session_state:
        st.session_state.image_assignments_merged = {}
    if "qc_progress" not in st.session_state:
        st.session_state.qc_progress = {"reviewed": {}, "corrections": {}, "metadata": {}}
    if "current_step" not in st.session_state:
        st.session_state.current_step = "source"
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = DEFAULT_MODEL_NAME
    if "qc_selected_idx" not in st.session_state:
        st.session_state.qc_selected_idx = 0

    if is_fresh_init:
        st.session_state.initialized = True


def clear_session_data():
    """Clear all session data for switching to a new PDF."""
    st.session_state.pages = None
    st.session_state.chapters = None
    st.session_state.chapter_texts = {}
    st.session_state.questions = {}
    st.session_state.questions_merged = {}
    st.session_state.images = []
    st.session_state.image_assignments = {}
    st.session_state.image_assignments_merged = {}
    st.session_state.qc_progress = {"reviewed": {}, "corrections": {}, "metadata": {}}
    st.session_state.qc_selected_idx = 0


# =============================================================================
# Load Functions
# =============================================================================

def load_qc_progress() -> dict:
    """Load QC progress from file."""
    qc_file = get_qc_progress_file()
    if os.path.exists(qc_file):
        with open(qc_file) as f:
            return json.load(f)
    return {"reviewed": {}, "corrections": {}, "metadata": {}}


def load_settings():
    """Load user settings from file."""
    settings_file = get_settings_file()
    if os.path.exists(settings_file):
        with open(settings_file) as f:
            settings = json.load(f)
            if "selected_model" in settings:
                st.session_state.selected_model = settings["selected_model"]
            if "current_step" in settings:
                st.session_state.current_step = settings["current_step"]
            if "qc_selected_idx" in settings:
                st.session_state.qc_selected_idx = settings["qc_selected_idx"]


def load_saved_data():
    """Load previously saved data if available."""
    pages_file = get_pages_file()
    chapters_file = get_chapters_file()
    chapter_text_file = get_chapter_text_file()
    questions_file = get_questions_file()
    questions_merged_file = get_questions_merged_file()
    images_file = get_images_file()
    assignments_file = get_image_assignments_file()
    assignments_merged_file = get_image_assignments_merged_file()

    if os.path.exists(pages_file):
        with open(pages_file) as f:
            st.session_state.pages = json.load(f)
    if os.path.exists(chapters_file):
        with open(chapters_file) as f:
            st.session_state.chapters = json.load(f)
    if os.path.exists(chapter_text_file):
        with open(chapter_text_file) as f:
            st.session_state.chapter_texts = json.load(f)
    if os.path.exists(questions_file):
        with open(questions_file) as f:
            st.session_state.questions = json.load(f)
    if os.path.exists(questions_merged_file):
        with open(questions_merged_file) as f:
            st.session_state.questions_merged = json.load(f)
    if os.path.exists(images_file):
        with open(images_file) as f:
            st.session_state.images = json.load(f)
    if os.path.exists(assignments_file):
        with open(assignments_file) as f:
            st.session_state.image_assignments = json.load(f)
    if os.path.exists(assignments_merged_file):
        with open(assignments_merged_file) as f:
            st.session_state.image_assignments_merged = json.load(f)

    # Assign chapters to images if chapters exist but images don't have chapter info
    if st.session_state.chapters and st.session_state.images:
        needs_chapter_update = any(
            img.get("chapter") is None for img in st.session_state.images
        )
        if needs_chapter_update:
            st.session_state.images = assign_chapters_to_images(
                st.session_state.images, st.session_state.chapters
            )
            save_images()


# =============================================================================
# Save Functions
# =============================================================================

def save_qc_progress():
    """Save QC progress to file."""
    st.session_state.qc_progress["metadata"]["last_saved"] = datetime.now().isoformat()
    os.makedirs(get_output_dir(), exist_ok=True)
    with open(get_qc_progress_file(), "w") as f:
        json.dump(st.session_state.qc_progress, f, indent=2)


def save_chapters():
    """Save chapters to file."""
    os.makedirs(get_output_dir(), exist_ok=True)
    with open(get_chapters_file(), "w") as f:
        json.dump(st.session_state.chapters, f, indent=2)
    with open(get_chapter_text_file(), "w") as f:
        json.dump(st.session_state.chapter_texts, f, indent=2)


def save_questions():
    """Save questions to file."""
    os.makedirs(get_output_dir(), exist_ok=True)
    with open(get_questions_file(), "w") as f:
        json.dump(st.session_state.questions, f, indent=2)


def save_images():
    """Save image metadata to file."""
    os.makedirs(get_output_dir(), exist_ok=True)
    with open(get_images_file(), "w") as f:
        json.dump(st.session_state.images, f, indent=2)


def save_pages():
    """Save raw PDF pages to file."""
    if st.session_state.pages:
        os.makedirs(get_output_dir(), exist_ok=True)
        with open(get_pages_file(), "w") as f:
            json.dump(st.session_state.pages, f, indent=2)


def save_image_assignments():
    """Save image-to-question assignments to file."""
    os.makedirs(get_output_dir(), exist_ok=True)
    with open(get_image_assignments_file(), "w") as f:
        json.dump(st.session_state.image_assignments, f, indent=2)


def save_questions_merged():
    """Save merged questions (with context associated) to file."""
    os.makedirs(get_output_dir(), exist_ok=True)
    with open(get_questions_merged_file(), "w") as f:
        json.dump(st.session_state.questions_merged, f, indent=2)


def save_image_assignments_merged():
    """Save merged image-to-question assignments to file."""
    os.makedirs(get_output_dir(), exist_ok=True)
    with open(get_image_assignments_merged_file(), "w") as f:
        json.dump(st.session_state.image_assignments_merged, f, indent=2)


def save_settings():
    """Save user settings to file."""
    os.makedirs(get_output_dir(), exist_ok=True)
    settings = {
        "selected_model": st.session_state.selected_model,
        "current_step": st.session_state.current_step,
        "qc_selected_idx": st.session_state.qc_selected_idx,
        "current_pdf": st.session_state.get("current_pdf", ""),
        "last_saved": datetime.now().isoformat()
    }
    with open(get_settings_file(), "w") as f:
        json.dump(settings, f, indent=2)
