"""
UI Helper Functions

Contains utility functions used across UI components.
"""

import os
import re
import streamlit as st

from state_management import get_output_dir, get_images_dir
from pdf_extraction import (
    extract_text_with_lines, insert_image_markers, build_chapter_text_with_lines
)
from llm_extraction import get_model_id


def play_completion_sound():
    """Play a pleasant notification sound when processing completes.

    Uses Web Audio API to generate a short, pleasant tone.
    """
    import streamlit.components.v1 as components

    # JavaScript to play a completion sound using Web Audio API
    # Creates a pleasant two-tone chime
    sound_js = """
    <script>
    (function() {
        try {
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

            // First tone (higher pitch)
            const osc1 = audioCtx.createOscillator();
            const gain1 = audioCtx.createGain();
            osc1.connect(gain1);
            gain1.connect(audioCtx.destination);
            osc1.frequency.value = 880; // A5
            osc1.type = 'sine';
            gain1.gain.setValueAtTime(0.3, audioCtx.currentTime);
            gain1.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.3);
            osc1.start(audioCtx.currentTime);
            osc1.stop(audioCtx.currentTime + 0.3);

            // Second tone (lower pitch, slightly delayed)
            const osc2 = audioCtx.createOscillator();
            const gain2 = audioCtx.createGain();
            osc2.connect(gain2);
            gain2.connect(audioCtx.destination);
            osc2.frequency.value = 1318.5; // E6
            osc2.type = 'sine';
            gain2.gain.setValueAtTime(0, audioCtx.currentTime + 0.1);
            gain2.gain.setValueAtTime(0.3, audioCtx.currentTime + 0.1);
            gain2.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.5);
            osc2.start(audioCtx.currentTime + 0.1);
            osc2.stop(audioCtx.currentTime + 0.5);
        } catch (e) {
            console.log('Audio playback not supported:', e);
        }
    })();
    </script>
    """
    components.html(sound_js, height=0, width=0)


def build_block_aware_image_assignments(formatted_questions: dict) -> dict:
    """
    Build image assignments from questions' image_files arrays.

    The LLM handles image distribution during formatting:
    - Context images are included in ALL sub-questions' image_files
    - Sub-question-specific images are only in that sub-question's image_files

    This function:
    - Builds the image_assignments dict (maps image filename -> question ID)
    - Sets context_from on non-first questions in each block for inheritance

    Args:
        formatted_questions: Dict of ch_key -> list of question dicts

    Returns:
        Dict mapping image filenames to question full_ids
    """
    image_assignments = {}

    for ch_key, qs in formatted_questions.items():
        # Group questions by block_id to set context_from
        block_questions = {}  # block_id -> list of question dicts

        for q in qs:
            block_id = q.get("block_id")

            # Assign each image to this question (first occurrence wins)
            for img_file in q.get("image_files", []):
                if img_file not in image_assignments:
                    image_assignments[img_file] = q["full_id"]

            # Group questions by block for context_from assignment
            if block_id:
                if block_id not in block_questions:
                    block_questions[block_id] = []
                block_questions[block_id].append(q)

        # Set context_from for non-first questions in each block
        for block_id, questions in block_questions.items():
            if len(questions) > 1:
                first_q_id = questions[0]["full_id"]
                for q in questions[1:]:
                    if not q.get("context_from"):
                        q["context_from"] = first_q_id

    return image_assignments


def get_selected_model_id() -> str:
    """Get the currently selected Claude model ID."""
    return get_model_id(st.session_state.selected_model)


def clear_step_data(step_id: str, cascade: bool = True):
    """Clear data for a specific step and all subsequent steps, allowing re-run from that point.

    Args:
        step_id: One of 'source', 'chapters', 'questions', 'format', 'qc', 'generate', 'export'
        cascade: If True, also clear all subsequent steps (default True)
    """
    # Define step order for cascading
    step_order = ["source", "chapters", "questions", "format", "qc", "generate", "export"]

    # Find the index of the current step
    try:
        start_idx = step_order.index(step_id)
    except ValueError:
        start_idx = 0

    # Determine which steps to clear
    steps_to_clear = step_order[start_idx:] if cascade else [step_id]

    output_dir = get_output_dir()

    for step in steps_to_clear:
        if step == "source":
            # Clear pages, images, extracted text
            st.session_state.pages = None
            st.session_state.images = []
            st.session_state.pdf_path = None
            if "extracted_text" in st.session_state:
                del st.session_state.extracted_text
            # Delete files
            for filename in ["pages.json", "images.json", "extracted_text.txt"]:
                filepath = os.path.join(output_dir, filename)
                if os.path.exists(filepath):
                    os.remove(filepath)
            # Delete images directory
            images_dir = get_images_dir()
            if os.path.exists(images_dir):
                import shutil
                shutil.rmtree(images_dir)

        elif step == "chapters":
            # Clear chapters and chapter texts
            st.session_state.chapters = None
            st.session_state.chapter_texts = {}
            # Delete files
            for filename in ["chapters.json", "chapter_text.json"]:
                filepath = os.path.join(output_dir, filename)
                if os.path.exists(filepath):
                    os.remove(filepath)

        elif step == "questions":
            # Clear raw questions
            st.session_state.raw_questions = {}
            filepath = os.path.join(output_dir, "raw_questions.json")
            if os.path.exists(filepath):
                os.remove(filepath)

        elif step == "format":
            # Clear formatted questions and image assignments
            st.session_state.questions = {}
            st.session_state.image_assignments = {}
            for filename in ["questions_by_chapter.json", "image_assignments.json"]:
                filepath = os.path.join(output_dir, filename)
                if os.path.exists(filepath):
                    os.remove(filepath)

        elif step == "qc":
            # Clear QC progress
            st.session_state.qc_progress = {"reviewed": {}, "corrections": {}, "metadata": {}}
            st.session_state.qc_selected_idx = 0
            filepath = os.path.join(output_dir, "qc_progress.json")
            if os.path.exists(filepath):
                os.remove(filepath)

        elif step == "generate":
            # Clear generated cloze questions
            st.session_state.generated_questions = {"metadata": {}, "generated_cards": {}}
            filepath = os.path.join(output_dir, "generated_questions.json")
            if os.path.exists(filepath):
                os.remove(filepath)

        elif step == "export":
            # Delete any exported .apkg files
            if os.path.exists(output_dir):
                for filename in os.listdir(output_dir):
                    if filename.endswith(".apkg"):
                        os.remove(os.path.join(output_dir, filename))


def sort_chapter_keys(keys: list) -> list:
    """Sort chapter keys naturally (ch1, ch2, ..., ch10, ch11, not ch1, ch10, ch11, ch2).

    Handles keys like 'ch1', 'ch2', 'ch10', 'ch20' and sorts them numerically.
    """
    def extract_chapter_num(key: str) -> int:
        match = re.match(r'ch(\d+)', key)
        return int(match.group(1)) if match else 0
    return sorted(keys, key=extract_chapter_num)


def prepare_chapter_for_two_pass(ch_num: int, chapters: list[dict]) -> tuple[str, list[str]]:
    """
    Prepare line-numbered chapter text with inline image markers for two-pass extraction.

    Args:
        ch_num: Chapter number
        chapters: List of all chapters (to determine page ranges)

    Returns:
        Tuple of (line_numbered_text, lines_with_images)
        Returns (None, None) if preparation fails
    """
    pdf_path = st.session_state.get("source_path")
    if not pdf_path or not os.path.exists(pdf_path):
        return None, None

    # Find chapter page range
    ch_idx = next((i for i, ch in enumerate(chapters) if ch["chapter_number"] == ch_num), None)
    if ch_idx is None:
        return None, None

    ch = chapters[ch_idx]
    start_page = ch["start_page"]
    end_page = chapters[ch_idx + 1]["start_page"] if ch_idx + 1 < len(chapters) else None

    # Extract text with lines
    pages, all_lines = extract_text_with_lines(pdf_path)

    # Get chapter images
    ch_images = [img for img in st.session_state.images
                 if start_page <= img["page"] < (end_page or 9999)]

    # Insert image markers
    lines_with_images = insert_image_markers(all_lines, ch_images, pages)

    # Build line-numbered chapter text
    chapter_text, _ = build_chapter_text_with_lines(
        lines_with_images, pages, start_page, end_page
    )

    return chapter_text, lines_with_images


def question_sort_key(q_id: str) -> tuple:
    """Sort key for question IDs."""
    match = re.match(r'ch(\d+)_(\d+)([a-z]?)', q_id, re.IGNORECASE)
    if match:
        return (int(match.group(1)), int(match.group(2)), match.group(3) or "")
    return (999, 999, q_id)


def get_images_for_question(q_id: str) -> list[dict]:
    """
    Get question images (from image_files array).

    Prefers questions over questions_merged to use the freshest data
    (in case formatting was re-run).
    """
    # Search questions first (fresher data), then questions_merged as fallback
    questions_to_check = []
    for _, qs in st.session_state.questions.items():
        questions_to_check.extend(qs)
    for _, qs in st.session_state.questions_merged.items():
        questions_to_check.extend(qs)

    for q in questions_to_check:
        if q["full_id"] == q_id:
            q_image_files = set(q.get("image_files", []))
            return [img for img in st.session_state.images if img["filename"] in q_image_files]

    return []


def get_answer_images_for_question(q_id: str) -> list[dict]:
    """
    Get answer/explanation images (from answer_image_files array).

    These images should only appear in the explanation section, not with the question.
    """
    questions_to_check = []
    for _, qs in st.session_state.questions.items():
        questions_to_check.extend(qs)
    for _, qs in st.session_state.questions_merged.items():
        questions_to_check.extend(qs)

    for q in questions_to_check:
        if q["full_id"] == q_id:
            answer_image_files = set(q.get("answer_image_files", []))
            return [img for img in st.session_state.images if img["filename"] in answer_image_files]

    return []


def get_all_question_options() -> list[str]:
    """Get list of all question IDs for reassignment dropdown."""
    options = ["(none)"]
    for ch_key in sorted(st.session_state.questions.keys(), key=lambda x: int(x[2:]) if x[2:].isdigit() else 0):
        for q in st.session_state.questions[ch_key]:
            options.append(q["full_id"])
    return options
