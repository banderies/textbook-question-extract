"""
UI Components Module

Contains all Streamlit UI rendering functions.
"""

import os
import re
import copy
import json
from pathlib import Path
from datetime import datetime
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st

from state_management import (
    SOURCE_DIR, get_pdf_slug, get_output_dir, get_images_dir,
    get_available_textbooks, clear_session_data, load_saved_data,
    load_settings, load_qc_progress, save_settings, save_chapters,
    save_questions, save_raw_questions, save_images, save_pages, save_image_assignments,
    save_questions_merged, save_image_assignments_merged, save_qc_progress,
    get_raw_questions_file, get_questions_file
)
from pdf_extraction import (
    extract_images_from_pdf, assign_chapters_to_images,
    extract_chapter_text, render_pdf_page,
    extract_text_with_lines, insert_image_markers, build_chapter_text_with_lines,
    extract_lines_by_range, extract_lines_by_range_mapped
)
from llm_extraction import (
    get_anthropic_client, get_model_options, get_model_id,
    identify_chapters_llm, extract_qa_pairs_llm, process_chapter_extraction,
    extract_line_ranges_llm, format_qa_pair_llm,
    match_images_to_questions_llm, associate_context_llm, add_page_numbers_to_questions,
    load_prompts, save_prompts, reload_prompts,
    get_extraction_logger, get_log_file_path, reset_logger
)


# =============================================================================
# Helper Functions
# =============================================================================

def get_selected_model_id() -> str:
    """Get the currently selected Claude model ID."""
    return get_model_id(st.session_state.selected_model)


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
    Get all images for a question, including inherited images from context_from or image_group.
    """
    images = []
    directly_assigned = set()

    # First, get directly assigned images
    for img in st.session_state.images:
        assigned_to = st.session_state.image_assignments.get(img["filename"])
        if assigned_to == q_id:
            images.append(img)
            directly_assigned.add(img["filename"])

    if images:
        return images

    # Check for inherited images (context_from or image_group)
    # First check merged questions, then regular questions
    questions_to_check = []
    for ch_key, qs in st.session_state.questions_merged.items():
        questions_to_check.extend(qs)
    for ch_key, qs in st.session_state.questions.items():
        questions_to_check.extend(qs)

    for q in questions_to_check:
        if q["full_id"] == q_id:
            # Check context_from first (for merged context)
            context_from = q.get("context_from")
            if context_from:
                for img in st.session_state.images:
                    assigned_to = st.session_state.image_assignments.get(img["filename"])
                    if assigned_to == context_from and img["filename"] not in directly_assigned:
                        images.append(img)
                if images:
                    return images

            # Check image_group (for shared images within a group)
            image_group = q.get("image_group")
            if image_group:
                ch_prefix = q_id.split("_")[0]
                base_id = f"{ch_prefix}_{image_group}"

                for img in st.session_state.images:
                    assigned_to = st.session_state.image_assignments.get(img["filename"])
                    if assigned_to == base_id and img["filename"] not in directly_assigned:
                        images.append(img)

            return images

    return images


def get_all_question_options() -> list[str]:
    """Get list of all question IDs for reassignment dropdown."""
    options = ["(none)"]
    for ch_key in sorted(st.session_state.questions.keys(), key=lambda x: int(x[2:]) if x[2:].isdigit() else 0):
        for q in st.session_state.questions[ch_key]:
            options.append(q["full_id"])
    return options


# =============================================================================
# Sidebar
# =============================================================================

def render_sidebar():
    """Render sidebar with navigation."""
    st.sidebar.title("Textbook Q&A Extractor")

    if st.session_state.current_pdf:
        textbook_name = get_pdf_slug(st.session_state.current_pdf).replace('_', ' ')
        st.sidebar.caption(f"Working on: **{textbook_name}**")

    st.sidebar.markdown("---")

    steps = [
        ("source", "1. Select Source"),
        ("chapters", "2. Extract Chapters"),
        ("questions", "3. Extract Questions"),
        ("format", "4. Format Questions"),
        ("context", "5. Associate Context"),
        ("qc", "6. QC Questions"),
        ("export", "7. Export"),
        ("prompts", "8. Edit Prompts")
    ]

    for step_id, step_name in steps:
        if st.sidebar.button(step_name, key=f"nav_{step_id}"):
            st.session_state.current_step = step_id
            save_settings()

    st.sidebar.markdown("---")

    # Status summary - Order: Chapters, Images, Raw Questions, Formatted, Context, QC
    st.sidebar.subheader("Status")
    if st.session_state.chapters:
        st.sidebar.success(f"Chapters: {len(st.session_state.chapters)}")
    else:
        st.sidebar.info("Chapters: Not extracted")

    img_count = len(st.session_state.images)
    if img_count > 0:
        assigned = len(st.session_state.image_assignments)
        st.sidebar.success(f"Images: {img_count} ({assigned} assigned)")
    else:
        st.sidebar.info("Images: Not extracted")

    raw_q_count = sum(len(qs) for qs in st.session_state.get("raw_questions", {}).values())
    if raw_q_count > 0:
        st.sidebar.success(f"Raw Q&A: {raw_q_count}")
    else:
        st.sidebar.info("Raw Q&A: Not extracted")

    q_count = sum(len(qs) for qs in st.session_state.questions.values())
    if q_count > 0:
        st.sidebar.success(f"Formatted: {q_count}")
    else:
        st.sidebar.info("Formatted: Not done")

    merged_count = sum(len(qs) for qs in st.session_state.questions_merged.values())
    if merged_count > 0:
        st.sidebar.success(f"Context: Associated")
    else:
        st.sidebar.info("Context: Not associated")

    reviewed = len(st.session_state.qc_progress.get("reviewed", {}))
    if reviewed > 0:
        st.sidebar.success(f"QC'd: {reviewed}/{q_count}")


# =============================================================================
# Step 1: Source Selection
# =============================================================================

def render_source_step():
    """Render source PDF selection step."""
    st.header("Step 1: Select Source PDF")

    pdf_files = list(Path(SOURCE_DIR).glob("*.pdf")) if os.path.exists(SOURCE_DIR) else []

    if not pdf_files:
        st.warning(f"No PDF files found in '{SOURCE_DIR}/' directory. Please add a PDF file.")
        return

    pdf_options = [f.name for f in pdf_files]
    available_textbooks = get_available_textbooks()

    if available_textbooks:
        st.subheader("Load Existing Textbook")
        st.caption("These textbooks have saved progress:")

        textbook_col1, textbook_col2 = st.columns([3, 1])
        with textbook_col1:
            selected_textbook = st.selectbox(
                "Select saved textbook:",
                available_textbooks,
                format_func=lambda x: x.replace('_', ' '),
                key="textbook_selector"
            )
        with textbook_col2:
            if st.button("Load Textbook", type="primary"):
                for pdf_name in pdf_options:
                    if get_pdf_slug(pdf_name) == selected_textbook:
                        st.session_state.current_pdf = pdf_name
                        break
                else:
                    st.session_state.current_pdf = selected_textbook + ".pdf"

                clear_session_data()
                reset_logger()  # Reset logger for new textbook
                load_saved_data()
                load_settings()
                st.session_state.qc_progress = load_qc_progress()
                st.success(f"Loaded: {selected_textbook}")
                st.rerun()

        st.markdown("---")
        st.subheader("Start New Textbook")

    selected_pdf = st.selectbox("Select PDF file:", pdf_options)

    if selected_pdf:
        pdf_path = f"{SOURCE_DIR}/{selected_pdf}"
        output_slug = get_pdf_slug(selected_pdf)
        st.info(f"Selected: {pdf_path}")
        st.caption(f"Output folder: output/{output_slug}/")

        has_existing_data = output_slug in available_textbooks

        col1, col2 = st.columns(2)

        with col1:
            btn_label = "Load PDF (Fresh Start)" if has_existing_data else "Load PDF"
            if st.button(btn_label, type="primary"):
                st.session_state.current_pdf = selected_pdf
                clear_session_data()
                reset_logger()  # Reset logger for new PDF

                with st.spinner("Extracting images from PDF..."):
                    st.session_state.images = extract_images_from_pdf(pdf_path, get_images_dir())
                    save_images()

                with st.spinner("Extracting text with image markers..."):
                    # Extract text with positions for accurate image marker placement
                    pages, all_lines = extract_text_with_lines(pdf_path)
                    st.session_state.pages = pages
                    st.session_state.pdf_path = pdf_path
                    save_pages()

                    # Insert image markers at correct positions
                    lines_with_markers = insert_image_markers(all_lines, st.session_state.images, pages)

                    # Build the full document text with line numbers
                    numbered_lines = []
                    line_num = 1
                    for line in lines_with_markers:
                        if line.startswith("[IMAGE:"):
                            numbered_lines.append(line)
                        else:
                            numbered_lines.append(f"[LINE:{line_num:06d}] {line}")
                            line_num += 1

                    full_text = "\n".join(numbered_lines)
                    st.session_state.extracted_text = full_text

                    # Save to file
                    output_dir = get_output_dir()
                    text_file_path = os.path.join(output_dir, "extracted_text.txt")
                    with open(text_file_path, "w", encoding="utf-8") as f:
                        f.write(full_text)

                st.success(f"Loaded {len(st.session_state.pages)} pages, {len(st.session_state.images)} images")
                st.info(f"Saved extracted text to: {text_file_path}")
                st.rerun()

        with col2:
            if has_existing_data:
                if st.button("Load Existing Progress"):
                    st.session_state.current_pdf = selected_pdf
                    clear_session_data()
                    reset_logger()  # Reset logger for loaded textbook
                    load_saved_data()
                    load_settings()
                    st.session_state.qc_progress = load_qc_progress()

                    # Load extracted text from file if it exists
                    text_file_path = os.path.join(get_output_dir(), "extracted_text.txt")
                    if os.path.exists(text_file_path):
                        with open(text_file_path, "r", encoding="utf-8") as f:
                            st.session_state.extracted_text = f.read()

                    st.success("Loaded previous session data")
                    st.rerun()

        if st.session_state.pages:
            st.success(f"PDF loaded: {len(st.session_state.pages)} pages, {len(st.session_state.images)} images")
            st.markdown("**Next:** Go to 'Extract Chapters' to identify chapter boundaries.")

            # Full extracted text with image markers
            st.markdown("---")
            st.subheader("Extracted Text with Image Markers")

            extracted_text = st.session_state.get("extracted_text")
            if extracted_text:
                # Show stats
                line_count = extracted_text.count("[LINE:")
                image_count = extracted_text.count("[IMAGE:")
                st.caption(f"{line_count} text lines, {image_count} image markers")

                # Show file path
                text_file_path = os.path.join(get_output_dir(), "extracted_text.txt")
                st.caption(f"File: `{text_file_path}`")

                # Full scrollable text area
                st.text_area(
                    "Full document (scroll to view):",
                    extracted_text,
                    height=600,
                    key="full_extracted_text"
                )
            else:
                st.warning("Extracted text not available. Try re-loading the PDF.")


# =============================================================================
# Step 2: Extract Chapters
# =============================================================================

def render_chapters_step():
    """Render chapter extraction step."""
    st.header("Step 2: Extract Chapters")

    has_pages = st.session_state.pages is not None
    has_chapters = st.session_state.chapters is not None

    if not has_pages and not has_chapters:
        st.warning("Please load a PDF first (Step 1)")
        return

    client = get_anthropic_client()
    if not client:
        st.error("ANTHROPIC_API_KEY not set. Please configure your .env file.")
        return

    if has_pages:
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            model_options = get_model_options()
            current_idx = model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
            selected_model = st.selectbox("Model:", model_options, index=current_idx, key="chapters_model")
            if selected_model != st.session_state.selected_model:
                st.session_state.selected_model = selected_model
                save_settings()

        with col2:
            btn_label = "Re-extract Chapters" if has_chapters else "Extract Chapters"
            if st.button(btn_label, type="primary"):
                with st.spinner(f"Using {st.session_state.selected_model} to identify chapters..."):
                    chapters = identify_chapters_llm(client, st.session_state.pages, get_selected_model_id())
                    st.session_state.chapters = chapters

                    for i, ch in enumerate(chapters):
                        start_page = ch["start_page"]
                        end_page = chapters[i + 1]["start_page"] if i + 1 < len(chapters) else None
                        ch_key = f"ch{ch['chapter_number']}"
                        st.session_state.chapter_texts[ch_key] = extract_chapter_text(
                            st.session_state.pages, start_page, end_page
                        )

                    save_chapters()

                    if st.session_state.images:
                        st.session_state.images = assign_chapters_to_images(
                            st.session_state.images, chapters
                        )
                        save_images()

                st.success(f"Found {len(chapters)} chapters")
                st.rerun()
    elif has_chapters:
        st.info("Chapters already extracted. Go to Step 1 to reload PDF if you need to re-extract.")

    if st.session_state.chapters:
        st.subheader("Extracted Chapters")

        for ch in st.session_state.chapters:
            st.markdown(f"**Chapter {ch['chapter_number']}:** {ch['title']} (page {ch['start_page']})")

        st.markdown("---")

        st.subheader("Preview Chapter Text")

        chapter_options = [f"Ch{ch['chapter_number']}: {ch['title'][:40]}..."
                         for ch in st.session_state.chapters]
        selected_ch_idx = st.selectbox("Select chapter to preview:",
                                        range(len(chapter_options)),
                                        format_func=lambda x: chapter_options[x])

        if selected_ch_idx is not None:
            ch = st.session_state.chapters[selected_ch_idx]
            ch_key = f"ch{ch['chapter_number']}"
            ch_text = st.session_state.chapter_texts.get(ch_key, "")

            st.text_area("Chapter text:", ch_text, height=400, disabled=True)
            st.caption(f"Text length: {len(ch_text):,} characters")


# =============================================================================
# Step 3: Extract Questions (Raw)
# =============================================================================

def render_questions_step():
    """Render raw question extraction step - identifies line ranges and extracts raw text."""
    st.header("Step 3: Extract Questions")

    st.markdown("""
    This step identifies question and answer boundaries in the text and extracts raw Q&A pairs.
    The LLM identifies line ranges for each question/answer, then the raw text is extracted.
    """)

    if not st.session_state.chapters:
        st.warning("Please extract chapters first (Step 2)")
        return

    client = get_anthropic_client()
    if not client:
        st.error("ANTHROPIC_API_KEY not set. Please configure your .env file.")
        return

    # Initialize logger
    output_dir = get_output_dir()

    # Ensure raw_questions is loaded from file if session state is empty
    # This prevents data loss on session reset
    raw_questions_file = get_raw_questions_file()
    if not st.session_state.raw_questions and os.path.exists(raw_questions_file):
        with open(raw_questions_file, "r") as f:
            st.session_state.raw_questions = json.load(f)

    logger = get_extraction_logger(output_dir)

    # Chapter selector
    chapter_options = [f"Ch{ch['chapter_number']}: {ch['title'][:40]}..."
                      for ch in st.session_state.chapters]
    selected_ch_idx = st.selectbox("Select chapter:",
                                    range(len(chapter_options)),
                                    format_func=lambda x: chapter_options[x],
                                    key="extract_ch_selector")

    ch = st.session_state.chapters[selected_ch_idx]
    ch_num = ch["chapter_number"]
    ch_key = f"ch{ch_num}"

    # Model selection, workers, and buttons
    model_col, workers_col, btn_col1, btn_col2 = st.columns([2, 1.5, 2, 2])

    with model_col:
        model_options = get_model_options()
        current_idx = model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
        selected_model = st.selectbox("Model:", model_options, index=current_idx, key="questions_model")
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            save_settings()

    with workers_col:
        extract_workers = st.number_input(
            "Parallel workers:",
            min_value=1,
            max_value=50,
            value=10,
            help="Number of chapters to extract in parallel. Tier 1: 5-10 | Tier 2+: 10-20",
            key="extract_workers"
        )

    # Helper function to extract a single chapter
    def extract_single_chapter(ch_idx: int, pages_with_lines: list, lines_with_images: list, on_progress=None):
        ch = st.session_state.chapters[ch_idx]
        ch_num = ch["chapter_number"]
        ch_key = f"ch{ch_num}"
        model_id = get_selected_model_id()

        start_page = ch["start_page"]
        end_page = st.session_state.chapters[ch_idx + 1]["start_page"] if ch_idx + 1 < len(st.session_state.chapters) else None
        ch_text, line_mapping = build_chapter_text_with_lines(
            lines_with_images, pages_with_lines, start_page, end_page
        )

        line_ranges = extract_line_ranges_llm(get_anthropic_client(), ch_num, ch_text, model_id, on_progress=on_progress)

        if not line_ranges:
            logger.warning(f"Chapter {ch_num}: No line ranges extracted")
            return []

        raw_questions = []
        for lr in line_ranges:
            q_id = lr.get("question_id", "?")
            q_start = lr.get("question_start", 0)
            q_end = lr.get("question_end", 0)
            a_start = lr.get("answer_start", 0)
            a_end = lr.get("answer_end", 0)

            q_text = extract_lines_by_range_mapped(lines_with_images, q_start, q_end, line_mapping) if q_start > 0 else ""
            a_text = extract_lines_by_range_mapped(lines_with_images, a_start, a_end, line_mapping) if a_start > 0 else ""

            raw_questions.append({
                "full_id": f"ch{ch_num}_{q_id}",
                "local_id": q_id,
                "chapter": ch_num,
                "question_start": q_start,
                "question_end": q_end,
                "answer_start": a_start,
                "answer_end": a_end,
                "correct_letter": lr.get("correct_letter", ""),
                "image_files": lr.get("image_files", []),
                "question_text": q_text,
                "answer_text": a_text
            })

        return raw_questions

    with btn_col1:
        if st.button(f"Extract Chapter {ch_num}", type="primary"):
            progress_text = st.empty()
            progress_text.text(f"Extracting Chapter {ch_num}... Preparing text...")

            pdf_path = st.session_state.get("pdf_path")
            if not pdf_path or not os.path.exists(pdf_path):
                st.error("PDF path not found. Please reload the PDF in Step 1.")
            else:
                pages_with_lines, all_lines = extract_text_with_lines(pdf_path)
                lines_with_images = insert_image_markers(
                    all_lines, st.session_state.images, pages_with_lines
                )

                progress_text.text(f"Extracting Chapter {ch_num}... Streaming response...")

                # Progress callback for streaming
                def update_progress(tokens, text):
                    # Count questions found so far by counting question_id occurrences
                    q_count = text.count('"question_id"')
                    progress_text.text(f"Extracting Chapter {ch_num}... {tokens} tokens, ~{q_count} questions found")

                raw_questions = extract_single_chapter(selected_ch_idx, pages_with_lines, lines_with_images, update_progress)

                if raw_questions:
                    st.session_state.raw_questions[ch_key] = raw_questions
                    save_raw_questions()
                    progress_text.empty()
                    st.success(f"Extracted {len(raw_questions)} raw Q&A pairs from Chapter {ch_num}")
                else:
                    progress_text.empty()
                    st.warning(f"No questions extracted from Chapter {ch_num}")
                st.rerun()

    with btn_col2:
        if st.button("Extract ALL Chapters"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            chapter_status = st.empty()

            model_id = get_selected_model_id()
            total_chapters = len(st.session_state.chapters)
            max_workers = min(extract_workers, total_chapters)

            # Prepare text with line numbers and image markers
            status_text.text("Preparing text with line numbers and image markers...")
            pdf_path = st.session_state.get("pdf_path")

            if not pdf_path or not os.path.exists(pdf_path):
                st.error("PDF path not found. Please reload the PDF in Step 1.")
                return

            pages_with_lines, all_lines = extract_text_with_lines(pdf_path)
            lines_with_images = insert_image_markers(
                all_lines, st.session_state.images, pages_with_lines
            )
            logger.info(f"Prepared {len(lines_with_images)} lines with {len(st.session_state.images)} image markers")

            # Precompute chapter data for all chapters
            status_text.text("Preparing chapter data...")
            chapter_data = []
            for i, ch in enumerate(st.session_state.chapters):
                ch_num = ch["chapter_number"]
                start_page = ch["start_page"]
                end_page = st.session_state.chapters[i + 1]["start_page"] if i + 1 < len(st.session_state.chapters) else None
                ch_text, line_mapping = build_chapter_text_with_lines(
                    lines_with_images, pages_with_lines, start_page, end_page
                )
                chapter_data.append({
                    "ch_num": ch_num,
                    "ch_key": f"ch{ch_num}",
                    "ch_text": ch_text,
                    "line_mapping": line_mapping
                })

            # Worker function for parallel extraction
            def extract_chapter_worker(ch_data: dict) -> tuple[str, list]:
                """Extract a single chapter. Returns (ch_key, raw_questions)."""
                ch_num = ch_data["ch_num"]
                ch_key = ch_data["ch_key"]
                ch_text = ch_data["ch_text"]
                line_mapping = ch_data["line_mapping"]

                line_ranges = extract_line_ranges_llm(
                    get_anthropic_client(), ch_num, ch_text, model_id
                )

                if not line_ranges:
                    logger.warning(f"Chapter {ch_num}: No line ranges extracted")
                    return ch_key, []

                raw_questions = []
                for lr in line_ranges:
                    q_id = lr.get("question_id", "?")
                    q_start = lr.get("question_start", 0)
                    q_end = lr.get("question_end", 0)
                    a_start = lr.get("answer_start", 0)
                    a_end = lr.get("answer_end", 0)

                    q_text = extract_lines_by_range_mapped(lines_with_images, q_start, q_end, line_mapping) if q_start > 0 else ""
                    a_text = extract_lines_by_range_mapped(lines_with_images, a_start, a_end, line_mapping) if a_start > 0 else ""

                    raw_questions.append({
                        "full_id": f"ch{ch_num}_{q_id}",
                        "local_id": q_id,
                        "chapter": ch_num,
                        "question_start": q_start,
                        "question_end": q_end,
                        "answer_start": a_start,
                        "answer_end": a_end,
                        "correct_letter": lr.get("correct_letter", ""),
                        "image_files": lr.get("image_files", []),
                        "question_text": q_text,
                        "answer_text": a_text
                    })

                logger.info(f"Chapter {ch_num}: Extracted {len(raw_questions)} raw Q&A pairs")
                return ch_key, raw_questions

            # Extract chapters in parallel
            status_text.text(f"Extracting {total_chapters} chapters with {max_workers} parallel workers...")
            completed = 0
            in_progress = set()
            results = {}

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all chapter extraction tasks
                future_to_ch = {
                    executor.submit(extract_chapter_worker, ch_data): ch_data["ch_num"]
                    for ch_data in chapter_data
                }

                # Track in-progress chapters
                for ch_data in chapter_data[:max_workers]:
                    in_progress.add(ch_data["ch_num"])
                chapter_status.text(f"In progress: Ch {', Ch '.join(map(str, sorted(in_progress)))}")

                # Process completed futures as they finish
                for future in as_completed(future_to_ch):
                    ch_num = future_to_ch[future]
                    try:
                        ch_key, raw_questions = future.result()
                        results[ch_key] = raw_questions
                    except Exception as e:
                        logger.error(f"Chapter {ch_num}: Extraction failed - {e}")
                        results[f"ch{ch_num}"] = []

                    completed += 1
                    in_progress.discard(ch_num)

                    # Add next chapter to in-progress set
                    if completed + len(in_progress) <= total_chapters:
                        for ch_data in chapter_data:
                            if ch_data["ch_num"] not in in_progress and f"ch{ch_data['ch_num']}" not in results:
                                in_progress.add(ch_data["ch_num"])
                                break

                    progress_bar.progress(completed / total_chapters)
                    status_text.text(f"Completed {completed}/{total_chapters} chapters...")
                    if in_progress:
                        chapter_status.text(f"In progress: Ch {', Ch '.join(map(str, sorted(in_progress)))}")
                    else:
                        chapter_status.empty()

            # Store results
            st.session_state.raw_questions = results
            save_raw_questions()
            status_text.text("Done!")
            chapter_status.empty()

            total_raw = sum(len(qs) for qs in results.values())
            st.success(f"Extracted {total_raw} raw Q&A pairs from {total_chapters} chapters")
            st.info("**Next:** Go to **Step 4: Format Questions** to format the raw Q&A pairs.")
            st.rerun()

    # Display raw questions if available
    raw_questions = st.session_state.get("raw_questions", {})
    if raw_questions:
        st.markdown("---")
        st.subheader("Raw Extracted Q&A Pairs")

        total_raw = sum(len(qs) for qs in raw_questions.values())
        st.success(f"Total: {total_raw} raw Q&A pairs across {len(raw_questions)} chapters")
        st.info("**Next:** Go to **Step 4: Format Questions** to format these into structured data.")

        # Chapter selector for preview
        ch_options = list(raw_questions.keys())
        if ch_options:
            selected_ch = st.selectbox("Preview chapter:", ch_options, key="raw_preview_ch")

            if selected_ch and selected_ch in raw_questions:
                ch_raw = raw_questions[selected_ch]
                st.caption(f"{len(ch_raw)} Q&A pairs in {selected_ch}")

                for rq in ch_raw:
                    q_preview = rq["question_text"][:100] + "..." if len(rq["question_text"]) > 100 else rq["question_text"]
                    img_indicator = f" [{len(rq['image_files'])} img]" if rq["image_files"] else ""

                    with st.expander(f"Q{rq['local_id']}{img_indicator}: {q_preview}"):
                        st.markdown(f"**Lines:** Q={rq['question_start']}-{rq['question_end']}, A={rq['answer_start']}-{rq['answer_end']}")
                        if rq["correct_letter"]:
                            st.markdown(f"**Correct:** {rq['correct_letter']}")
                        if rq["image_files"]:
                            st.markdown(f"**Images:** {', '.join(rq['image_files'])}")

                        st.markdown("**Question Text:**")
                        st.text_area("", rq["question_text"], height=150, key=f"raw_q_{rq['full_id']}", disabled=True)

                        if rq["answer_text"]:
                            st.markdown("**Answer Text:**")
                            st.text_area("", rq["answer_text"], height=150, key=f"raw_a_{rq['full_id']}", disabled=True)


# =============================================================================
# Step 4: Format Questions
# =============================================================================

def render_format_step():
    """Render question formatting step - formats raw Q&A pairs using parallel LLM calls."""
    st.header("Step 4: Format Questions")

    st.markdown("""
    This step takes the raw Q&A pairs and formats them into structured data using parallel LLM calls.
    Each Q&A pair is processed individually, extracting choices, correct answer, and explanation.
    """)

    # Ensure raw_questions is loaded from file if session state is empty
    raw_questions_file = get_raw_questions_file()
    if not st.session_state.raw_questions and os.path.exists(raw_questions_file):
        with open(raw_questions_file, "r") as f:
            st.session_state.raw_questions = json.load(f)

    raw_questions = st.session_state.get("raw_questions", {})
    if not raw_questions:
        st.warning("Please extract raw questions first (Step 3)")
        return

    client = get_anthropic_client()
    if not client:
        st.error("ANTHROPIC_API_KEY not set. Please configure your .env file.")
        return

    # Initialize logger
    output_dir = get_output_dir()
    logger = get_extraction_logger(output_dir)

    total_raw = sum(len(qs) for qs in raw_questions.values())
    total_formatted = sum(len(qs) for qs in st.session_state.questions.values())

    st.info(f"Raw Q&A pairs: {total_raw} | Formatted: {total_formatted}")

    # Model selection and parallel workers
    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        model_options = get_model_options()
        current_idx = model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
        selected_model = st.selectbox("Model:", model_options, index=current_idx, key="format_model")
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            save_settings()

    with col2:
        max_workers = st.number_input(
            "Parallel workers:",
            min_value=1,
            max_value=100,
            value=20,
            help="Tier 1: use 5-10 | Tier 2+: use 20-50 | Tier 4: use 50-100",
            key="format_workers"
        )

    with col3:
        if st.button("Format ALL Questions", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            model_id = get_selected_model_id()

            # Collect all raw questions
            all_raw = []
            for ch_key, ch_raw in raw_questions.items():
                for rq in ch_raw:
                    all_raw.append((ch_key, rq))

            total = len(all_raw)
            status_text.text(f"Formatting {total} Q&A pairs with {max_workers} parallel workers...")

            # Format in parallel
            formatted_by_chapter = {}
            completed = 0

            def format_single(item):
                ch_key, rq = item
                ch_num = rq["chapter"]
                formatted = format_qa_pair_llm(
                    client,
                    rq["local_id"],
                    rq["question_text"],
                    rq["answer_text"],
                    model_id,
                    ch_num
                )
                # Add metadata
                formatted["full_id"] = rq["full_id"]
                formatted["local_id"] = rq["local_id"]
                formatted["image_files"] = rq["image_files"]
                # Use correct_letter from extraction if LLM didn't find it
                if not formatted.get("correct_answer") and rq.get("correct_letter"):
                    formatted["correct_answer"] = rq["correct_letter"]
                return ch_key, formatted

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(format_single, item): item for item in all_raw}

                for future in as_completed(futures):
                    try:
                        ch_key, formatted = future.result()
                        if ch_key not in formatted_by_chapter:
                            formatted_by_chapter[ch_key] = []
                        formatted_by_chapter[ch_key].append(formatted)
                    except Exception as e:
                        ch_key, rq = futures[future]
                        logger.error(f"Error formatting {rq['full_id']}: {e}")
                        # Add placeholder
                        if ch_key not in formatted_by_chapter:
                            formatted_by_chapter[ch_key] = []
                        formatted_by_chapter[ch_key].append({
                            "full_id": rq["full_id"],
                            "local_id": rq["local_id"],
                            "text": rq["question_text"],
                            "choices": {},
                            "correct_answer": rq.get("correct_letter", ""),
                            "explanation": rq["answer_text"],
                            "image_files": rq["image_files"],
                            "error": str(e)
                        })

                    completed += 1
                    progress_bar.progress(completed / total)
                    status_text.text(f"Formatted {completed}/{total} Q&A pairs...")

            # Sort questions within each chapter
            for ch_key in formatted_by_chapter:
                formatted_by_chapter[ch_key].sort(key=lambda q: question_sort_key(q["full_id"]))

            st.session_state.questions = formatted_by_chapter
            save_questions()

            # Build image assignments from image_files
            for ch_key, questions in formatted_by_chapter.items():
                for q in questions:
                    for img_file in q.get("image_files", []):
                        st.session_state.image_assignments[img_file] = q["full_id"]
            save_image_assignments()

            status_text.text("Done!")
            st.success(f"Formatted {total} Q&A pairs")
            st.info("**Next:** Go to **Step 5: Associate Context** to link context questions to sub-questions.")
            st.rerun()

    # Display formatted questions if available
    if st.session_state.questions:
        st.markdown("---")
        st.subheader("Formatted Questions")

        ch_options = list(st.session_state.questions.keys())
        if ch_options:
            selected_ch = st.selectbox("Preview chapter:", ch_options, key="format_preview_ch")

            if selected_ch and selected_ch in st.session_state.questions:
                questions = st.session_state.questions[selected_ch]
                st.caption(f"{len(questions)} formatted questions in {selected_ch}")

                for q in questions:
                    q_images = [img for img in st.session_state.images
                               if st.session_state.image_assignments.get(img["filename"]) == q["full_id"]]

                    img_indicator = f" [{len(q_images)} img]" if q_images else ""
                    error_indicator = " [ERROR]" if q.get("error") else ""
                    display_text = q['text'][:70] + "..." if len(q['text']) > 70 else q['text']

                    with st.expander(f"Q{q['local_id']}{img_indicator}{error_indicator}: {display_text}"):
                        if q.get("error"):
                            st.error(f"Formatting error: {q['error']}")

                        col1, col2 = st.columns([2, 1])

                        with col1:
                            st.markdown(f"**Question:** {q['text']}")

                            if q.get("choices"):
                                st.markdown("**Choices:**")
                                for letter, choice in q.get("choices", {}).items():
                                    st.markdown(f"- {letter}: {choice}")
                                st.markdown(f"**Correct Answer:** {q.get('correct_answer', 'N/A')}")

                            if q.get("explanation"):
                                st.markdown(f"**Explanation:** {q.get('explanation', '')}")

                        with col2:
                            if q_images:
                                for img in q_images:
                                    if os.path.exists(img["filepath"]):
                                        st.image(img["filepath"], caption=f"Page {img['page']}", width=200)


# =============================================================================
# Step 5: Associate Context
# =============================================================================

def render_context_step():
    """Render context association step."""
    st.header("Step 5: Associate Context")

    if not st.session_state.questions:
        st.warning("Please format questions first (Step 4)")
        return

    st.markdown("""
    This step identifies **context-only questions** (clinical scenarios without answer choices)
    and associates their text and images with the related sub-questions.

    **Example:**
    - Q1 contains a clinical scenario and image (no answer choices)
    - Q1a, Q1b, Q1c are the actual questions with choices
    - After association, Q1's text is prepended to Q1a/Q1b/Q1c and images are linked
    """)

    context_only_count = 0
    total_questions = 0
    for ch_key, ch_questions in st.session_state.questions.items():
        for q in ch_questions:
            total_questions += 1
            if not q.get("choices"):
                context_only_count += 1

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Questions", total_questions)
    with col2:
        st.metric("Potential Context Questions", context_only_count)

    st.markdown("---")

    merged_count = sum(len(qs) for qs in st.session_state.questions_merged.values())
    if merged_count > 0:
        context_only_merged = sum(
            1 for qs in st.session_state.questions_merged.values()
            for q in qs if q.get("is_context_only")
        )
        actual_cards = merged_count - context_only_merged

        st.success(f"Context already associated. {actual_cards} card-ready questions" +
                  (f" + {context_only_merged} context-only" if context_only_merged > 0 else "") +
                  " saved.")

        filter_col1, filter_col2 = st.columns([1, 2])
        with filter_col1:
            hide_context_only = st.checkbox("Hide context-only entries", value=True, key="context_hide_ctx")
        with filter_col2:
            chapter_filter = st.selectbox(
                "Filter by chapter:",
                ["All chapters"] + list(st.session_state.questions_merged.keys()),
                key="context_chapter_filter"
            )

        st.subheader("Merged Questions Preview")

        assignments_to_use = st.session_state.image_assignments_merged if st.session_state.image_assignments_merged else st.session_state.image_assignments

        all_merged_questions = []
        for ch_key in sorted(st.session_state.questions_merged.keys()):
            if chapter_filter != "All chapters" and ch_key != chapter_filter:
                continue
            for q in st.session_state.questions_merged[ch_key]:
                if hide_context_only and q.get("is_context_only"):
                    continue
                all_merged_questions.append((ch_key, q))

        st.caption(f"Showing {len(all_merged_questions)} questions")

        current_chapter = None
        for ch_key, q in all_merged_questions:
            if ch_key != current_chapter:
                current_chapter = ch_key
                questions_in_ch = st.session_state.questions_merged[ch_key]
                ch_context_only = sum(1 for qx in questions_in_ch if qx.get("is_context_only"))
                ch_actual = len(questions_in_ch) - ch_context_only
                st.markdown(f"### Chapter {ch_key} ({ch_actual} questions" +
                           (f" + {ch_context_only} context-only" if ch_context_only > 0 else "") + ")")

            # Get images - check direct assignment first, then inherited from context
            q_images = [img for img in st.session_state.images
                       if assignments_to_use.get(img["filename"]) == q["full_id"]]

            # If no direct images and has context_from, inherit images from context question
            if not q_images and q.get("context_from"):
                context_id = q.get("context_from")
                q_images = [img for img in st.session_state.images
                           if assignments_to_use.get(img["filename"]) == context_id]

            indicators = []

            if q.get("is_context_only"):
                indicators.append("[CTX-ONLY]")
            elif q.get("context_merged"):
                indicators.append("[+CTX]")

            if q_images:
                indicators.append(f"[{len(q_images)} img]")
            elif q.get("has_image"):
                indicators.append("[needs img]")

            indicator_str = " ".join(indicators)
            if indicator_str:
                indicator_str = " " + indicator_str

            display_text = q['text'][:70] + "..." if len(q['text']) > 70 else q['text']

            with st.expander(f"Q{q['local_id']}{indicator_str}: {display_text}"):
                col1, col2 = st.columns([2, 1])

                with col1:
                    if q.get("is_context_only"):
                        st.warning("**CONTEXT ONLY** - This provides context for sub-questions and will NOT become an Anki card")

                    if q.get("context_merged"):
                        st.success(f"Context merged from Q{q.get('context_from', '?').split('_')[-1]}")

                    st.markdown(f"**Question:** {q['text']}")

                    if q.get("choices"):
                        st.markdown("**Choices:**")
                        for letter, choice in q.get("choices", {}).items():
                            st.markdown(f"- {letter}: {choice}")
                        st.markdown(f"**Correct Answer:** {q.get('correct_answer', 'N/A')}")
                        st.markdown(f"**Explanation:** {q.get('explanation', 'N/A')}")

                with col2:
                    if q_images:
                        for img in q_images:
                            if os.path.exists(img["filepath"]):
                                st.image(img["filepath"], caption=f"Page {img['page']}", width=200)
                    elif q.get("has_image"):
                        st.warning("Needs image assignment")

    st.markdown("---")

    col1, col2, col3 = st.columns([2, 1.5, 2])

    with col1:
        model_options = get_model_options()
        current_idx = model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
        selected_model = st.selectbox("Model:", model_options, index=current_idx, key="context_model")
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            save_settings()

    with col2:
        context_workers = st.number_input(
            "Parallel workers:",
            min_value=1,
            max_value=50,
            value=10,
            help="Number of chapters to process in parallel",
            key="context_workers"
        )

    with col3:
        if st.button("Associate Context", type="primary"):
            client = get_anthropic_client()
            if not client:
                st.error("ANTHROPIC_API_KEY not set. Please set the environment variable.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                chapter_status = st.empty()

                questions_copy = copy.deepcopy(st.session_state.questions)
                assignments_copy = copy.deepcopy(st.session_state.image_assignments)
                model_id = get_model_id(st.session_state.selected_model)

                total_chapters = len(questions_copy)
                max_workers = min(context_workers, total_chapters)
                status_text.text(f"Processing {total_chapters} chapters with {max_workers} parallel workers...")

                # Worker function for parallel context association
                def process_chapter_context(ch_key: str, ch_questions: list) -> tuple:
                    """Process context for one chapter. Returns (ch_key, updated_questions, ch_stats)."""
                    from llm_extraction import get_prompt, stream_message, get_extraction_logger
                    import json
                    import re

                    logger = get_extraction_logger()
                    ch_stats = {"context_questions_found": 0, "sub_questions_updated": 0, "images_copied": 0}

                    if not ch_questions:
                        return ch_key, ch_questions, ch_stats

                    # Build summary for LLM
                    questions_summary = []
                    for q in ch_questions:
                        has_choices = bool(q.get("choices"))
                        summary = {
                            "full_id": q["full_id"],
                            "local_id": q["local_id"],
                            "text_preview": q["text"][:200] + "..." if len(q["text"]) > 200 else q["text"],
                            "has_choices": has_choices
                        }
                        questions_summary.append(summary)

                    prompt = get_prompt("associate_context",
                                       questions_summary=json.dumps(questions_summary, indent=2))

                    try:
                        response_text, usage = stream_message(
                            get_anthropic_client(),
                            model_id,
                            messages=[{"role": "user", "content": prompt}]
                        )

                        logger.info(f"Context association {ch_key}: input={usage['input_tokens']:,}, output={usage['output_tokens']:,}")

                        if "```" in response_text:
                            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
                            if match:
                                response_text = match.group(1)

                        result = json.loads(response_text)
                        mappings = result.get("context_mappings", [])

                        question_by_id = {q["full_id"]: q for q in ch_questions}

                        for mapping in mappings:
                            context_id = mapping.get("context_id")
                            sub_ids = mapping.get("sub_question_ids", [])

                            if not context_id or context_id not in question_by_id:
                                continue

                            context_q = question_by_id[context_id]
                            context_text = context_q.get("text", "").strip()
                            context_q["is_context_only"] = True
                            ch_stats["context_questions_found"] += 1

                            # Count context images
                            context_images = [img for img, assigned_to in assignments_copy.items() if assigned_to == context_id]
                            ch_stats["images_copied"] += len(context_images)

                            for sub_id in sub_ids:
                                if sub_id not in question_by_id:
                                    continue
                                sub_q = question_by_id[sub_id]
                                if sub_q.get("context_merged"):
                                    continue
                                original_text = sub_q.get("text", "").strip()
                                sub_q["text"] = f"{context_text} {original_text}"
                                sub_q["context_merged"] = True
                                sub_q["context_from"] = context_id
                                sub_q["is_context_only"] = False
                                ch_stats["sub_questions_updated"] += 1

                    except json.JSONDecodeError as e:
                        logger.error(f"Context association {ch_key}: JSON parse error - {e}")
                    except Exception as e:
                        logger.error(f"Context association {ch_key}: API error - {type(e).__name__}: {e}")

                    # Ensure all questions have is_context_only set
                    for q in ch_questions:
                        if "is_context_only" not in q:
                            q["is_context_only"] = False

                    return ch_key, ch_questions, ch_stats

                # Process chapters in parallel
                completed = 0
                updated_questions = {}
                total_stats = {"context_questions_found": 0, "sub_questions_updated": 0, "images_copied": 0}

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_ch = {
                        executor.submit(process_chapter_context, ch_key, ch_questions): ch_key
                        for ch_key, ch_questions in questions_copy.items()
                    }

                    in_progress = set(list(questions_copy.keys())[:max_workers])
                    chapter_status.text(f"In progress: {', '.join(sorted(in_progress))}")

                    for future in as_completed(future_to_ch):
                        ch_key = future_to_ch[future]
                        try:
                            result_ch_key, result_questions, ch_stats = future.result()
                            updated_questions[result_ch_key] = result_questions
                            for key in total_stats:
                                total_stats[key] += ch_stats[key]
                        except Exception as e:
                            logger.error(f"Context association {ch_key}: Failed - {e}")
                            updated_questions[ch_key] = questions_copy[ch_key]

                        completed += 1
                        in_progress.discard(ch_key)

                        # Update in-progress set
                        for next_ch in questions_copy.keys():
                            if next_ch not in in_progress and next_ch not in updated_questions:
                                in_progress.add(next_ch)
                                break

                        progress_bar.progress(completed / total_chapters)
                        status_text.text(f"Completed {completed}/{total_chapters} chapters...")
                        if in_progress:
                            chapter_status.text(f"In progress: {', '.join(sorted(in_progress))}")
                        else:
                            chapter_status.empty()

                status_text.text("Saving merged data...")

                st.session_state.questions_merged = updated_questions
                st.session_state.image_assignments_merged = assignments_copy
                save_questions_merged()
                save_image_assignments_merged()

                progress_bar.progress(1.0)
                status_text.text("Done!")
                chapter_status.empty()

                st.success(
                    f"Context association complete!\n\n"
                    f"- Context questions found: {total_stats['context_questions_found']}\n"
                    f"- Sub-questions updated: {total_stats['sub_questions_updated']}\n"
                    f"- Images copied: {total_stats['images_copied']}"
                )

                st.rerun()

    if merged_count > 0:
        st.markdown("---")
        st.subheader("Manage Merged Data")

        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            context_only_ids = []
            for ch_key, ch_questions in st.session_state.questions_merged.items():
                for q in ch_questions:
                    if q.get("is_context_only"):
                        context_only_ids.append(q["full_id"])

            if context_only_ids:
                if st.button(f"Remove {len(context_only_ids)} Context-Only Questions", type="primary"):
                    for ch_key in st.session_state.questions_merged:
                        st.session_state.questions_merged[ch_key] = [
                            q for q in st.session_state.questions_merged[ch_key]
                            if not q.get("is_context_only")
                        ]
                    if st.session_state.image_assignments_merged:
                        st.session_state.image_assignments_merged = {
                            img: q_id for img, q_id in st.session_state.image_assignments_merged.items()
                            if q_id not in context_only_ids
                        }
                    save_questions_merged()
                    save_image_assignments_merged()
                    st.success(f"Removed {len(context_only_ids)} context-only entries. Only card-ready questions remain.")
                    st.rerun()
            else:
                st.info("No context-only questions to remove.")

        with btn_col2:
            if st.button("Clear All Merged Data", type="secondary"):
                st.session_state.questions_merged = {}
                st.session_state.image_assignments_merged = {}
                from state_management import get_questions_merged_file, get_image_assignments_merged_file
                merged_file = get_questions_merged_file()
                assignments_merged_file = get_image_assignments_merged_file()
                if os.path.exists(merged_file):
                    os.remove(merged_file)
                if os.path.exists(assignments_merged_file):
                    os.remove(assignments_merged_file)
                st.success("Merged data cleared. Original questions remain intact.")
                st.rerun()


# =============================================================================
# Step 6: QC Questions
# =============================================================================

def render_qc_step():
    """Render QC review step."""
    st.header("Step 6: QC Questions")

    if not st.session_state.questions:
        st.warning("Please format questions first (Step 4)")
        return

    all_questions = []
    for ch_key, questions in st.session_state.questions.items():
        for q in questions:
            all_questions.append((ch_key, q))

    all_questions.sort(key=lambda x: question_sort_key(x[1]["full_id"]))

    if not all_questions:
        st.warning("No questions to review")
        return

    reviewed = st.session_state.qc_progress.get("reviewed", {})
    total = len(all_questions)
    reviewed_count = len(reviewed)

    st.progress(reviewed_count / total if total > 0 else 0)
    st.caption(f"Progress: {reviewed_count}/{total} questions reviewed")

    col1, col2, col3 = st.columns(3)
    with col1:
        filter_option = st.radio("Show:", ["All", "Unreviewed only", "Reviewed only"], horizontal=True)
    with col2:
        chapter_filter = st.selectbox("Filter by chapter:",
                                       ["All chapters"] + list(st.session_state.questions.keys()))
    with col3:
        hide_context = st.checkbox("Hide context-only entries", value=True)

    filtered_questions = []
    for ch_key, q in all_questions:
        q_id = q["full_id"]
        is_reviewed = q_id in reviewed

        if filter_option == "Unreviewed only" and is_reviewed:
            continue
        if filter_option == "Reviewed only" and not is_reviewed:
            continue
        if chapter_filter != "All chapters" and ch_key != chapter_filter:
            continue
        if hide_context and q.get("is_context_only"):
            continue

        filtered_questions.append((ch_key, q))

    st.caption(f"Showing {len(filtered_questions)} questions")

    if filtered_questions:
        def format_question_option(q):
            prefix = ""
            if q.get("is_context_only"):
                prefix = "[CTX] "
            elif q.get("context"):
                prefix = "[+ctx] "
            return f"{prefix}{q['full_id']}: {q['text'][:50]}..."

        question_options = [format_question_option(q) for _, q in filtered_questions]

        if st.session_state.qc_selected_idx >= len(filtered_questions):
            st.session_state.qc_selected_idx = 0

        # Define callbacks for navigation (run before widgets render)
        def go_previous():
            if st.session_state.qc_selected_idx > 0:
                st.session_state.qc_selected_idx -= 1
                # Delete the selectbox's stored state so it uses the new index
                if "qc_question_selector" in st.session_state:
                    del st.session_state.qc_question_selector
                save_settings()

        def go_next():
            if st.session_state.qc_selected_idx < len(filtered_questions) - 1:
                st.session_state.qc_selected_idx += 1
                # Delete the selectbox's stored state so it uses the new index
                if "qc_question_selector" in st.session_state:
                    del st.session_state.qc_question_selector
                save_settings()

        selected_idx = st.selectbox(
            "Select question:",
            range(len(question_options)),
            index=st.session_state.qc_selected_idx,
            format_func=lambda x: question_options[x],
            key="qc_question_selector"
        )
        if selected_idx != st.session_state.qc_selected_idx:
            st.session_state.qc_selected_idx = selected_idx
            save_settings()

        selected_idx = st.session_state.qc_selected_idx
        if selected_idx is not None and selected_idx < len(filtered_questions):
            ch_key, q = filtered_questions[selected_idx]
            q_id = q["full_id"]

            # Get review status for this question
            is_approved = q_id in reviewed and reviewed[q_id].get("status") == "approved"
            is_flagged = q_id in reviewed and reviewed[q_id].get("status") == "flagged"

            # Define callbacks for QC actions
            def approve_and_next():
                reviewed[q_id] = {"status": "approved", "timestamp": datetime.now().isoformat()}
                st.session_state.qc_progress["reviewed"] = reviewed
                save_qc_progress()
                if st.session_state.qc_selected_idx < len(filtered_questions) - 1:
                    st.session_state.qc_selected_idx += 1
                    if "qc_question_selector" in st.session_state:
                        del st.session_state.qc_question_selector
                    save_settings()

            def flag_issue():
                reviewed[q_id] = {"status": "flagged", "timestamp": datetime.now().isoformat()}
                st.session_state.qc_progress["reviewed"] = reviewed
                save_qc_progress()

            def unapprove():
                reviewed.pop(q_id, None)
                st.session_state.qc_progress["reviewed"] = reviewed
                save_qc_progress()

            # Navigation and action buttons in a single row (fixed position)
            # Column widths tuned to prevent text wrapping
            col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1.5, 1.2, 1.3, 1.2])

            with col1:
                st.button("Previous", disabled=(st.session_state.qc_selected_idx <= 0), on_click=go_previous)

            with col2:
                st.button("Next", disabled=(st.session_state.qc_selected_idx >= len(filtered_questions) - 1), on_click=go_next)

            with col3:
                st.button("Approve & Next", type="primary", disabled=is_approved, on_click=approve_and_next)

            with col4:
                st.button("Flag Issue", disabled=is_flagged, on_click=flag_issue)

            with col5:
                if st.button("Detect Pages", help="Detect PDF pages for each question"):
                    with st.spinner("Detecting page numbers..."):
                        add_page_numbers_to_questions(
                            st.session_state.questions,
                            st.session_state.pages,
                            st.session_state.chapters
                        )
                        save_questions()
                        st.rerun()

            with col6:
                if is_approved or is_flagged:
                    st.button("Unapprove", on_click=unapprove)

            st.markdown("---")

            left_col, right_col = st.columns([1, 1])

            with left_col:
                # Status indicator with color
                if is_approved:
                    status_html = '<span style="color: green;">(approved)</span>'
                elif is_flagged:
                    status_html = '<span style="color: orange;">(flagged)</span>'
                else:
                    status_html = '<span style="color: red;">(pending)</span>'
                st.markdown(f"### Question {q['local_id']} {status_html}", unsafe_allow_html=True)

                if q.get("is_context_only"):
                    st.warning("**CONTEXT ONLY** - This entry provides context for sub-questions and will not be exported to Anki.")
                    st.markdown(f"**Context Text:**\n\n{q['text']}")
                else:
                    if q.get("context"):
                        st.info(f"**Context (from Q{q.get('context_question_id', '?')}):**\n\n{q['context']}")

                    st.markdown(f"**Question:** {q['text']}")

                    if q.get("choices"):
                        st.markdown("**Choices:**")
                        for letter, choice in q.get("choices", {}).items():
                            if letter == q.get("correct_answer"):
                                st.markdown(f"- **{letter}: {choice}** (correct)")
                            else:
                                st.markdown(f"- {letter}: {choice}")
                    else:
                        st.caption("No answer choices")

                    st.markdown(f"**Explanation:** {q.get('explanation', 'N/A')}")

            with right_col:
                assigned_images = get_images_for_question(q_id)

                ch_num = int(ch_key[2:])
                ch_start = next((c["start_page"] for c in st.session_state.chapters if c["chapter_number"] == ch_num), 1)
                ch_end = 9999
                for i, c in enumerate(st.session_state.chapters):
                    if c["chapter_number"] == ch_num and i + 1 < len(st.session_state.chapters):
                        ch_end = st.session_state.chapters[i + 1]["start_page"]

                # Define callback functions for image operations
                def remove_image(img_filename):
                    st.session_state.image_assignments.pop(img_filename, None)
                    save_image_assignments()
                    if st.session_state.image_assignments_merged:
                        st.session_state.image_assignments_merged.pop(img_filename, None)
                        save_image_assignments_merged()
                    if "qc_question_selector" in st.session_state:
                        del st.session_state.qc_question_selector

                def assign_image(img_filename, question_id):
                    st.session_state.image_assignments[img_filename] = question_id
                    save_image_assignments()
                    if st.session_state.image_assignments_merged:
                        st.session_state.image_assignments_merged[img_filename] = question_id
                        save_image_assignments_merged()
                    if "qc_question_selector" in st.session_state:
                        del st.session_state.qc_question_selector

                # Create tabs for Images and PDF Pages
                images_tab, pdf_tab = st.tabs(["Images", "PDF Pages"])

                with images_tab:
                    if assigned_images:
                        st.subheader("Assigned Image(s)")
                        for img in assigned_images:
                            filepath = img["filepath"]
                            if os.path.exists(filepath):
                                st.image(filepath, caption=f"Page {img['page']} - {img['filename']}", use_column_width=True)

                                st.button("Remove Image", key=f"img_remove_{img['filename']}",
                                         on_click=remove_image, args=(img["filename"],))
                            else:
                                st.warning(f"Image not found: {filepath}")

                        with st.expander("Assign different/additional image"):
                            unassigned = [img for img in st.session_state.images
                                          if img["filename"] not in st.session_state.image_assignments
                                          and ch_start <= img["page"] < ch_end]
                            if unassigned:
                                for img in unassigned[:5]:
                                    filepath = img["filepath"]
                                    if os.path.exists(filepath):
                                        st.image(filepath, caption=f"Page {img['page']}", width=150)
                                        st.button(f"Add this image", key=f"add_{img['filename']}",
                                                 on_click=assign_image, args=(img["filename"], q_id))
                            else:
                                st.caption("No more unassigned images in this chapter")

                    else:
                        if q.get("has_image"):
                            st.subheader("Image Required")
                            st.warning("This question needs an image but none assigned yet.")
                        else:
                            st.subheader("No Image Assigned")
                            st.caption("No image currently linked to this question")

                        with st.expander("Manually assign an image" if not q.get("has_image") else "Select from chapter images", expanded=q.get("has_image", False)):
                            unassigned = [img for img in st.session_state.images
                                          if img["filename"] not in st.session_state.image_assignments
                                          and ch_start <= img["page"] < ch_end]

                            if unassigned:
                                for img in unassigned[:6]:
                                    filepath = img["filepath"]
                                    if os.path.exists(filepath):
                                        st.image(filepath, caption=f"Page {img['page']}", width=180)
                                        st.button(f"Assign", key=f"assign_{img['filename']}",
                                                 on_click=assign_image, args=(img["filename"], q_id))
                            else:
                                st.caption("No unassigned images in this chapter")

                with pdf_tab:
                    # Support both single page (legacy) and multi-page formats
                    question_pages = q.get("question_pages", [])
                    answer_pages = q.get("answer_pages", [])
                    # Fall back to single page if lists not available
                    if not question_pages and q.get("question_page"):
                        question_pages = [q.get("question_page")]
                    if not answer_pages and q.get("answer_page"):
                        answer_pages = [q.get("answer_page")]

                    pdf_path = st.session_state.get("pdf_path")

                    if not pdf_path or not os.path.exists(pdf_path):
                        st.warning("PDF file not available. Re-load the textbook in Step 1 to enable PDF preview.")
                    elif not question_pages and not answer_pages:
                        st.info("Page numbers not detected for this question.")
                        if st.button("Detect Page Numbers", key=f"detect_pages_{q_id}"):
                            with st.spinner("Detecting page numbers..."):
                                add_page_numbers_to_questions(
                                    st.session_state.questions,
                                    st.session_state.pages,
                                    st.session_state.chapters
                                )
                                save_questions()
                                st.rerun()
                    else:
                        # Show question pages and answer pages side by side
                        q_col, a_col = st.columns(2)

                        with q_col:
                            if len(question_pages) > 1:
                                st.markdown(f"**Question Pages ({len(question_pages)})**")
                            else:
                                st.markdown("**Question Page**")

                            if question_pages:
                                for page_num in question_pages:
                                    png_bytes = render_pdf_page(pdf_path, page_num, zoom=1.2)
                                    if png_bytes:
                                        st.image(png_bytes, caption=f"Page {page_num}", use_column_width=True)
                                    else:
                                        st.error(f"Failed to render page {page_num}")
                            else:
                                st.caption("Page not detected")

                        with a_col:
                            if len(answer_pages) > 1:
                                st.markdown(f"**Answer Pages ({len(answer_pages)})**")
                            else:
                                st.markdown("**Answer Page**")

                            if answer_pages:
                                for page_num in answer_pages:
                                    png_bytes = render_pdf_page(pdf_path, page_num, zoom=1.2)
                                    if png_bytes:
                                        st.image(png_bytes, caption=f"Page {page_num}", use_column_width=True)
                                    else:
                                        st.error(f"Failed to render page {page_num}")
                            else:
                                st.caption("Page not detected")


# =============================================================================
# Step 7: Export
# =============================================================================

def generate_anki_deck(book_name: str, questions: dict, chapters: list, image_assignments: dict,
                       images: list, include_images: bool, only_approved: bool, qc_progress: dict) -> str:
    """Generate Anki deck and return path to .apkg file."""
    import genanki
    import hashlib

    # Generate stable IDs based on name
    def stable_id(name: str) -> int:
        return int(hashlib.md5(name.encode()).hexdigest()[:8], 16)

    model_id = stable_id(f"{book_name}_model")

    # Create card model (template)
    model = genanki.Model(
        model_id,
        f'{book_name} Model',
        fields=[
            {'name': 'Question'},
            {'name': 'Choices'},
            {'name': 'Answer'},
            {'name': 'Explanation'},
            {'name': 'Image'},
            {'name': 'Chapter'},
        ],
        templates=[
            {
                'name': 'Card 1',
                'qfmt': '''
                    <div class="question">{{Question}}</div>
                    {{#Image}}<div class="image">{{Image}}</div>{{/Image}}
                    <div class="choices">{{Choices}}</div>
                ''',
                'afmt': '''
                    {{FrontSide}}
                    <hr id="answer">
                    <div class="answer"><b>Answer:</b> {{Answer}}</div>
                    <div class="explanation">{{Explanation}}</div>
                ''',
            },
        ],
        css='''
            .card { font-family: arial; font-size: 16px; text-align: left; }
            .question { font-weight: bold; margin-bottom: 10px; }
            .choices { margin: 10px 0; }
            .answer { color: green; font-weight: bold; margin: 10px 0; }
            .explanation { margin-top: 10px; font-style: italic; }
            .image { margin: 10px 0; }
            .image img { max-width: 100%; height: auto; }
        '''
    )

    # Build image lookup
    image_lookup = {img['filename']: img for img in images}

    # Track media files to include
    media_files = []

    # Collect all chapter decks
    all_decks = []

    # Create sub-decks for each chapter
    reviewed = qc_progress.get("reviewed", {})

    for ch in chapters:
        ch_num = ch['chapter_number']
        ch_key = f"ch{ch_num}"
        ch_title = ch.get('title', f'Chapter {ch_num}')
        ch_questions = questions.get(ch_key, [])

        if not ch_questions:
            continue

        # Create chapter sub-deck
        ch_deck_name = f"{book_name}::{ch_num}. {ch_title}"
        ch_deck_id = stable_id(ch_deck_name)
        ch_deck = genanki.Deck(ch_deck_id, ch_deck_name)

        for q in ch_questions:
            q_id = q['full_id']

            # Skip context-only questions
            if q.get('is_context_only'):
                continue

            # Skip if only approved and not approved
            if only_approved:
                review_status = reviewed.get(q_id, {}).get('status')
                if review_status != 'approved':
                    continue

            # Build question text (include context if merged)
            q_text = q.get('text', '')

            # Build choices HTML
            choices = q.get('choices', {})
            if choices:
                choices_html = '<br>'.join([f"{letter}. {text}" for letter, text in sorted(choices.items())])
            else:
                choices_html = ''

            # Get correct answer
            correct = q.get('correct_answer', '')

            # Get explanation
            explanation = q.get('explanation', '')

            # Handle image
            image_html = ''
            if include_images:
                # Find images assigned to this question
                assigned_imgs = [fname for fname, assigned_q in image_assignments.items() if assigned_q == q_id]

                # Also check for inherited images via context_from
                if not assigned_imgs and q.get('context_from'):
                    context_id = q['context_from']
                    assigned_imgs = [fname for fname, assigned_q in image_assignments.items() if assigned_q == context_id]

                for img_fname in assigned_imgs:
                    if img_fname in image_lookup:
                        img_data = image_lookup[img_fname]
                        filepath = img_data.get('filepath', '')
                        if os.path.exists(filepath):
                            media_files.append(filepath)
                            image_html += f'<img src="{img_fname}">'

            # Create note
            note = genanki.Note(
                model=model,
                fields=[q_text, choices_html, correct, explanation, image_html, ch_title],
                tags=[f"chapter{ch_num}"]
            )
            ch_deck.notes.append(note)

        # Add chapter deck to list if it has notes
        if ch_deck.notes:
            all_decks.append(ch_deck)

    # Create package with all chapter decks
    output_dir = get_output_dir()
    safe_name = re.sub(r'[^\w\-]', '_', book_name)
    output_path = os.path.join(output_dir, f"{safe_name}.apkg")

    if not all_decks:
        raise ValueError("No cards to export")

    package = genanki.Package(all_decks)
    if media_files:
        package.media_files = media_files

    package.write_to_file(output_path)

    return output_path


def render_export_step():
    """Render export step."""
    st.header("Step 7: Export to Anki")

    if not st.session_state.questions:
        st.warning("Please format questions first (Step 4)")
        return

    # Calculate stats
    total = sum(len(qs) for qs in st.session_state.questions.values())
    context_only = sum(1 for qs in st.session_state.questions.values()
                       for q in qs if q.get('is_context_only'))
    exportable = total - context_only

    reviewed = st.session_state.qc_progress.get("reviewed", {})
    approved = sum(1 for r in reviewed.values() if r.get("status") == "approved")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Questions", total)
    with col2:
        st.metric("Exportable", exportable, help="Excludes context-only entries")
    with col3:
        st.metric("Approved (QC'd)", approved)

    st.markdown("---")

    # Book name setting
    default_name = Path(st.session_state.current_pdf).stem if st.session_state.current_pdf else "Textbook"
    default_name = default_name.replace('_', ' ').replace('-', ' ')

    book_name = st.text_input("Book Name (used as deck name)",
                              value=default_name,
                              help="This will be the parent deck name in Anki")

    # Export options
    col1, col2 = st.columns(2)
    with col1:
        only_approved = st.checkbox("Only export approved questions",
                                   value=False,
                                   help="Only include questions marked as approved in QC")
    with col2:
        include_images = st.checkbox("Include images",
                                    value=True,
                                    help="Embed assigned images in cards")

    st.markdown("---")

    # Preview deck structure
    with st.expander("Preview Deck Structure"):
        st.markdown(f"**{book_name}**")
        for ch in (st.session_state.chapters or []):
            ch_num = ch['chapter_number']
            ch_title = ch.get('title', f'Chapter {ch_num}')
            ch_key = f"ch{ch_num}"
            ch_count = len([q for q in st.session_state.questions.get(ch_key, [])
                           if not q.get('is_context_only')])
            if ch_count > 0:
                st.markdown(f"  - {ch_num}. {ch_title} ({ch_count} cards)")

    # Export button
    if st.button("Export to Anki Deck", type="primary"):
        if not book_name.strip():
            st.error("Please enter a book name")
            return

        with st.spinner("Generating Anki deck..."):
            try:
                output_path = generate_anki_deck(
                    book_name=book_name.strip(),
                    questions=st.session_state.questions,
                    chapters=st.session_state.chapters or [],
                    image_assignments=st.session_state.image_assignments,
                    images=st.session_state.images,
                    include_images=include_images,
                    only_approved=only_approved,
                    qc_progress=st.session_state.qc_progress
                )

                st.success(f"Deck exported successfully!")
                st.info(f"Saved to: `{output_path}`")

                # Provide download button
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="Download .apkg file",
                        data=f,
                        file_name=os.path.basename(output_path),
                        mime="application/octet-stream"
                    )

            except Exception as e:
                st.error(f"Export failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


# =============================================================================
# Step 8: Edit Prompts
# =============================================================================

def render_prompts_step():
    """Render prompt editor step."""
    st.header("Step 8: Edit Prompts")

    st.caption("Edit the LLM prompts used for extraction. Changes are saved to `scripts/config/prompts.yaml`.")

    prompts = load_prompts()

    # Create tabs for each prompt
    prompt_names = list(prompts.keys())
    tabs = st.tabs([prompts[name].get("description", name) for name in prompt_names])

    for i, prompt_name in enumerate(prompt_names):
        with tabs[i]:
            prompt_data = prompts[prompt_name]
            description = prompt_data.get("description", "")
            current_prompt = prompt_data.get("prompt", "")

            st.markdown(f"**Prompt name:** `{prompt_name}`")
            st.caption(description)

            # Text area for editing
            new_prompt = st.text_area(
                "Prompt template",
                value=current_prompt,
                height=400,
                key=f"prompt_editor_{prompt_name}",
                help="Use {variable_name} syntax for variables that get filled in at runtime"
            )

            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Save", key=f"save_{prompt_name}", type="primary"):
                    prompts[prompt_name]["prompt"] = new_prompt
                    save_prompts(prompts)
                    st.success(f"Saved {prompt_name}!")
            with col2:
                if st.button("Reset to saved", key=f"reset_{prompt_name}"):
                    reload_prompts()
                    st.rerun()

    st.markdown("---")
    st.subheader("Tips")
    st.markdown("""
    - **Variables**: Use `{variable_name}` syntax. Available variables depend on the prompt:
      - `identify_chapters`: `{page_index}`
      - `extract_qa_pairs`: `{chapter_num}`, `{chapter_text}`
      - `match_images_to_questions`: `{chapter_num}`, `{questions_text}`, `{images_text}`
      - `associate_context`: `{questions_summary}`
    - **JSON output**: Most prompts expect JSON output. Keep the output format instructions.
    - **Testing**: After editing, re-run the relevant extraction step to test your changes.
    """)
