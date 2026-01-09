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
    load_prompts, save_prompts, reload_prompts, get_prompt, stream_message,
    get_extraction_logger, get_log_file_path, reset_logger
)


# =============================================================================
# Helper Functions
# =============================================================================

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


def get_selected_model_id() -> str:
    """Get the currently selected Claude model ID."""
    return get_model_id(st.session_state.selected_model)


def clear_step_data(step_id: str, cascade: bool = True):
    """Clear data for a specific step and all subsequent steps, allowing re-run from that point.

    Args:
        step_id: One of 'source', 'chapters', 'questions', 'format', 'context', 'qc', 'generate', 'export'
        cascade: If True, also clear all subsequent steps (default True)
    """
    # Define step order for cascading
    step_order = ["source", "chapters", "questions", "format", "context", "qc", "generate", "export"]

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

        elif step == "context":
            # Clear merged questions and assignments
            st.session_state.questions_merged = {}
            st.session_state.image_assignments_merged = {}
            for filename in ["questions_merged.json", "image_assignments_merged.json"]:
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

def get_step_completion_status() -> dict[str, bool]:
    """Check which steps are completed based on session state data."""
    # QC is complete only when ALL questions have been reviewed
    total_questions = sum(len(qs) for qs in st.session_state.questions.values())
    reviewed_count = len(st.session_state.qc_progress.get("reviewed", {}))
    qc_complete = total_questions > 0 and reviewed_count >= total_questions

    # Generate step is complete when we have generated cards
    generated_cards = st.session_state.generated_questions.get("generated_cards", {})
    generate_complete = bool(generated_cards) and sum(len(c) for c in generated_cards.values()) > 0

    return {
        "source": st.session_state.pages is not None and len(st.session_state.pages) > 0,
        "chapters": st.session_state.chapters is not None and len(st.session_state.chapters) > 0,
        "questions": bool(st.session_state.get("raw_questions")) and sum(len(qs) for qs in st.session_state.raw_questions.values()) > 0,
        "format": bool(st.session_state.questions) and sum(len(qs) for qs in st.session_state.questions.values()) > 0,
        "context": bool(st.session_state.questions_merged) and sum(len(qs) for qs in st.session_state.questions_merged.values()) > 0,
        "qc": qc_complete,
        "generate": generate_complete,
        "export": False,  # Export is an action, not a state
        "prompts": False,  # Prompts step is always accessible, not completable
    }


def render_sidebar():
    """Render sidebar with navigation."""
    st.sidebar.title("Textbook Q&A Extractor")

    if st.session_state.current_pdf:
        textbook_name = get_pdf_slug(st.session_state.current_pdf).replace('_', ' ')
        st.sidebar.caption(f"Working on: **{textbook_name}**")

    st.sidebar.markdown("---")

    # Get step completion status
    completion_status = get_step_completion_status()

    # CSS for green completed step buttons
    st.sidebar.markdown("""
        <style>
        /* Green styling for completed step buttons (primary type) in sidebar */
        section[data-testid="stSidebar"] button[kind="primary"],
        [data-testid="stSidebar"] button[kind="primary"] {
            background-color: #28a745 !important;
            color: white !important;
            border-color: #28a745 !important;
        }
        section[data-testid="stSidebar"] button[kind="primary"]:hover,
        [data-testid="stSidebar"] button[kind="primary"]:hover {
            background-color: #218838 !important;
            border-color: #1e7e34 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    steps = [
        ("source", "1. Select Source"),
        ("chapters", "2. Extract Chapters"),
        ("questions", "3. Extract Questions"),
        ("format", "4. Format Questions"),
        ("context", "5. Associate Context"),
        ("qc", "6. QC Questions"),
        ("generate", "7. Generate Questions"),
        ("export", "8. Export"),
        ("prompts", "9. Edit Prompts")
    ]

    for step_id, step_name in steps:
        is_complete = completion_status.get(step_id, False)
        btn_type = "primary" if is_complete else "secondary"

        if st.sidebar.button(step_name, key=f"nav_{step_id}", type=btn_type):
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

def run_feeling_lucky(pdf_path: str, models: dict, workers: dict):
    """Run all steps automatically from PDF loading through context association.

    Args:
        pdf_path: Path to the PDF file
        models: Dict with model names for each step: 'chapters', 'questions', 'format', 'context'
        workers: Dict with worker counts for parallel steps: 'questions', 'format', 'context'
    """
    client = get_anthropic_client()
    logger = get_extraction_logger()

    # Step 1: Load PDF
    st.markdown("### Step 1: Loading PDF...")
    progress_step1 = st.progress(0)

    progress_step1.progress(20, "Extracting images...")
    st.session_state.images = extract_images_from_pdf(pdf_path, get_images_dir())
    save_images()

    progress_step1.progress(50, "Extracting text...")
    pages, all_lines = extract_text_with_lines(pdf_path)
    st.session_state.pages = pages
    st.session_state.pdf_path = pdf_path
    save_pages()

    progress_step1.progress(70, "Inserting image markers...")
    lines_with_markers = insert_image_markers(all_lines, st.session_state.images, pages)

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

    output_dir = get_output_dir()
    text_file_path = os.path.join(output_dir, "extracted_text.txt")
    with open(text_file_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    progress_step1.progress(100, "PDF loaded!")
    st.success(f"Step 1 complete: {len(pages)} pages, {len(st.session_state.images)} images")

    # Step 2: Extract Chapters
    st.markdown("### Step 2: Extracting Chapters...")
    progress_step2 = st.progress(0)

    chapters_model_id = get_model_id(models['chapters'])
    progress_step2.progress(30, f"Identifying chapter boundaries with {models['chapters']}...")
    chapters = identify_chapters_llm(client, pages, chapters_model_id)

    if chapters:
        chapters = sorted(chapters, key=lambda ch: ch.get("chapter_number", 0))
        st.session_state.chapters = chapters

        # Extract chapter texts
        for i, ch in enumerate(chapters):
            start_page = ch["start_page"]
            end_page = chapters[i + 1]["start_page"] if i + 1 < len(chapters) else None
            ch_key = f"ch{ch['chapter_number']}"
            st.session_state.chapter_texts[ch_key] = extract_chapter_text(pages, start_page, end_page)

        save_chapters()

        # Assign chapters to images
        if st.session_state.images:
            st.session_state.images = assign_chapters_to_images(st.session_state.images, chapters)
            save_images()

        progress_step2.progress(100, "Chapters extracted!")
        st.success(f"Step 2 complete: {len(chapters)} chapters found")
    else:
        st.error("No chapters found!")
        return False

    # Step 3: Extract Questions (Raw)
    st.markdown("### Step 3: Extracting Questions...")
    progress_step3 = st.progress(0)

    questions_model_id = get_model_id(models['questions'])

    # Use the raw lines with image markers (without line prefixes) from Step 1
    # build_chapter_text_with_lines will add line prefixes
    lines_with_images = lines_with_markers

    chapters = st.session_state.chapters
    total_chapters = len(chapters)
    results = {}

    # Define extract_chapter_questions inline for this context
    def extract_chapter_questions(ch_idx):
        chapter = chapters[ch_idx]
        ch_num = chapter.get("chapter_number", ch_idx + 1)
        ch_key = f"ch{ch_num}"

        start_page = chapter["start_page"]
        end_page = chapters[ch_idx + 1]["start_page"] if ch_idx + 1 < len(chapters) else None

        # Build chapter text with line numbers
        ch_text, line_mapping = build_chapter_text_with_lines(
            lines_with_images, pages, start_page, end_page
        )

        if not ch_text.strip():
            return ch_key, []

        # Extract line ranges using LLM
        try:
            line_ranges = extract_line_ranges_llm(client, ch_num, ch_text, questions_model_id)
        except Exception as e:
            logger.error(f"Error extracting {ch_key}: {e}")
            return ch_key, []

        if not line_ranges:
            return ch_key, []

        # Extract raw text for each question
        raw_questions = []
        for lr in line_ranges:
            q_id = lr.get("question_id", "?")
            q_start = lr.get("question_start", 0)
            q_end = lr.get("question_end", 0)
            a_start = lr.get("answer_start", 0)
            a_end = lr.get("answer_end", 0)

            if not all([q_start, q_end]):
                continue

            q_text = extract_lines_by_range_mapped(lines_with_images, q_start, q_end, line_mapping) if q_start > 0 else ""
            a_text = extract_lines_by_range_mapped(lines_with_images, a_start, a_end, line_mapping) if a_start > 0 else ""

            # Get images from the LLM's response
            image_files = lr.get("image_files", [])

            raw_questions.append({
                "question_id": q_id,
                "local_id": q_id,
                "full_id": f"{ch_key}_{q_id}",
                "chapter": ch_num,
                "question_text": q_text,
                "answer_text": a_text,
                "image_files": image_files,
                "correct_letter": lr.get("correct_letter", "")
            })

        return ch_key, raw_questions

    # Process chapters in parallel
    questions_max_workers = workers.get('questions', 10)
    completed = 0

    with ThreadPoolExecutor(max_workers=questions_max_workers) as executor:
        futures = {executor.submit(extract_chapter_questions, idx): idx for idx in range(total_chapters)}

        for future in as_completed(futures):
            idx = futures[future]
            ch_num = chapters[idx].get("chapter_number", idx + 1)

            try:
                ch_key, raw_questions = future.result()
                if raw_questions:
                    results[ch_key] = raw_questions
                    # Save incrementally
                    st.session_state.raw_questions = results
                    save_raw_questions()
            except Exception as e:
                logger.error(f"Error extracting chapter {ch_num}: {e}")

            completed += 1
            progress_step3.progress(completed / total_chapters, f"Extracted {completed}/{total_chapters} chapters...")

    st.session_state.raw_questions = results
    save_raw_questions()

    total_raw = sum(len(qs) for qs in results.values())
    progress_step3.progress(100, "Questions extracted!")
    st.success(f"Step 3 complete: {total_raw} raw Q&A pairs from {len(results)} chapters")

    # Step 4: Format Questions
    st.markdown("### Step 4: Formatting Questions...")
    progress_step4 = st.progress(0)

    format_model_id = get_model_id(models['format'])
    format_max_workers = workers.get('format', 10)

    raw_questions = st.session_state.raw_questions
    formatted_questions = {}
    total_questions = sum(len(qs) for qs in raw_questions.values())

    # Build flat list of all questions with chapter keys
    all_raw = []
    for ch_key in sort_chapter_keys(list(raw_questions.keys())):
        for rq in raw_questions[ch_key]:
            all_raw.append((ch_key, rq))

    def format_single(item):
        ch_key, rq = item
        try:
            formatted = format_qa_pair_llm(
                client,
                rq["local_id"],
                rq["question_text"],
                rq["answer_text"],
                format_model_id,
                rq["chapter"]
            )
            formatted["full_id"] = rq["full_id"]
            formatted["local_id"] = rq["local_id"]
            formatted["image_files"] = rq["image_files"]
            if not formatted.get("correct_answer") and rq.get("correct_letter"):
                formatted["correct_answer"] = rq["correct_letter"]
            return ch_key, formatted
        except Exception as e:
            logger.error(f"Error formatting {rq['full_id']}: {e}")
            return ch_key, {
                "full_id": rq["full_id"],
                "local_id": rq["local_id"],
                "text": rq["question_text"],
                "choices": {},
                "correct_answer": rq.get("correct_letter", ""),
                "explanation": rq["answer_text"],
                "image_files": rq["image_files"],
                "error": str(e)
            }

    processed = 0
    with ThreadPoolExecutor(max_workers=format_max_workers) as executor:
        futures = {executor.submit(format_single, item): item for item in all_raw}

        for future in as_completed(futures):
            try:
                ch_key, formatted = future.result()
                if ch_key not in formatted_questions:
                    formatted_questions[ch_key] = []
                formatted_questions[ch_key].append(formatted)
            except Exception as e:
                ch_key, rq = futures[future]
                logger.error(f"Error formatting {rq['full_id']}: {e}")

            processed += 1
            progress_step4.progress(processed / total_questions, f"Formatted {processed}/{total_questions} questions...")

            # Save incrementally every 10 questions
            if processed % 10 == 0:
                st.session_state.questions = formatted_questions
                save_questions()

    # Sort questions within each chapter
    for ch_key in formatted_questions:
        formatted_questions[ch_key].sort(key=lambda q: question_sort_key(q["full_id"]))

    st.session_state.questions = formatted_questions
    save_questions()

    # Build image assignments
    st.session_state.image_assignments = {}
    for ch_key, qs in formatted_questions.items():
        for q in qs:
            for img_file in q.get("image_files", []):
                st.session_state.image_assignments[img_file] = q["full_id"]
    save_image_assignments()

    total_formatted = sum(len(qs) for qs in formatted_questions.values())
    progress_step4.progress(100, "Questions formatted!")
    st.success(f"Step 4 complete: {total_formatted} formatted Q&A pairs")

    # Step 5: Associate Context
    st.markdown("### Step 5: Associating Context...")
    progress_step5 = st.progress(0)

    context_model_id = get_model_id(models['context'])
    context_max_workers = workers.get('context', 10)

    questions_copy = copy.deepcopy(st.session_state.questions)
    assignments_copy = copy.deepcopy(st.session_state.image_assignments)

    total_chapters = len(questions_copy)

    # Worker function for parallel context association
    def process_chapter_context(ch_key: str, ch_questions: list) -> tuple:
        """Process context for one chapter. Returns (ch_key, updated_questions, ch_stats)."""
        ch_stats = {"context_questions_found": 0, "sub_questions_updated": 0, "images_copied": 0}

        if not ch_questions:
            return ch_key, ch_questions, ch_stats

        # Build summary for LLM with rich context
        questions_summary = []
        for q in ch_questions:
            choices = q.get("choices", {})
            has_choices = bool(choices)
            summary = {
                "full_id": q["full_id"],
                "local_id": q["local_id"],
                "text_preview": q["text"][:300] + "..." if len(q["text"]) > 300 else q["text"],
                "has_choices": has_choices,
                "num_choices": len(choices),
                "has_correct_answer": bool(q.get("correct_answer")),
                "has_explanation": bool(q.get("explanation"))
            }
            questions_summary.append(summary)

        prompt = get_prompt("associate_context",
                           questions_summary=json.dumps(questions_summary, indent=2))

        try:
            response_text, usage = stream_message(
                client,
                context_model_id,
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

                # Only validation: filter out self-references (question inheriting from itself)
                sub_ids = [sid for sid in sub_ids if sid != context_id]
                if not sub_ids:
                    continue

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

    with ThreadPoolExecutor(max_workers=context_max_workers) as executor:
        future_to_ch = {
            executor.submit(process_chapter_context, ch_key, ch_questions): ch_key
            for ch_key, ch_questions in questions_copy.items()
        }

        for future in as_completed(future_to_ch):
            ch_key = future_to_ch[future]
            try:
                result_ch_key, result_questions, ch_stats = future.result()
                updated_questions[result_ch_key] = result_questions
                for key in total_stats:
                    total_stats[key] += ch_stats[key]
                # Save incrementally after each chapter completes
                st.session_state.questions_merged[result_ch_key] = result_questions
                save_questions_merged()
            except Exception as e:
                logger.error(f"Context association {ch_key}: Failed - {e}")
                updated_questions[ch_key] = questions_copy[ch_key]

            completed += 1
            progress_step5.progress(completed / total_chapters, f"Context: {completed}/{total_chapters} chapters...")

    st.session_state.questions_merged = updated_questions
    st.session_state.image_assignments_merged = assignments_copy

    # Detect page numbers for merged questions
    if st.session_state.pages and st.session_state.chapters:
        add_page_numbers_to_questions(
            st.session_state.questions_merged,
            st.session_state.pages,
            st.session_state.chapters
        )

    save_questions_merged()
    save_image_assignments_merged()

    progress_step5.progress(100, "Context associated!")
    st.success(
        f"Step 5 complete: Context association finished\n\n"
        f"- Context questions found: {total_stats['context_questions_found']}\n"
        f"- Sub-questions updated: {total_stats['sub_questions_updated']}\n"
        f"- Images copied: {total_stats['images_copied']}"
    )

    # Play completion sound
    play_completion_sound()

    return True


def render_source_step():
    """Render source PDF selection step."""

    # Check if we're in "feeling lucky" mode
    if st.session_state.get("feeling_lucky_mode"):
        st.header("I'm Feeling Lucky - Automatic Processing")

        selected_pdf = st.session_state.get("feeling_lucky_pdf")
        pdf_path = st.session_state.get("feeling_lucky_pdf_path")

        st.info(f"Selected PDF: {selected_pdf}")

        # Model selection for each step
        model_options = get_model_options()

        default_idx = 0
        if st.session_state.selected_model in model_options:
            default_idx = model_options.index(st.session_state.selected_model)

        st.markdown("**Select model and parallel workers for each step:**")

        # Step 2: Chapters (no parallelization)
        st.markdown("##### Step 2 - Extract Chapters")
        chapters_model = st.selectbox(
            "Model:",
            model_options,
            index=default_idx,
            key="lucky_chapters_model"
        )

        # Step 3: Questions (parallel by chapter)
        st.markdown("##### Step 3 - Extract Questions")
        col3a, col3b = st.columns([3, 1])
        with col3a:
            questions_model = st.selectbox(
                "Model:",
                model_options,
                index=default_idx,
                key="lucky_questions_model"
            )
        with col3b:
            questions_workers = st.number_input(
                "Workers:",
                min_value=1,
                max_value=50,
                value=20,
                key="lucky_questions_workers",
                help="Parallel chapters to process"
            )

        # Step 4: Format (parallel by question)
        st.markdown("##### Step 4 - Format Questions")
        col4a, col4b = st.columns([3, 1])
        with col4a:
            format_model = st.selectbox(
                "Model:",
                model_options,
                index=default_idx,
                key="lucky_format_model"
            )
        with col4b:
            format_workers = st.number_input(
                "Workers:",
                min_value=1,
                max_value=50,
                value=50,
                key="lucky_format_workers",
                help="Parallel questions to format"
            )

        # Step 5: Context (parallel by chapter)
        st.markdown("##### Step 5 - Associate Context")
        col5a, col5b = st.columns([3, 1])
        with col5a:
            context_model = st.selectbox(
                "Model:",
                model_options,
                index=default_idx,
                key="lucky_context_model"
            )
        with col5b:
            context_workers = st.number_input(
                "Workers:",
                min_value=1,
                max_value=50,
                value=50,
                key="lucky_context_workers",
                help="Parallel chapters to process"
            )

        st.markdown("---")

        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            if st.button("Start Automatic Processing", type="primary"):
                # Clear any existing data and start fresh
                st.session_state.current_pdf = selected_pdf
                clear_session_data()
                reset_logger()

                models = {
                    'chapters': chapters_model,
                    'questions': questions_model,
                    'format': format_model,
                    'context': context_model
                }
                workers = {
                    'questions': questions_workers,
                    'format': format_workers,
                    'context': context_workers
                }

                success = run_feeling_lucky(pdf_path, models, workers)

                if success:
                    st.balloons()
                    st.success("All steps completed! Ready for QC.")
                    st.session_state.feeling_lucky_mode = False
                    st.session_state.current_step = "qc"
                    save_settings()
                    st.rerun()

        with btn_col2:
            if st.button("Cancel", type="secondary"):
                st.session_state.feeling_lucky_mode = False
                st.rerun()

        return

    col1, col2 = st.columns([4, 1])
    with col1:
        st.header("Step 1: Select Source PDF")
    with col2:
        if st.button("Reset To This Step", key="clear_source", type="secondary", help="Clear this step and all subsequent steps"):
            clear_step_data("source")
            st.success("All data cleared")
            st.rerun()

    pdf_files = list(Path(SOURCE_DIR).glob("*.pdf")) if os.path.exists(SOURCE_DIR) else []

    if not pdf_files:
        st.warning(f"No PDF files found in '{SOURCE_DIR}/' directory. Please add a PDF file.")
        return

    pdf_options = sorted([f.name for f in pdf_files])
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

        col1, col2, col3 = st.columns(3)

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

        with col3:
            if st.button("I'm Feeling Lucky", type="secondary", help="Run all steps automatically up to QC"):
                st.session_state.feeling_lucky_mode = True
                st.session_state.feeling_lucky_pdf = selected_pdf
                st.session_state.feeling_lucky_pdf_path = pdf_path
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
    col1, col2 = st.columns([4, 1])
    with col1:
        st.header("Step 2: Extract Chapters")
    with col2:
        if st.button("Reset To This Step", key="clear_chapters", type="secondary", help="Clear this step and all subsequent steps"):
            clear_step_data("chapters")
            st.success("Chapters and subsequent data cleared")
            st.rerun()

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
                    # Sort chapters by chapter_number to ensure correct ordering
                    chapters = sorted(chapters, key=lambda ch: ch.get("chapter_number", 0))
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
                play_completion_sound()
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
    col1, col2 = st.columns([4, 1])
    with col1:
        st.header("Step 3: Extract Questions")
    with col2:
        if st.button("Reset To This Step", key="clear_questions", type="secondary", help="Clear this step and all subsequent steps"):
            clear_step_data("questions")
            st.success("Questions and subsequent data cleared")
            st.rerun()

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
            value=20,
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
                    play_completion_sound()
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
                        # Save incrementally after each chapter completes
                        st.session_state.raw_questions[ch_key] = raw_questions
                        save_raw_questions()
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

            # Final save (in case session state differs from results)
            st.session_state.raw_questions = results
            save_raw_questions()
            status_text.text("Done!")
            chapter_status.empty()

            total_raw = sum(len(qs) for qs in results.values())
            st.success(f"Extracted {total_raw} raw Q&A pairs from {total_chapters} chapters")
            play_completion_sound()
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
        ch_options = sort_chapter_keys(raw_questions.keys())
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
    col1, col2 = st.columns([4, 1])
    with col1:
        st.header("Step 4: Format Questions")
    with col2:
        if st.button("Reset To This Step", key="clear_format", type="secondary", help="Clear this step and all subsequent steps"):
            clear_step_data("format")
            st.success("Format and subsequent data cleared")
            st.rerun()

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

    # Chapter selector
    if not st.session_state.chapters:
        st.warning("No chapters available. Please extract chapters first (Step 2)")
        return

    chapter_options = [f"Ch{ch['chapter_number']}: {ch['title'][:40]}..."
                      for ch in st.session_state.chapters]
    selected_ch_idx = st.selectbox("Select chapter:",
                                    range(len(chapter_options)),
                                    format_func=lambda x: chapter_options[x],
                                    key="format_ch_selector")

    selected_ch = st.session_state.chapters[selected_ch_idx]
    selected_ch_num = selected_ch["chapter_number"]
    selected_ch_key = f"ch{selected_ch_num}"

    # Count raw questions for selected chapter
    ch_raw_questions = raw_questions.get(selected_ch_key, [])
    ch_raw_count = len(ch_raw_questions)

    # Model selection and parallel workers
    model_col, workers_col, btn_col1, btn_col2 = st.columns([2, 1.5, 2, 2])

    with model_col:
        model_options = get_model_options()
        current_idx = model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
        selected_model = st.selectbox("Model:", model_options, index=current_idx, key="format_model")
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            save_settings()

    with workers_col:
        max_workers = st.number_input(
            "Parallel workers:",
            min_value=1,
            max_value=100,
            value=50,
            help="Tier 1: use 5-10 | Tier 2+: use 20-50 | Tier 4: use 50-100",
            key="format_workers"
        )

    # Capture model_id before threads start (session state not accessible in worker threads)
    model_id = get_selected_model_id()

    # Helper function to format a single question
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

    with btn_col1:
        format_chapter = st.button(
            f"Format Chapter {selected_ch_num} ({ch_raw_count})",
            type="primary",
            key="format_chapter_btn",
            disabled=ch_raw_count == 0
        )

    with btn_col2:
        format_all = st.button("Format ALL Questions", type="secondary", key="format_all_btn")

    # Single chapter formatting logic
    if format_chapter:
        if ch_raw_count == 0:
            st.warning(f"No raw questions in Chapter {selected_ch_num}")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Collect raw questions for this chapter
            ch_raw = [(selected_ch_key, rq) for rq in ch_raw_questions]
            total = len(ch_raw)
            status_text.text(f"Formatting {total} Q&A pairs from Chapter {selected_ch_num}...")

            formatted_list = []
            completed = 0

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(format_single, item): item for item in ch_raw}

                for future in as_completed(futures):
                    try:
                        ch_key, formatted = future.result()
                        formatted_list.append(formatted)
                    except Exception as e:
                        ch_key, rq = futures[future]
                        logger.error(f"Error formatting {rq['full_id']}: {e}")
                        formatted_list.append({
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
                    status_text.text(f"Chapter {selected_ch_num}: {completed}/{total} formatted...")

            # Sort and save
            formatted_list.sort(key=lambda q: question_sort_key(q["full_id"]))
            st.session_state.questions[selected_ch_key] = formatted_list
            save_questions()

            # Build image assignments for this chapter
            for q in formatted_list:
                for img_file in q.get("image_files", []):
                    st.session_state.image_assignments[img_file] = q["full_id"]
            save_image_assignments()

            status_text.text("Done!")
            st.success(f"Formatted {total} Q&A pairs from Chapter {selected_ch_num}")
            play_completion_sound()
            st.rerun()

    # All chapters formatting logic
    if format_all:
        progress_bar = st.progress(0)
        status_text = st.empty()

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

                # Save incrementally every 10 questions
                if completed % 10 == 0:
                    st.session_state.questions = formatted_by_chapter
                    save_questions()

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
        play_completion_sound()
        st.info("**Next:** Go to **Step 5: Associate Context** to link context questions to sub-questions.")
        st.rerun()

    # Display formatted questions if available
    if st.session_state.questions:
        st.markdown("---")
        st.subheader("Formatted Questions")

        ch_options = sort_chapter_keys(st.session_state.questions.keys())
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
    col1, col2 = st.columns([4, 1])
    with col1:
        st.header("Step 5: Associate Context")
    with col2:
        if st.button("Reset To This Step", key="clear_context", type="secondary", help="Clear this step and all subsequent steps"):
            clear_step_data("context")
            st.success("Context and subsequent data cleared")
            st.rerun()

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

    # Count questions and pre-extracted context
    context_only_count = 0
    pre_extracted_context_count = 0
    total_questions = 0
    for ch_key, ch_questions in st.session_state.questions.items():
        for q in ch_questions:
            total_questions += 1
            if not q.get("choices"):
                context_only_count += 1
            if q.get("context_source"):
                pre_extracted_context_count += 1

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Questions", total_questions)
    with col2:
        st.metric("Questions Without Choices", context_only_count)
    with col3:
        st.metric("Pre-extracted Context Links", pre_extracted_context_count,
                  help="Questions with context_source already identified during extraction")

    # Controls section
    st.markdown("---")

    # Show different options based on whether context was pre-extracted
    if pre_extracted_context_count > 0:
        st.success(f"Context relationships were detected during extraction for {pre_extracted_context_count} questions.")
        st.markdown("You can apply these directly without an additional LLM call, or run the LLM to re-analyze.")

        ctrl_col1, ctrl_col2 = st.columns(2)
        with ctrl_col1:
            apply_extracted = st.button("Apply Extracted Context", type="primary",
                                        help="Use context_source from extraction (no LLM call)")
        with ctrl_col2:
            run_context_association = st.button("Run LLM Association",
                                                help="Re-analyze with LLM (ignores pre-extracted)")

        # LLM options (collapsed by default since extracted is preferred)
        with st.expander("LLM Association Options", expanded=False):
            llm_col1, llm_col2 = st.columns(2)
            with llm_col1:
                model_options = get_model_options()
                current_idx = model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
                selected_model = st.selectbox("Model:", model_options, index=current_idx, key="context_model")
                if selected_model != st.session_state.selected_model:
                    st.session_state.selected_model = selected_model
                    save_settings()
            with llm_col2:
                context_workers = st.number_input(
                    "Parallel workers:",
                    min_value=1,
                    max_value=50,
                    value=50,
                    help="Number of chapters to process in parallel",
                    key="context_workers"
                )
    else:
        st.info("No context relationships were detected during extraction. Run LLM association to identify them.")
        apply_extracted = False

        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1.5, 2])

        with ctrl_col1:
            model_options = get_model_options()
            current_idx = model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
            selected_model = st.selectbox("Model:", model_options, index=current_idx, key="context_model")
            if selected_model != st.session_state.selected_model:
                st.session_state.selected_model = selected_model
                save_settings()

        with ctrl_col2:
            context_workers = st.number_input(
                "Parallel workers:",
                min_value=1,
                max_value=50,
                value=50,
                help="Number of chapters to process in parallel",
                key="context_workers"
            )

        with ctrl_col3:
            run_context_association = st.button("Associate Context", type="primary")

    st.markdown("---")

    # Preview potential context-only questions (before association)
    if context_only_count > 0:
        potential_context = []
        for ch_key in sort_chapter_keys(st.session_state.questions.keys()):
            for q in st.session_state.questions[ch_key]:
                if not q.get("choices"):
                    potential_context.append((ch_key, q))

        with st.expander(f"Context-Only Questions Preview ({len(potential_context)} questions)", expanded=False):
            # Chapter filter for context questions
            context_chapters = sorted(set(ch_key for ch_key, _ in potential_context),
                                      key=lambda x: int(x.replace("ch", "")))
            filter_col1, filter_col2 = st.columns([1, 3])
            with filter_col1:
                ctx_chapter_filter = st.selectbox(
                    "Filter by chapter:",
                    ["All chapters"] + context_chapters,
                    key="ctx_preview_chapter_filter"
                )

            # Filter questions by chapter
            filtered_context = [(ch, q) for ch, q in potential_context
                               if ctx_chapter_filter == "All chapters" or ch == ctx_chapter_filter]

            st.caption(f"Showing {len(filtered_context)} of {len(potential_context)} context questions")

            # Display each context question in a card-like format
            for idx, (ch_key, q) in enumerate(filtered_context):
                # Get images assigned to this question
                q_images = [img for img in st.session_state.images
                           if st.session_state.image_assignments.get(img["filename"]) == q["full_id"]]

                # Card container - use divider and container instead of nested expander
                st.markdown("---")
                img_indicator = f" [{len(q_images)} img]" if q_images else ""
                st.markdown(f"**{q['full_id']}**{img_indicator}")

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**Question Text:**")
                    st.markdown(q.get("text", "No text"))

                    # Show potential sub-questions that would inherit this context
                    # A sub-question of "1" is "1a", "1b", etc. (NOT "10", "11")
                    # The character immediately after the context ID must be a letter
                    q_id = q.get("local_id", "")
                    sub_questions = []
                    for sq in st.session_state.questions.get(ch_key, []):
                        sq_id = sq.get("local_id", "")
                        if (sq_id.startswith(q_id) and
                            len(sq_id) > len(q_id) and
                            sq_id[len(q_id)].isalpha()):
                            sub_questions.append(sq)
                    if sub_questions:
                        st.markdown(f"**Sub-questions:** {', '.join(sq.get('local_id', '?') for sq in sub_questions)}")

                with col2:
                    if q_images:
                        for img in q_images:
                            if os.path.exists(img["filepath"]):
                                st.image(img["filepath"], caption=f"Page {img['page']}", use_container_width=True)
                    else:
                        st.caption("No images assigned")

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
                ["All chapters"] + sort_chapter_keys(st.session_state.questions_merged.keys()),
                key="context_chapter_filter"
            )

        # Context-only questions preview section
        context_only_questions = []
        for ch_key in sort_chapter_keys(st.session_state.questions_merged.keys()):
            for q in st.session_state.questions_merged[ch_key]:
                if q.get("is_context_only"):
                    context_only_questions.append((ch_key, q))

        if context_only_questions:
            with st.expander(f"Context-Only Questions ({len(context_only_questions)} total) - These will NOT become Anki cards", expanded=False):
                assignments_merged = st.session_state.image_assignments_merged if st.session_state.image_assignments_merged else st.session_state.image_assignments

                # Chapter filter for context-only
                ctx_merged_chapters = sorted(set(ch for ch, _ in context_only_questions),
                                            key=lambda x: int(x.replace("ch", "")))
                ctx_filter = st.selectbox(
                    "Filter:",
                    ["All chapters"] + ctx_merged_chapters,
                    key="ctx_merged_filter"
                )

                filtered_ctx = [(ch, q) for ch, q in context_only_questions
                               if ctx_filter == "All chapters" or ch == ctx_filter]

                for ch_key, q in filtered_ctx:
                    q_images = [img for img in st.session_state.images
                               if assignments_merged.get(img["filename"]) == q["full_id"]]

                    img_indicator = f" [{len(q_images)} img]" if q_images else ""

                    # Use container with divider instead of nested expander
                    st.markdown(f"---")
                    st.markdown(f"**{q['full_id']}**{img_indicator}")
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown(q.get("text", "No text")[:300] + "..." if len(q.get("text", "")) > 300 else q.get("text", "No text"))

                        # Show which sub-questions inherited this context
                        inherited_by = [sq["local_id"] for sq in st.session_state.questions_merged.get(ch_key, [])
                                       if sq.get("context_from") == q["full_id"]]
                        if inherited_by:
                            st.success(f"Context inherited by: {', '.join(inherited_by)}")

                    with col2:
                        if q_images:
                            for img in q_images:
                                if os.path.exists(img["filepath"]):
                                    st.image(img["filepath"], caption=f"Page {img['page']}", use_container_width=True)

        st.subheader("Merged Questions Preview")

        assignments_to_use = st.session_state.image_assignments_merged if st.session_state.image_assignments_merged else st.session_state.image_assignments

        all_merged_questions = []
        for ch_key in sort_chapter_keys(st.session_state.questions_merged.keys()):
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

    # Button processing logic (button rendered at top of page)
    if apply_extracted:
        # Apply pre-extracted context without LLM call
        from llm_extraction import apply_extracted_context

        status_text = st.empty()
        status_text.text("Applying pre-extracted context relationships...")

        questions_copy = copy.deepcopy(st.session_state.questions)
        assignments_copy = copy.deepcopy(st.session_state.image_assignments)

        updated_questions, updated_assignments, stats = apply_extracted_context(
            questions_copy, assignments_copy
        )

        # Save results
        st.session_state.questions_merged = updated_questions
        st.session_state.image_assignments = updated_assignments
        save_questions_merged()
        save_image_assignments()

        status_text.empty()
        st.success(
            f"Applied extracted context: {stats['context_questions_found']} context questions identified, "
            f"{stats['sub_questions_updated']} sub-questions updated"
        )
        st.rerun()

    if run_context_association:
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

                # Build summary for LLM with rich context
                questions_summary = []
                for q in ch_questions:
                    choices = q.get("choices", {})
                    has_choices = bool(choices)
                    summary = {
                        "full_id": q["full_id"],
                        "local_id": q["local_id"],
                        "text_preview": q["text"][:300] + "..." if len(q["text"]) > 300 else q["text"],
                        "has_choices": has_choices,
                        "num_choices": len(choices),
                        "has_correct_answer": bool(q.get("correct_answer")),
                        "has_explanation": bool(q.get("explanation"))
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

                        # Only validation: filter out self-references
                        sub_ids = [sid for sid in sub_ids if sid != context_id]
                        if not sub_ids:
                            continue

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
                        # Save incrementally after each chapter completes
                        st.session_state.questions_merged[result_ch_key] = result_questions
                        save_questions_merged()
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

            status_text.text("Saving final merged data...")

            st.session_state.questions_merged = updated_questions
            st.session_state.image_assignments_merged = assignments_copy

            # Detect page numbers for merged questions
            if st.session_state.pages and st.session_state.chapters:
                status_text.text("Detecting page numbers...")
                add_page_numbers_to_questions(
                    st.session_state.questions_merged,
                    st.session_state.pages,
                    st.session_state.chapters
                )

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
            play_completion_sound()

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
    col1, col2 = st.columns([4, 1])
    with col1:
        st.header("Step 6: QC Questions")
    with col2:
        if st.button("Reset To This Step", key="clear_qc", type="secondary", help="Clear this step and all subsequent steps"):
            clear_step_data("qc")
            st.success("QC and export data cleared")
            st.rerun()

    # Use merged questions if available (after Step 5), otherwise use formatted questions (Step 4)
    questions_source = st.session_state.questions_merged if st.session_state.questions_merged else st.session_state.questions

    if not questions_source:
        st.warning("Please format questions first (Step 4)")
        return

    all_questions = []
    for ch_key, questions in questions_source.items():
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
                                       ["All chapters"] + sort_chapter_keys(questions_source.keys()))
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
                        # Detect pages for original questions
                        add_page_numbers_to_questions(
                            st.session_state.questions,
                            st.session_state.pages,
                            st.session_state.chapters
                        )
                        save_questions()
                        # Also detect pages for merged questions if they exist
                        if st.session_state.questions_merged:
                            add_page_numbers_to_questions(
                                st.session_state.questions_merged,
                                st.session_state.pages,
                                st.session_state.chapters
                            )
                            save_questions_merged()
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
                                st.image(filepath, caption=f"Page {img['page']} - {img['filename']}", use_container_width=True)

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
                                # Also detect for merged questions
                                if st.session_state.questions_merged:
                                    add_page_numbers_to_questions(
                                        st.session_state.questions_merged,
                                        st.session_state.pages,
                                        st.session_state.chapters
                                    )
                                    save_questions_merged()
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
                                        st.image(png_bytes, caption=f"Page {page_num}", use_container_width=True)
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
                                        st.image(png_bytes, caption=f"Page {page_num}", use_container_width=True)
                                    else:
                                        st.error(f"Failed to render page {page_num}")
                            else:
                                st.caption("Page not detected")


# =============================================================================
# Step 7: Export
# =============================================================================

def generate_anki_deck(book_name: str, questions: dict, chapters: list, image_assignments: dict,
                       images: list, include_images: bool, only_approved: bool, qc_progress: dict,
                       generated_questions: dict = None, include_generated: bool = False) -> str:
    """Generate Anki deck and return path to .apkg file.

    Args:
        book_name: Name for the deck
        questions: Dict of chapter questions
        chapters: List of chapter info
        image_assignments: Dict mapping image files to question IDs
        images: List of image metadata
        include_images: Whether to include images in cards
        only_approved: Only export QC-approved questions
        qc_progress: QC progress dict
        generated_questions: Optional dict of generated cloze cards
        include_generated: Whether to include generated cloze cards
    """
    import genanki
    import hashlib

    # Generate stable IDs based on name
    def stable_id(name: str) -> int:
        return int(hashlib.md5(name.encode()).hexdigest()[:8], 16)

    model_id = stable_id(f"{book_name}_model")
    cloze_model_id = stable_id(f"{book_name}_cloze_model")

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
            {'name': 'Source'},
        ],
        templates=[
            {
                'name': 'Card 1',
                'qfmt': '''
                    <div class="card-content">
                        <div class="question">{{Question}}</div>
                        {{#Image}}<div class="image">{{Image}}</div>{{/Image}}
                        <div class="choices">{{Choices}}</div>
                    </div>
                ''',
                'afmt': '''
                    <div class="card-content">
                        <div class="question">{{Question}}</div>
                        {{#Image}}<div class="image">{{Image}}</div>{{/Image}}
                        <div class="choices">{{Choices}}</div>
                        <hr class="answer-divider">
                        <div class="answer">{{Answer}}</div>
                        <div class="explanation">{{Explanation}}</div>
                        {{#Source}}<div class="source-ref">{{Source}}</div>{{/Source}}
                    </div>
                ''',
            },
        ],
        css='''
            .card {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                font-size: 17px;
                line-height: 1.5;
                color: #2c3e50;
                background: #fafafa;
            }
            .card-content {
                max-width: 650px;
                margin: 0 auto;
                padding: 25px;
                text-align: center;
            }
            .question {
                font-size: 18px;
                font-weight: 500;
                margin-bottom: 20px;
                text-align: left;
                line-height: 1.6;
            }
            .image {
                margin: 20px 0;
            }
            .image img {
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                box-shadow: 0 2px 12px rgba(0,0,0,0.12);
            }
            .choices {
                text-align: left;
                margin: 20px 0;
            }
            .choice {
                margin: 10px 0;
                padding: 10px 14px;
                background: #ffffff;
                color: #2c3e50;
                border-radius: 6px;
                border: 1px solid #e1e5e9;
                border-left: 4px solid #3498db;
            }
            .choice-letter {
                font-weight: 700;
                color: #3498db;
                margin-right: 10px;
            }
            .answer-divider {
                border: none;
                border-top: 2px solid #e1e5e9;
                margin: 25px 0;
            }
            .answer {
                font-size: 20px;
                font-weight: 600;
                color: #27ae60;
                margin: 20px 0;
                padding: 12px 20px;
                background: linear-gradient(135deg, #d5f4e0 0%, #c8f0d8 100%);
                border-radius: 8px;
                display: inline-block;
            }
            .explanation {
                text-align: left;
                margin-top: 20px;
                padding: 18px;
                background: #ffffff;
                border-radius: 8px;
                border: 1px solid #e1e5e9;
                line-height: 1.7;
                color: #34495e !important;
            }
            .source-ref {
                margin-top: 20px;
                padding: 10px 14px;
                background: #f8f9fa;
                border-radius: 6px;
                font-size: 13px;
                color: #6c757d !important;
                text-align: left;
            }
            .source-ref-label {
                font-weight: 600;
                color: #495057 !important;
            }
        '''
    )

    # Create cloze model for generated cards
    cloze_model = genanki.Model(
        cloze_model_id,
        f'{book_name} Cloze Model',
        model_type=genanki.Model.CLOZE,
        fields=[
            {'name': 'Text'},
            {'name': 'Extra'},
            {'name': 'Source'},
        ],
        templates=[
            {
                'name': 'Cloze Card',
                'qfmt': '''
                    <div class="cloze-card">
                        <div class="cloze-text">{{cloze:Text}}</div>
                    </div>
                ''',
                'afmt': '''
                    <div class="cloze-card">
                        <div class="cloze-text">{{cloze:Text}}</div>
                        {{#Extra}}
                        <hr class="cloze-divider">
                        <div class="cloze-extra">{{Extra}}</div>
                        {{/Extra}}
                        {{#Source}}<div class="cloze-source">{{Source}}</div>{{/Source}}
                    </div>
                ''',
            },
        ],
        css='''
            .card {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                font-size: 18px;
                line-height: 1.6;
                color: #2c3e50;
                background: #fafafa;
            }
            .cloze-card {
                max-width: 650px;
                margin: 0 auto;
                padding: 25px;
            }
            .cloze-text {
                font-size: 18px;
                line-height: 1.7;
                text-align: left;
            }
            .cloze {
                font-weight: 700;
                color: #3498db;
            }
            .cloze-divider {
                border: none;
                border-top: 2px solid #e1e5e9;
                margin: 20px 0;
            }
            .cloze-extra {
                font-size: 15px;
                color: #5d6d7e;
                text-align: left;
                padding: 12px;
                background: #f8f9fa;
                border-radius: 6px;
            }
            .cloze-source {
                margin-top: 15px;
                font-size: 12px;
                color: #95a5a6;
                text-align: left;
            }
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

        # Create chapter sub-deck for extracted questions
        ch_deck_name = f"{book_name}::{ch_num}. {ch_title}::Extracted"
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

            # Build choices HTML with styled formatting
            choices = q.get('choices', {})
            if choices:
                choice_items = []
                for letter, text in sorted(choices.items()):
                    choice_items.append(
                        f'<div class="choice"><span class="choice-letter">{letter}.</span>{text}</div>'
                    )
                choices_html = ''.join(choice_items)
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

            # Build source reference
            local_id = q.get('local_id', q_id.split('_')[-1])
            source_parts = [f'<span class="source-ref-label">Source:</span> Chapter {ch_num} ({ch_title}), Question {local_id}']

            # Add page numbers if available
            question_page = q.get('question_page')
            answer_page = q.get('answer_page')
            if question_page and answer_page:
                source_parts.append(f'(Q: p.{question_page}, A: p.{answer_page})')
            elif question_page:
                source_parts.append(f'(p. {question_page})')
            elif answer_page:
                source_parts.append(f'(A: p.{answer_page})')

            if q.get('context_from'):
                context_local_id = q['context_from'].split('_')[-1]
                source_parts.append(f'[context from Q{context_local_id}]')

            source_ref = ' '.join(source_parts)

            # Create note
            note = genanki.Note(
                model=model,
                fields=[q_text, choices_html, correct, explanation, image_html, ch_title, source_ref],
                tags=[f"chapter{ch_num}"]
            )
            ch_deck.notes.append(note)

        # Add chapter deck to list if it has notes
        if ch_deck.notes:
            all_decks.append(ch_deck)

    # Add generated cloze cards if requested (organized by chapter)
    if include_generated and generated_questions:
        generated_cards = generated_questions.get("generated_cards", {})

        if generated_cards:
            # generated_cards is keyed by chapter (e.g., "ch4"), each value is a list of cards
            # Each card has source_question_id (e.g., "ch4_2a") for the source question

            # Build question lookup to get source explanations
            question_lookup = {}
            for ch_key_q, ch_qs in questions.items():
                for q in ch_qs:
                    question_lookup[q['full_id']] = q

            # Create a generated sub-deck for each chapter
            for ch in chapters:
                ch_num = str(ch['chapter_number'])
                ch_key = f"ch{ch_num}"
                ch_title = ch.get('title', f'Chapter {ch_num}')

                ch_cards = generated_cards.get(ch_key, [])
                if not ch_cards:
                    continue

                # Create chapter::Generated sub-deck
                gen_deck_name = f"{book_name}::{ch_num}. {ch_title}::Generated"
                gen_deck_id = stable_id(gen_deck_name)
                gen_deck = genanki.Deck(gen_deck_id, gen_deck_name)

                for card in ch_cards:
                    cloze_text = card.get('cloze_text', '')
                    if not cloze_text:
                        continue

                    # Get source question ID from card data
                    source_q_id = card.get('source_question_id', '')
                    local_id = source_q_id.split('_')[-1] if '_' in source_q_id else source_q_id

                    # Look up source question to get explanation
                    source_q = question_lookup.get(source_q_id, {})
                    source_explanation = source_q.get('explanation', '')

                    # Extra info: learning point, category, confidence, and source explanation
                    extra_parts = []
                    if card.get('learning_point'):
                        extra_parts.append(f"<b>Learning point:</b> {card['learning_point']}")
                    if card.get('category'):
                        extra_parts.append(f"<b>Category:</b> {card['category']}")
                    if card.get('confidence'):
                        extra_parts.append(f"<b>Confidence:</b> {card['confidence']}")
                    if source_explanation:
                        extra_parts.append(f"<hr><b>Source:</b><br>{source_explanation}")
                    extra = '<br>'.join(extra_parts)

                    # Source reference
                    source_ref = f"Generated from Q{local_id}"

                    # Create cloze note
                    cloze_note = genanki.Note(
                        model=cloze_model,
                        fields=[cloze_text, extra, source_ref],
                        tags=[f"chapter{ch_num}", "generated", card.get('category', 'general')]
                    )
                    gen_deck.notes.append(cloze_note)

                if gen_deck.notes:
                    all_decks.append(gen_deck)

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


def render_generate_step():
    """Render the cloze card generation step."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datetime import datetime
    from llm_extraction import generate_cloze_cards_llm
    from state_management import save_generated_questions

    col1, col2 = st.columns([4, 1])
    with col1:
        st.header("Step 7: Generate Questions")
    with col2:
        if st.button("Reset To This Step", key="clear_generate", type="secondary",
                     help="Clear generated cloze cards"):
            clear_step_data("generate")
            st.success("Generated cards cleared")
            st.rerun()

    st.markdown("""
    Generate cloze deletion flashcards from question explanations.
    Each explanation is analyzed to extract key learning points that become additional Anki cards.
    """)

    # Check prerequisites - need either questions_merged or questions
    questions_source = st.session_state.questions_merged or st.session_state.questions
    if not questions_source:
        st.warning("Please format questions first (Step 4)")
        return

    client = get_anthropic_client()
    if not client:
        st.error("ANTHROPIC_API_KEY not set. Please set your API key in environment variables.")
        return

    # Count questions with explanations (skip context-only)
    all_questions = []
    for ch_key in sort_chapter_keys(questions_source.keys()):
        ch_questions = questions_source[ch_key]
        ch_num = int(ch_key[2:]) if ch_key.startswith("ch") else 0
        for q in ch_questions:
            explanation = q.get("explanation", "")
            if explanation and len(explanation) >= 50 and not q.get("is_context_only"):
                all_questions.append((ch_key, ch_num, q))

    total_with_explanations = len(all_questions)
    generated_cards = st.session_state.generated_questions.get("generated_cards", {})
    total_generated = sum(len(cards) for cards in generated_cards.values())

    # Stats row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Questions with Explanations", total_with_explanations)
    with col2:
        st.metric("Generated Cards", total_generated)
    with col3:
        avg = total_generated / total_with_explanations if total_with_explanations > 0 else 0
        st.metric("Avg Cards/Question", f"{avg:.1f}")

    st.markdown("---")

    # Chapter selector
    if not st.session_state.chapters:
        st.warning("No chapters available. Please extract chapters first (Step 2)")
        return

    chapter_options = [f"Ch{ch['chapter_number']}: {ch['title'][:40]}..."
                      for ch in st.session_state.chapters]
    selected_ch_idx = st.selectbox("Select chapter:",
                                    range(len(chapter_options)),
                                    format_func=lambda x: chapter_options[x],
                                    key="generate_ch_selector")

    selected_ch = st.session_state.chapters[selected_ch_idx]
    selected_ch_num = selected_ch["chapter_number"]
    selected_ch_key = f"ch{selected_ch_num}"

    # Count questions with explanations for selected chapter
    ch_questions_with_explanations = [
        (ch_key, ch_num, q) for ch_key, ch_num, q in all_questions
        if ch_key == selected_ch_key
    ]
    ch_question_count = len(ch_questions_with_explanations)

    # Generation controls
    model_col, workers_col, btn_col1, btn_col2 = st.columns([2, 1.5, 2, 2])

    with model_col:
        model_options = get_model_options()
        current_idx = model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
        selected_model = st.selectbox("Model:", model_options, index=current_idx, key="generate_model")
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            save_settings()

    with workers_col:
        gen_workers = st.number_input(
            "Parallel workers:",
            min_value=1, max_value=50, value=20,
            key="generate_workers",
            help="Number of questions to process in parallel"
        )

    with btn_col1:
        generate_chapter = st.button(
            f"Generate Chapter {selected_ch_num} ({ch_question_count})",
            type="primary",
            key="gen_chapter_btn",
            disabled=ch_question_count == 0
        )

    with btn_col2:
        generate_all = st.button("Generate ALL Cards", type="secondary", key="gen_all_btn")

    # Single chapter generation logic
    if generate_chapter:
        if ch_question_count == 0:
            st.warning(f"No questions with explanations in Chapter {selected_ch_num}")
        else:
            model_id = get_selected_model_id()
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Initialize generated_questions structure
            if not st.session_state.generated_questions.get("metadata"):
                st.session_state.generated_questions["metadata"] = {
                    "created_at": datetime.now().isoformat(),
                    "model_used": model_id,
                    "total_generated": 0,
                    "source_questions_processed": 0
                }
            if "generated_cards" not in st.session_state.generated_questions:
                st.session_state.generated_questions["generated_cards"] = {}

            # Clear existing cards for this chapter
            if selected_ch_key in st.session_state.generated_questions["generated_cards"]:
                st.session_state.generated_questions["generated_cards"][selected_ch_key] = []

            completed = 0
            total_cards_generated = 0

            def process_question(item):
                ch_key, ch_num, q = item
                cards = generate_cloze_cards_llm(client, q, ch_num, model_id)
                return ch_key, q, cards

            with ThreadPoolExecutor(max_workers=gen_workers) as executor:
                futures = {executor.submit(process_question, item): item for item in ch_questions_with_explanations}

                for future in as_completed(futures):
                    ch_key, q, cards = future.result()
                    completed += 1

                    if cards:
                        # Initialize chapter list if needed
                        if ch_key not in st.session_state.generated_questions["generated_cards"]:
                            st.session_state.generated_questions["generated_cards"][ch_key] = []

                        # Build card objects
                        for i, card in enumerate(cards, 1):
                            full_card = {
                                "generated_id": f"{q['full_id']}_gen_{i}",
                                "source_question_id": q["full_id"],
                                "cloze_text": card.get("cloze_text", ""),
                                "learning_point": card.get("learning_point", ""),
                                "confidence": card.get("confidence", "medium"),
                                "category": card.get("category", "")
                            }
                            st.session_state.generated_questions["generated_cards"][ch_key].append(full_card)
                            total_cards_generated += 1

                    # Update progress
                    progress = completed / ch_question_count
                    progress_bar.progress(progress)
                    status_text.text(f"Chapter {selected_ch_num}: {completed}/{ch_question_count} questions | {total_cards_generated} cards generated")

                    # Save periodically (every 10 questions)
                    if completed % 10 == 0:
                        st.session_state.generated_questions["metadata"]["total_generated"] = sum(
                            len(cards) for cards in st.session_state.generated_questions["generated_cards"].values()
                        )
                        save_generated_questions()

            # Final save
            st.session_state.generated_questions["metadata"]["total_generated"] = sum(
                len(cards) for cards in st.session_state.generated_questions["generated_cards"].values()
            )
            save_generated_questions()

            progress_bar.progress(1.0)
            status_text.text(f"Complete: {completed} questions processed, {total_cards_generated} cards generated")
            st.success(f"Generated {total_cards_generated} cloze cards from {completed} questions in Chapter {selected_ch_num}!")
            st.rerun()

    # All chapters generation logic
    if generate_all:
        if total_with_explanations == 0:
            st.warning("No questions with explanations found to process")
        else:
            model_id = get_selected_model_id()
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Initialize generated_questions structure
            if not st.session_state.generated_questions.get("metadata"):
                st.session_state.generated_questions["metadata"] = {
                    "created_at": datetime.now().isoformat(),
                    "model_used": model_id,
                    "total_generated": 0,
                    "source_questions_processed": 0
                }
            if "generated_cards" not in st.session_state.generated_questions:
                st.session_state.generated_questions["generated_cards"] = {}

            completed = 0
            total_cards_generated = 0

            def process_question(item):
                ch_key, ch_num, q = item
                cards = generate_cloze_cards_llm(client, q, ch_num, model_id)
                return ch_key, q, cards

            with ThreadPoolExecutor(max_workers=gen_workers) as executor:
                futures = {executor.submit(process_question, item): item for item in all_questions}

                for future in as_completed(futures):
                    ch_key, q, cards = future.result()
                    completed += 1

                    if cards:
                        # Initialize chapter list if needed
                        if ch_key not in st.session_state.generated_questions["generated_cards"]:
                            st.session_state.generated_questions["generated_cards"][ch_key] = []

                        # Build card objects (source data is looked up from questions_merged)
                        for i, card in enumerate(cards, 1):
                            full_card = {
                                "generated_id": f"{q['full_id']}_gen_{i}",
                                "source_question_id": q["full_id"],
                                "cloze_text": card.get("cloze_text", ""),
                                "learning_point": card.get("learning_point", ""),
                                "confidence": card.get("confidence", "medium"),
                                "category": card.get("category", "")
                            }
                            st.session_state.generated_questions["generated_cards"][ch_key].append(full_card)
                            total_cards_generated += 1

                    # Update progress
                    progress = completed / total_with_explanations
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: {completed}/{total_with_explanations} questions | {total_cards_generated} cards generated")

                    # Save periodically (every 10 questions)
                    if completed % 10 == 0:
                        st.session_state.generated_questions["metadata"]["total_generated"] = total_cards_generated
                        st.session_state.generated_questions["metadata"]["source_questions_processed"] = completed
                        save_generated_questions()

            # Final save
            st.session_state.generated_questions["metadata"]["total_generated"] = total_cards_generated
            st.session_state.generated_questions["metadata"]["source_questions_processed"] = completed
            save_generated_questions()

            progress_bar.progress(1.0)
            status_text.text(f"Complete: {completed} questions processed, {total_cards_generated} cards generated")
            st.success(f"Generated {total_cards_generated} cloze cards from {completed} questions!")
            st.rerun()

    # Preview section
    generated_cards = st.session_state.generated_questions.get("generated_cards", {})
    if generated_cards:
        st.markdown("---")
        st.subheader("Generated Cards Preview")

        # Filters
        col1, col2 = st.columns(2)
        with col1:
            ch_options = ["All Chapters"] + sort_chapter_keys(generated_cards.keys())
            ch_filter = st.selectbox("Filter by chapter:", ch_options, key="gen_ch_filter")
        with col2:
            # Get unique source question IDs
            source_ids = set()
            for ch_cards in generated_cards.values():
                for card in ch_cards:
                    source_ids.add(card["source_question_id"])
            source_options = ["All Questions"] + sorted(source_ids, key=question_sort_key)
            source_filter = st.selectbox("Filter by source:", source_options, key="gen_source_filter")

        # Collect filtered cards
        filtered_cards = []
        for ch_key in sort_chapter_keys(generated_cards.keys()):
            if ch_filter != "All Chapters" and ch_key != ch_filter:
                continue
            for card in generated_cards[ch_key]:
                if source_filter != "All Questions" and card["source_question_id"] != source_filter:
                    continue
                filtered_cards.append((ch_key, card))

        if not filtered_cards:
            st.caption("No cards match filters")
        else:
            st.caption(f"Showing {len(filtered_cards)} cards")

            # Card navigation
            if "gen_preview_idx" not in st.session_state:
                st.session_state.gen_preview_idx = 0
            if st.session_state.gen_preview_idx >= len(filtered_cards):
                st.session_state.gen_preview_idx = 0

            current_idx = st.session_state.gen_preview_idx
            ch_key, card = filtered_cards[current_idx]

            # Navigation
            nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
            with nav_col1:
                if st.button("< Previous", disabled=(current_idx == 0), key="gen_prev"):
                    st.session_state.gen_preview_idx -= 1
                    st.rerun()
            with nav_col2:
                st.markdown(f"<div style='text-align: center;'>Card {current_idx + 1} of {len(filtered_cards)}</div>",
                           unsafe_allow_html=True)
            with nav_col3:
                if st.button("Next >", disabled=(current_idx >= len(filtered_cards) - 1), key="gen_next"):
                    st.session_state.gen_preview_idx += 1
                    st.rerun()

            # Look up source question from questions_merged
            source_q_id = card["source_question_id"]
            source_q = None
            # Parse chapter from question ID (e.g., "ch1_12b" -> "ch1")
            if "_" in source_q_id:
                source_ch_key = source_q_id.split("_")[0]
                ch_questions = questions_source.get(source_ch_key, [])
                for q in ch_questions:
                    if q.get("full_id") == source_q_id:
                        source_q = q
                        break

            # Side-by-side display
            left_col, right_col = st.columns(2)

            with left_col:
                st.markdown("#### Source")
                st.markdown(f"**Question ID:** `{source_q_id}`")

                if source_q:
                    # Show full explanation text as the source reference
                    explanation = source_q.get("explanation", "")
                    if explanation:
                        st.text_area("Explanation (source):", explanation, height=300, disabled=True, key=f"src_exp_{current_idx}")
                    else:
                        st.caption("No explanation available")
                else:
                    st.warning("Source question not found")

            with right_col:
                st.markdown("#### Generated Cloze Card")

                # Display cloze with visual highlighting
                cloze_text = card.get("cloze_text", "")
                import re
                # Convert {{c1::text::hint}} to **[text::hint]** for visual display (keep hint)
                display_text = re.sub(r'\{\{c\d+::([^}]+)\}\}', r'**[\1]**', cloze_text)
                # Convert <b>...</b> to **...** for Markdown bold rendering
                display_text = re.sub(r'<b>([^<]+)</b>', r'**\1**', display_text)
                st.markdown(f"**Cloze:**")
                st.markdown(display_text)

                st.markdown(f"**Learning Point:** {card.get('learning_point', 'N/A')}")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Confidence:** {card.get('confidence', 'N/A')}")
                with col_b:
                    st.markdown(f"**Category:** {card.get('category', 'N/A')}")

                # Show all cards from same source
                same_source_cards = [c for _, c in filtered_cards if c["source_question_id"] == card["source_question_id"]]
                if len(same_source_cards) > 1:
                    with st.expander(f"All {len(same_source_cards)} cards from this source"):
                        for i, sc in enumerate(same_source_cards, 1):
                            # Convert cloze syntax and <b> tags for display (keep hint)
                            display = re.sub(r'\{\{c\d+::([^}]+)\}\}', r'**[\1]**', sc.get("cloze_text", ""))
                            display = re.sub(r'<b>([^<]+)</b>', r'**\1**', display)
                            st.markdown(f"{i}. {display}")


def render_export_step():
    """Render export step."""
    col1, col2 = st.columns([4, 1])
    with col1:
        st.header("Step 8: Export to Anki")
    with col2:
        if st.button("Clear Exports", key="clear_export", type="secondary", help="Delete exported .apkg files"):
            clear_step_data("export")
            st.success("Export files cleared")
            st.rerun()

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

    # Count generated cards
    generated_cards = st.session_state.generated_questions.get("generated_cards", {})
    total_generated = sum(len(cards) for cards in generated_cards.values())

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Questions", total)
    with col2:
        st.metric("Exportable", exportable, help="Excludes context-only entries")
    with col3:
        st.metric("Approved (QC'd)", approved)
    with col4:
        st.metric("Generated Cloze", total_generated, help="Cloze cards from Step 7")

    st.markdown("---")

    # Book name setting
    default_name = Path(st.session_state.current_pdf).stem if st.session_state.current_pdf else "Textbook"
    default_name = default_name.replace('_', ' ').replace('-', ' ')

    book_name = st.text_input("Book Name (used as deck name)",
                              value=default_name,
                              help="This will be the parent deck name in Anki")

    # Export options
    col1, col2, col3 = st.columns(3)
    with col1:
        only_approved = st.checkbox("Only export approved questions",
                                   value=False,
                                   help="Only include questions marked as approved in QC")
    with col2:
        include_images = st.checkbox("Include images",
                                    value=True,
                                    help="Embed assigned images in cards")
    with col3:
        include_generated = st.checkbox("Include generated cloze cards",
                                        value=total_generated > 0,
                                        disabled=total_generated == 0,
                                        help="Add cloze cards from Step 7 as a sub-deck")

    st.markdown("---")

    # Preview deck structure
    with st.expander("Preview Deck Structure"):
        st.markdown(f"**{book_name}**")

        # Count generated cards per chapter (keys are "ch4", "ch5", etc.)
        generated_by_chapter = {}
        if include_generated:
            for ch_key, cards in generated_cards.items():
                ch_num = ch_key.replace('ch', '')
                generated_by_chapter[ch_num] = len(cards)

        for ch in (st.session_state.chapters or []):
            ch_num = ch['chapter_number']
            ch_title = ch.get('title', f'Chapter {ch_num}')
            ch_key = f"ch{ch_num}"
            extracted_count = len([q for q in st.session_state.questions.get(ch_key, [])
                                   if not q.get('is_context_only')])
            gen_count = generated_by_chapter.get(str(ch_num), 0)

            if extracted_count > 0 or (include_generated and gen_count > 0):
                st.markdown(f"  - **{ch_num}. {ch_title}**")
                if extracted_count > 0:
                    st.markdown(f"    - Extracted ({extracted_count} cards)")
                if include_generated and gen_count > 0:
                    st.markdown(f"    - Generated ({gen_count} cards)")

    # Export button
    if st.button("Export to Anki Deck", type="primary"):
        if not book_name.strip():
            st.error("Please enter a book name")
            return

        with st.spinner("Generating Anki deck..."):
            try:
                # Use merged questions/assignments if available (includes context in text)
                questions_to_export = st.session_state.questions_merged if st.session_state.questions_merged else st.session_state.questions
                assignments_to_export = st.session_state.image_assignments_merged if st.session_state.image_assignments_merged else st.session_state.image_assignments

                output_path = generate_anki_deck(
                    book_name=book_name.strip(),
                    questions=questions_to_export,
                    chapters=st.session_state.chapters or [],
                    image_assignments=assignments_to_export,
                    images=st.session_state.images,
                    include_images=include_images,
                    only_approved=only_approved,
                    qc_progress=st.session_state.qc_progress,
                    generated_questions=st.session_state.generated_questions,
                    include_generated=include_generated
                )

                st.success(f"Deck exported successfully!")
                play_completion_sound()
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
