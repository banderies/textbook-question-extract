"""
UI Components Module

Contains all Streamlit UI rendering functions.
"""

import os
import re
import copy
from pathlib import Path
from datetime import datetime
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st

from state_management import (
    SOURCE_DIR, get_pdf_slug, get_output_dir, get_images_dir,
    get_available_textbooks, clear_session_data, load_saved_data,
    load_settings, load_qc_progress, save_settings, save_chapters,
    save_questions, save_images, save_pages, save_image_assignments,
    save_questions_merged, save_image_assignments_merged, save_qc_progress
)
from pdf_extraction import (
    extract_text_from_pdf, extract_images_from_pdf, assign_chapters_to_images,
    extract_chapter_text, render_pdf_page
)
from llm_extraction import (
    get_anthropic_client, get_model_options, get_model_id,
    identify_chapters_llm, extract_qa_pairs_llm, process_chapter_extraction,
    match_images_to_questions_llm, associate_context_llm, add_page_numbers_to_questions
)


# =============================================================================
# Helper Functions
# =============================================================================

def get_selected_model_id() -> str:
    """Get the currently selected Claude model ID."""
    return get_model_id(st.session_state.selected_model)


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
        ("context", "4. Associate Context"),
        ("qc", "5. QC Questions"),
        ("export", "6. Export")
    ]

    for step_id, step_name in steps:
        if st.sidebar.button(step_name, key=f"nav_{step_id}"):
            st.session_state.current_step = step_id
            save_settings()

    st.sidebar.markdown("---")

    # Status summary - Order: Chapters, Images, Questions, Context, QC
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

    q_count = sum(len(qs) for qs in st.session_state.questions.values())
    if q_count > 0:
        st.sidebar.success(f"Questions: {q_count}")
    else:
        st.sidebar.info("Questions: Not extracted")

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

                with st.spinner("Extracting text from PDF..."):
                    st.session_state.pages = extract_text_from_pdf(pdf_path)
                    st.session_state.pdf_path = pdf_path
                    save_pages()

                with st.spinner("Extracting images from PDF..."):
                    st.session_state.images = extract_images_from_pdf(pdf_path, get_images_dir())
                    save_images()

                st.success(f"Loaded {len(st.session_state.pages)} pages, {len(st.session_state.images)} images")
                st.rerun()

        with col2:
            if has_existing_data:
                if st.button("Load Existing Progress"):
                    st.session_state.current_pdf = selected_pdf
                    clear_session_data()
                    load_saved_data()
                    load_settings()
                    st.session_state.qc_progress = load_qc_progress()
                    st.success("Loaded previous session data")
                    st.rerun()

        if st.session_state.pages:
            st.success(f"PDF loaded: {len(st.session_state.pages)} pages, {len(st.session_state.images)} images")
            st.markdown("**Next:** Go to 'Extract Chapters' to identify chapter boundaries.")


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
# Step 3: Extract Questions
# =============================================================================

def render_questions_step():
    """Render question extraction step."""
    st.header("Step 3: Extract Questions")

    if not st.session_state.chapters:
        st.warning("Please extract chapters first (Step 2)")
        return

    client = get_anthropic_client()
    if not client:
        st.error("ANTHROPIC_API_KEY not set. Please configure your .env file.")
        return

    chapter_options = [f"Ch{ch['chapter_number']}: {ch['title'][:40]}..."
                      for ch in st.session_state.chapters]
    selected_ch_idx = st.selectbox("Select chapter:",
                                    range(len(chapter_options)),
                                    format_func=lambda x: chapter_options[x])

    if selected_ch_idx is not None:
        ch = st.session_state.chapters[selected_ch_idx]
        ch_num = ch["chapter_number"]
        ch_key = f"ch{ch_num}"

        model_col, btn_col1, btn_col2 = st.columns([2, 2, 2])

        with model_col:
            model_options = get_model_options()
            current_idx = model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
            selected_model = st.selectbox("Model:", model_options, index=current_idx, key="questions_model")
            if selected_model != st.session_state.selected_model:
                st.session_state.selected_model = selected_model
                save_settings()

        with btn_col1:
            if st.button(f"Extract Chapter {ch_num}", type="primary"):
                with st.spinner(f"Using {st.session_state.selected_model} to extract Q&A from Chapter {ch_num}..."):
                    ch_text = st.session_state.chapter_texts.get(ch_key, "")
                    result = extract_qa_pairs_llm(client, ch_num, ch_text, get_selected_model_id())

                    questions = []
                    for q in result.get("questions", []):
                        questions.append({
                            "full_id": f"ch{ch_num}_{q['id']}",
                            "local_id": q["id"],
                            "text": q.get("text", ""),
                            "choices": q.get("choices", {}),
                            "has_image": q.get("has_image", False),
                            "image_group": q.get("image_group"),
                            "correct_answer": q.get("correct_answer", ""),
                            "explanation": q.get("explanation", "")
                        })

                    st.session_state.questions[ch_key] = questions
                    save_questions()

                    if st.session_state.images and st.session_state.chapters:
                        with st.spinner("Matching images to questions..."):
                            st.session_state.image_assignments = match_images_to_questions_llm(
                                client,
                                st.session_state.images,
                                st.session_state.chapters,
                                st.session_state.questions,
                                get_selected_model_id()
                            )
                            save_image_assignments()

                st.success(f"Extracted {len(questions)} questions from Chapter {ch_num}")
                st.rerun()

        with btn_col2:
            if st.button("Extract ALL Chapters"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                model_id = get_selected_model_id()
                total_chapters = len(st.session_state.chapters)

                chapter_tasks = []
                for ch in st.session_state.chapters:
                    ch_num = ch["chapter_number"]
                    ch_key = f"ch{ch_num}"
                    ch_text = st.session_state.chapter_texts.get(ch_key, "")
                    chapter_tasks.append((ch_num, ch_key, ch_text))

                status_text.text(f"Processing {total_chapters} chapters in parallel...")

                completed = 0
                with ThreadPoolExecutor(max_workers=min(total_chapters, 5)) as executor:
                    future_to_chapter = {
                        executor.submit(
                            process_chapter_extraction,
                            client, ch_num, ch_key, ch_text, model_id
                        ): ch_key
                        for ch_num, ch_key, ch_text in chapter_tasks
                    }

                    for future in as_completed(future_to_chapter):
                        ch_key = future_to_chapter[future]
                        try:
                            result_key, questions = future.result()
                            st.session_state.questions[result_key] = questions
                            completed += 1
                            progress_bar.progress(completed / total_chapters)
                            status_text.text(f"Completed {completed}/{total_chapters} chapters...")
                        except Exception as e:
                            st.warning(f"Error processing {ch_key}: {e}")
                            completed += 1
                            progress_bar.progress(completed / total_chapters)

                save_questions()

                if st.session_state.images:
                    status_text.text("Matching images to questions (using Claude)...")
                    st.session_state.image_assignments = match_images_to_questions_llm(
                        client,
                        st.session_state.images,
                        st.session_state.chapters,
                        st.session_state.questions,
                        get_selected_model_id()
                    )
                    save_image_assignments()

                status_text.text("Done!")
                st.success(f"Extracted questions from all {total_chapters} chapters")
                st.rerun()

        if st.session_state.questions:
            st.markdown("---")
            st.info("**Next step:** Go to **Step 4: Associate Context** to link context and images from parent questions to sub-questions.")

        if ch_key in st.session_state.questions:
            st.markdown("---")
            st.subheader(f"Questions in Chapter {ch_num}")

            questions = st.session_state.questions[ch_key]

            total = len(questions)
            context_only_count = sum(1 for q in questions if q.get("is_context_only"))
            merged_count = sum(1 for q in questions if q.get("context_merged"))
            actual_questions = total - context_only_count

            st.info(f"Total: {actual_questions} questions" +
                   (f" + {context_only_count} context-only" if context_only_count > 0 else "") +
                   (f" ({merged_count} with merged context)" if merged_count > 0 else ""))

            for q in questions:
                q_images = [img for img in st.session_state.images
                           if st.session_state.image_assignments.get(img["filename"]) == q["full_id"]]

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


# =============================================================================
# Step 4: Associate Context
# =============================================================================

def render_context_step():
    """Render context association step."""
    st.header("Step 4: Associate Context")

    if not st.session_state.questions:
        st.warning("Please extract questions first (Step 3)")
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

    col1, col2 = st.columns([2, 3])

    with col1:
        model_options = get_model_options()
        current_idx = model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
        selected_model = st.selectbox("Model:", model_options, index=current_idx, key="context_model")
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            save_settings()

    with col2:
        st.write("")
        if st.button("Associate Context", type="primary"):
            client = get_anthropic_client()
            if not client:
                st.error("ANTHROPIC_API_KEY not set. Please set the environment variable.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                total_chapters = len(st.session_state.questions)
                status_text.text(f"Processing {total_chapters} chapters...")

                questions_copy = copy.deepcopy(st.session_state.questions)
                assignments_copy = copy.deepcopy(st.session_state.image_assignments)

                model_id = get_model_id(st.session_state.selected_model)
                progress_bar.progress(0.1)

                updated_questions, updated_assignments, stats = associate_context_llm(
                    client,
                    questions_copy,
                    assignments_copy,
                    model_id=model_id
                )

                progress_bar.progress(0.8)
                status_text.text("Saving merged data...")

                st.session_state.questions_merged = updated_questions
                st.session_state.image_assignments_merged = updated_assignments
                save_questions_merged()
                save_image_assignments_merged()

                progress_bar.progress(1.0)
                status_text.text("Done!")

                st.success(
                    f"Context association complete!\n\n"
                    f"- Context questions found: {stats['context_questions_found']}\n"
                    f"- Sub-questions updated: {stats['sub_questions_updated']}\n"
                    f"- Images copied: {stats['images_copied']}"
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
# Step 5: QC Questions
# =============================================================================

def render_qc_step():
    """Render QC review step."""
    st.header("Step 5: QC Questions")

    if not st.session_state.questions:
        st.warning("Please extract questions first (Step 3)")
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

    # Page number detection - always show the button
    with st.expander("PDF Page Detection", expanded=False):
        st.caption("Detect which PDF pages contain each question and answer for the PDF preview feature.")
        if st.button("Detect All Page Numbers", type="primary"):
            with st.spinner("Detecting page numbers for all questions..."):
                add_page_numbers_to_questions(
                    st.session_state.questions,
                    st.session_state.pages,
                    st.session_state.chapters
                )
                save_questions()
                st.success("Page numbers detected!")
                st.rerun()

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
            col1, col2, col3, col4, col5, col6 = st.columns(6)

            with col1:
                st.button("Previous", disabled=(st.session_state.qc_selected_idx <= 0), on_click=go_previous)

            with col2:
                st.button("Approve & Next", type="primary", disabled=is_approved, on_click=approve_and_next)

            with col3:
                st.button("Flag Issue", disabled=is_flagged, on_click=flag_issue)

            with col4:
                if is_approved or is_flagged:
                    st.button("Unapprove", on_click=unapprove)

            with col5:
                st.button("Next", disabled=(st.session_state.qc_selected_idx >= len(filtered_questions) - 1), on_click=go_next)

            with col6:
                if is_approved:
                    st.success("Approved")
                elif is_flagged:
                    st.warning("Flagged")

            st.markdown("---")

            left_col, right_col = st.columns([1, 1])

            with left_col:
                st.subheader(f"Question {q['local_id']}")

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
# Step 6: Export
# =============================================================================

def render_export_step():
    """Render export step."""
    st.header("Step 6: Export to Anki")

    if not st.session_state.questions:
        st.warning("Please extract questions first (Step 3)")
        return

    total = sum(len(qs) for qs in st.session_state.questions.values())
    reviewed = st.session_state.qc_progress.get("reviewed", {})
    approved = sum(1 for r in reviewed.values() if r.get("status") == "approved")

    st.info(f"Total questions: {total}")
    st.info(f"Approved (QC'd): {approved}")

    st.markdown("---")

    export_option = st.radio("Export:", [
        "All questions",
        "Only approved (QC'd) questions"
    ])

    if st.button("Export to Anki Deck", type="primary"):
        st.warning("Anki export functionality coming soon!")
        st.markdown("""
        **Planned features:**
        - Generate .apkg file for direct Anki import
        - Include question, choices, correct answer, and explanation
        - Tag cards by chapter
        - Optional: include associated images
        """)
