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
import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Optional
import fitz  # PyMuPDF

# Page config
st.set_page_config(
    page_title="Textbook Q&A Extractor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
SOURCE_DIR = "source"
OUTPUT_DIR = "output"
IMAGES_DIR = "images"

# Files
CHAPTERS_FILE = f"{OUTPUT_DIR}/chapters.json"
CHAPTER_TEXT_FILE = f"{OUTPUT_DIR}/chapter_text.json"
QUESTIONS_FILE = f"{OUTPUT_DIR}/questions_by_chapter.json"
IMAGES_FILE = f"{OUTPUT_DIR}/images.json"
IMAGE_ASSIGNMENTS_FILE = f"{OUTPUT_DIR}/image_assignments.json"
QC_PROGRESS_FILE = f"{OUTPUT_DIR}/qc_progress.json"


# =============================================================================
# PDF Extraction Functions (from llm_pipeline.py)
# =============================================================================

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract raw text from PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        pages.append({"page": page_num + 1, "text": text})
    doc.close()
    return pages


def extract_images_from_pdf(pdf_path: str, output_dir: str = IMAGES_DIR) -> list[dict]:
    """
    Extract images from PDF with page numbers and positions.
    Returns list of image metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]

            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Get image position on page
                img_rects = page.get_image_rects(xref)
                if img_rects:
                    rect = img_rects[0]
                    y_position = rect.y0
                    x_position = rect.x0
                else:
                    y_position = 0
                    x_position = 0

                # Create filename with page and position info
                filename = f"p{page_num + 1:03d}_y{int(y_position):04d}_x{int(x_position):04d}_{img_idx}.{image_ext}"
                filepath = os.path.join(output_dir, filename)

                # Save image
                with open(filepath, "wb") as f:
                    f.write(image_bytes)

                images.append({
                    "filename": filename,
                    "filepath": filepath,
                    "page": page_num + 1,
                    "y_position": y_position,
                    "x_position": x_position,
                    "width": rect.width if img_rects else 0,
                    "height": rect.height if img_rects else 0
                })

            except Exception as e:
                # Skip problematic images
                continue

    doc.close()

    # Sort by page and y-position
    images.sort(key=lambda x: (x["page"], x["y_position"]))

    return images


def match_images_to_questions(images: list[dict], chapters: list[dict], questions: dict) -> dict:
    """
    Auto-match images to questions based on page proximity.
    Returns dict mapping image filename to question full_id.
    """
    assignments = {}

    # Build a list of questions with their chapter's page range
    question_pages = []
    for i, ch in enumerate(chapters):
        ch_num = ch["chapter_number"]
        ch_key = f"ch{ch_num}"
        start_page = ch["start_page"]
        end_page = chapters[i + 1]["start_page"] if i + 1 < len(chapters) else 9999

        if ch_key in questions:
            for q in questions[ch_key]:
                question_pages.append({
                    "full_id": q["full_id"],
                    "chapter": ch_num,
                    "start_page": start_page,
                    "end_page": end_page,
                    "has_image": q.get("has_image", False)
                })

    # For each image, find the best matching question
    for img in images:
        img_page = img["page"]

        # Find questions in chapters that contain this page
        candidates = [q for q in question_pages
                      if q["start_page"] <= img_page < q["end_page"]]

        if candidates:
            # Prefer questions marked as having images
            image_questions = [q for q in candidates if q["has_image"]]
            if image_questions:
                # Assign to first unassigned question with has_image
                for q in image_questions:
                    if q["full_id"] not in assignments.values():
                        assignments[img["filename"]] = q["full_id"]
                        break
                else:
                    # All image questions assigned, use first candidate
                    assignments[img["filename"]] = candidates[0]["full_id"]
            else:
                # No questions marked with images, assign to first in chapter
                assignments[img["filename"]] = candidates[0]["full_id"]

    return assignments


def create_page_index(pages: list[dict]) -> str:
    """Create condensed index of pages for chapter identification."""
    index_parts = []
    for p in pages:
        preview = p["text"][:300].replace("\n", " ").strip()
        index_parts.append(f"[PAGE {p['page']}] {preview}")
    return "\n".join(index_parts)


def get_anthropic_client():
    """Get Anthropic client, loading API key from .env if needed."""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    import anthropic
    return anthropic.Anthropic()


def identify_chapters_llm(client, pages: list[dict]) -> list[dict]:
    """Use Claude to identify chapter boundaries."""
    page_index = create_page_index(pages)

    response = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""Analyze this textbook page index and identify all chapters that contain QUESTIONS sections.

This is a medical textbook where each chapter has:
- A chapter header (e.g., "Chapter 1: Title" or "1 Title")
- A QUESTIONS section with numbered questions
- An ANSWERS section with explanations

For each chapter with questions, provide:
- chapter_number: The chapter number (integer)
- title: The chapter title
- start_page: The page number where the chapter starts (from the [PAGE X] markers)
- has_questions: true (only include chapters that have questions)

Look for patterns like "QUESTIONS" or numbered questions (1., 2., etc.) to identify which chapters have Q&A content.

Return ONLY a JSON array, no other text:
[
  {{"chapter_number": 1, "title": "Chapter Title", "start_page": 14, "has_questions": true}},
  ...
]

PAGE INDEX:
{page_index}"""
        }]
    )

    response_text = response.content[0].text
    if "```" in response_text:
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if match:
            response_text = match.group(1)

    return json.loads(response_text)


def extract_chapter_text(pages: list[dict], start_page: int, end_page: Optional[int]) -> str:
    """Extract text for a specific chapter given page range."""
    chapter_pages = []
    for p in pages:
        if p["page"] >= start_page:
            if end_page is None or p["page"] < end_page:
                chapter_pages.append(f"[PAGE {p['page']}]\n{p['text']}")
    return "\n\n".join(chapter_pages)


def extract_qa_pairs_llm(client, chapter_num: int, chapter_text: str) -> dict:
    """Use Claude to extract Q&A pairs from a single chapter."""
    response = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=16000,
        messages=[{
            "role": "user",
            "content": f"""You are analyzing Chapter {chapter_num} of a medical textbook to extract all questions and their corresponding answers.

TASK:
1. Find all questions in the QUESTIONS section
2. Find all answers in the ANSWERS section
3. Match each question to its answer
4. Identify the correct answer choice (A, B, C, D, or E)

IMPORTANT:
- Questions may have sub-parts like 2a, 2b, 2c - treat each as a separate question
- Question IDs should match exactly as they appear in the ANSWERS section (e.g., "1", "2a", "2b", "3")
- Some questions reference figures/images - note when a question mentions "image below" or similar
- Extract the full question text and all answer choices

Return ONLY a JSON object in this exact format:
{{
  "chapter": {chapter_num},
  "questions": [
    {{
      "id": "1",
      "text": "Full question text here",
      "choices": {{
        "A": "Choice A text",
        "B": "Choice B text",
        "C": "Choice C text",
        "D": "Choice D text"
      }},
      "has_image": true,
      "correct_answer": "B",
      "explanation": "Brief explanation from the answer section"
    }}
  ]
}}

CHAPTER TEXT:
{chapter_text}"""
        }]
    )

    response_text = response.content[0].text
    if "```" in response_text:
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if match:
            response_text = match.group(1)

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        return {"chapter": chapter_num, "questions": [], "error": str(e)}


# =============================================================================
# State Management
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "pages" not in st.session_state:
        st.session_state.pages = None
    if "chapters" not in st.session_state:
        st.session_state.chapters = None
    if "chapter_texts" not in st.session_state:
        st.session_state.chapter_texts = {}
    if "questions" not in st.session_state:
        st.session_state.questions = {}
    if "images" not in st.session_state:
        st.session_state.images = []
    if "image_assignments" not in st.session_state:
        st.session_state.image_assignments = {}
    if "qc_progress" not in st.session_state:
        st.session_state.qc_progress = load_qc_progress()
    if "current_step" not in st.session_state:
        st.session_state.current_step = "source"


def load_qc_progress() -> dict:
    """Load QC progress from file."""
    if os.path.exists(QC_PROGRESS_FILE):
        with open(QC_PROGRESS_FILE) as f:
            return json.load(f)
    return {"reviewed": {}, "corrections": {}, "metadata": {}}


def save_qc_progress():
    """Save QC progress to file."""
    st.session_state.qc_progress["metadata"]["last_saved"] = datetime.now().isoformat()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(QC_PROGRESS_FILE, "w") as f:
        json.dump(st.session_state.qc_progress, f, indent=2)


def save_chapters():
    """Save chapters to file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(CHAPTERS_FILE, "w") as f:
        json.dump(st.session_state.chapters, f, indent=2)
    with open(CHAPTER_TEXT_FILE, "w") as f:
        json.dump(st.session_state.chapter_texts, f, indent=2)


def save_questions():
    """Save questions to file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(QUESTIONS_FILE, "w") as f:
        json.dump(st.session_state.questions, f, indent=2)


def save_images():
    """Save image metadata to file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(IMAGES_FILE, "w") as f:
        json.dump(st.session_state.images, f, indent=2)


def save_image_assignments():
    """Save image-to-question assignments to file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(IMAGE_ASSIGNMENTS_FILE, "w") as f:
        json.dump(st.session_state.image_assignments, f, indent=2)


def load_saved_data():
    """Load previously saved data if available."""
    if os.path.exists(CHAPTERS_FILE):
        with open(CHAPTERS_FILE) as f:
            st.session_state.chapters = json.load(f)
    if os.path.exists(CHAPTER_TEXT_FILE):
        with open(CHAPTER_TEXT_FILE) as f:
            st.session_state.chapter_texts = json.load(f)
    if os.path.exists(QUESTIONS_FILE):
        with open(QUESTIONS_FILE) as f:
            st.session_state.questions = json.load(f)
    if os.path.exists(IMAGES_FILE):
        with open(IMAGES_FILE) as f:
            st.session_state.images = json.load(f)
    if os.path.exists(IMAGE_ASSIGNMENTS_FILE):
        with open(IMAGE_ASSIGNMENTS_FILE) as f:
            st.session_state.image_assignments = json.load(f)


# =============================================================================
# UI Components
# =============================================================================

def render_sidebar():
    """Render sidebar with navigation."""
    st.sidebar.title("📚 Textbook Q&A Extractor")
    st.sidebar.markdown("---")

    steps = [
        ("source", "1. Select Source"),
        ("chapters", "2. Extract Chapters"),
        ("questions", "3. Extract Questions"),
        ("qc", "4. QC Questions"),
        ("export", "5. Export")
    ]

    for step_id, step_name in steps:
        if st.sidebar.button(step_name, key=f"nav_{step_id}", use_container_width=True):
            st.session_state.current_step = step_id

    st.sidebar.markdown("---")

    # Status summary
    st.sidebar.subheader("Status")
    if st.session_state.chapters:
        st.sidebar.success(f"Chapters: {len(st.session_state.chapters)}")
    else:
        st.sidebar.info("Chapters: Not extracted")

    q_count = sum(len(qs) for qs in st.session_state.questions.values())
    if q_count > 0:
        st.sidebar.success(f"Questions: {q_count}")
    else:
        st.sidebar.info("Questions: Not extracted")

    img_count = len(st.session_state.images)
    if img_count > 0:
        assigned = len(st.session_state.image_assignments)
        st.sidebar.success(f"Images: {img_count} ({assigned} assigned)")
    else:
        st.sidebar.info("Images: Not extracted")

    reviewed = len(st.session_state.qc_progress.get("reviewed", {}))
    if reviewed > 0:
        st.sidebar.success(f"QC'd: {reviewed}/{q_count}")


def render_source_step():
    """Render source PDF selection step."""
    st.header("Step 1: Select Source PDF")

    # Find available PDFs
    pdf_files = list(Path(SOURCE_DIR).glob("*.pdf")) if os.path.exists(SOURCE_DIR) else []

    if not pdf_files:
        st.warning(f"No PDF files found in '{SOURCE_DIR}/' directory. Please add a PDF file.")
        return

    pdf_options = [f.name for f in pdf_files]
    selected_pdf = st.selectbox("Select PDF file:", pdf_options)

    if selected_pdf:
        pdf_path = f"{SOURCE_DIR}/{selected_pdf}"
        st.info(f"Selected: {pdf_path}")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Load PDF", type="primary"):
                with st.spinner("Extracting text from PDF..."):
                    st.session_state.pages = extract_text_from_pdf(pdf_path)
                    st.session_state.pdf_path = pdf_path

                with st.spinner("Extracting images from PDF..."):
                    st.session_state.images = extract_images_from_pdf(pdf_path)
                    save_images()

                st.success(f"Loaded {len(st.session_state.pages)} pages, {len(st.session_state.images)} images")
                st.rerun()

        with col2:
            if st.button("Load Previous Session"):
                load_saved_data()
                st.success("Loaded previous session data")
                st.rerun()

        if st.session_state.pages:
            st.success(f"PDF loaded: {len(st.session_state.pages)} pages, {len(st.session_state.images)} images")
            st.markdown("**Next:** Go to 'Extract Chapters' to identify chapter boundaries.")


def render_chapters_step():
    """Render chapter extraction step."""
    st.header("Step 2: Extract Chapters")

    if not st.session_state.pages:
        st.warning("Please load a PDF first (Step 1)")
        return

    client = get_anthropic_client()
    if not client:
        st.error("ANTHROPIC_API_KEY not set. Please configure your .env file.")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("Extract Chapters", type="primary"):
            with st.spinner("Using Claude Opus 4.5 to identify chapters..."):
                chapters = identify_chapters_llm(client, st.session_state.pages)
                st.session_state.chapters = chapters

                # Extract text for each chapter
                for i, ch in enumerate(chapters):
                    start_page = ch["start_page"]
                    end_page = chapters[i + 1]["start_page"] if i + 1 < len(chapters) else None
                    ch_key = f"ch{ch['chapter_number']}"
                    st.session_state.chapter_texts[ch_key] = extract_chapter_text(
                        st.session_state.pages, start_page, end_page
                    )

                save_chapters()
            st.success(f"Found {len(chapters)} chapters")
            st.rerun()

    if st.session_state.chapters:
        st.subheader("Extracted Chapters")

        # Chapter list
        for ch in st.session_state.chapters:
            st.markdown(f"**Chapter {ch['chapter_number']}:** {ch['title']} (page {ch['start_page']})")

        st.markdown("---")

        # Chapter preview
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

    # Chapter selection
    chapter_options = [f"Ch{ch['chapter_number']}: {ch['title'][:40]}..."
                      for ch in st.session_state.chapters]
    selected_ch_idx = st.selectbox("Select chapter:",
                                    range(len(chapter_options)),
                                    format_func=lambda x: chapter_options[x])

    if selected_ch_idx is not None:
        ch = st.session_state.chapters[selected_ch_idx]
        ch_num = ch["chapter_number"]
        ch_key = f"ch{ch_num}"

        col1, col2 = st.columns(2)

        with col1:
            if st.button(f"Extract Questions for Chapter {ch_num}", type="primary"):
                with st.spinner(f"Using Claude Opus 4.5 to extract Q&A from Chapter {ch_num}..."):
                    ch_text = st.session_state.chapter_texts.get(ch_key, "")
                    result = extract_qa_pairs_llm(client, ch_num, ch_text)

                    # Convert to GUI format
                    questions = []
                    for q in result.get("questions", []):
                        questions.append({
                            "full_id": f"ch{ch_num}_{q['id']}",
                            "local_id": q["id"],
                            "text": q.get("text", ""),
                            "choices": q.get("choices", {}),
                            "has_image": q.get("has_image", False),
                            "correct_answer": q.get("correct_answer", ""),
                            "explanation": q.get("explanation", "")
                        })

                    st.session_state.questions[ch_key] = questions
                    save_questions()

                st.success(f"Extracted {len(questions)} questions from Chapter {ch_num}")
                st.rerun()

        with col2:
            if st.button("Extract ALL Chapters"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, ch in enumerate(st.session_state.chapters):
                    ch_num = ch["chapter_number"]
                    ch_key = f"ch{ch_num}"
                    status_text.text(f"Processing Chapter {ch_num}...")

                    ch_text = st.session_state.chapter_texts.get(ch_key, "")
                    result = extract_qa_pairs_llm(client, ch_num, ch_text)

                    questions = []
                    for q in result.get("questions", []):
                        questions.append({
                            "full_id": f"ch{ch_num}_{q['id']}",
                            "local_id": q["id"],
                            "text": q.get("text", ""),
                            "choices": q.get("choices", {}),
                            "has_image": q.get("has_image", False),
                            "correct_answer": q.get("correct_answer", ""),
                            "explanation": q.get("explanation", "")
                        })

                    st.session_state.questions[ch_key] = questions
                    progress_bar.progress((i + 1) / len(st.session_state.chapters))

                save_questions()

                # Auto-match images to questions
                if st.session_state.images:
                    status_text.text("Matching images to questions...")
                    st.session_state.image_assignments = match_images_to_questions(
                        st.session_state.images,
                        st.session_state.chapters,
                        st.session_state.questions
                    )
                    save_image_assignments()

                status_text.text("Done!")
                st.success(f"Extracted questions from all {len(st.session_state.chapters)} chapters")
                st.rerun()

        # Preview extracted questions
        if ch_key in st.session_state.questions:
            st.markdown("---")
            st.subheader(f"Questions in Chapter {ch_num}")

            questions = st.session_state.questions[ch_key]
            st.info(f"Total: {len(questions)} questions")

            # Question list
            for q in questions:
                with st.expander(f"Q{q['local_id']}: {q['text'][:80]}..."):
                    st.markdown(f"**Question:** {q['text']}")
                    st.markdown("**Choices:**")
                    for letter, choice in q.get("choices", {}).items():
                        st.markdown(f"- {letter}: {choice}")
                    st.markdown(f"**Correct Answer:** {q.get('correct_answer', 'N/A')}")
                    st.markdown(f"**Explanation:** {q.get('explanation', 'N/A')}")
                    if q.get("has_image"):
                        st.caption("📷 This question has an associated image")


def question_sort_key(q_id: str) -> tuple:
    """Sort key for question IDs."""
    match = re.match(r'ch(\d+)_(\d+)([a-z]?)', q_id, re.IGNORECASE)
    if match:
        return (int(match.group(1)), int(match.group(2)), match.group(3) or "")
    return (999, 999, q_id)


def get_images_for_question(q_id: str) -> list[dict]:
    """Get all images assigned to a question."""
    images = []
    for img in st.session_state.images:
        if st.session_state.image_assignments.get(img["filename"]) == q_id:
            images.append(img)
    return images


def get_all_question_options() -> list[str]:
    """Get list of all question IDs for reassignment dropdown."""
    options = ["(none)"]
    for ch_key in sorted(st.session_state.questions.keys(), key=lambda x: int(x[2:]) if x[2:].isdigit() else 0):
        for q in st.session_state.questions[ch_key]:
            options.append(q["full_id"])
    return options


def render_qc_step():
    """Render QC review step."""
    st.header("Step 4: QC Questions")

    if not st.session_state.questions:
        st.warning("Please extract questions first (Step 3)")
        return

    # Flatten all questions for QC
    all_questions = []
    for ch_key, questions in st.session_state.questions.items():
        for q in questions:
            all_questions.append((ch_key, q))

    all_questions.sort(key=lambda x: question_sort_key(x[1]["full_id"]))

    if not all_questions:
        st.warning("No questions to review")
        return

    # Progress
    reviewed = st.session_state.qc_progress.get("reviewed", {})
    total = len(all_questions)
    reviewed_count = len(reviewed)

    st.progress(reviewed_count / total if total > 0 else 0)
    st.caption(f"Progress: {reviewed_count}/{total} questions reviewed")

    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        filter_option = st.radio("Show:", ["All", "Unreviewed only", "Reviewed only"], horizontal=True)
    with col2:
        chapter_filter = st.selectbox("Filter by chapter:",
                                       ["All chapters"] + list(st.session_state.questions.keys()))

    # Filter questions
    filtered_questions = []
    for ch_key, q in all_questions:
        q_id = q["full_id"]
        is_reviewed = q_id in reviewed

        # Apply filters
        if filter_option == "Unreviewed only" and is_reviewed:
            continue
        if filter_option == "Reviewed only" and not is_reviewed:
            continue
        if chapter_filter != "All chapters" and ch_key != chapter_filter:
            continue

        filtered_questions.append((ch_key, q))

    st.caption(f"Showing {len(filtered_questions)} questions")

    # Question selector
    if filtered_questions:
        question_options = [f"{q['full_id']}: {q['text'][:50]}..." for _, q in filtered_questions]
        selected_idx = st.selectbox("Select question:", range(len(question_options)),
                                    format_func=lambda x: question_options[x])

        if selected_idx is not None:
            ch_key, q = filtered_questions[selected_idx]
            q_id = q["full_id"]

            st.markdown("---")

            # Two column layout: question on left, images on right
            left_col, right_col = st.columns([1, 1])

            with left_col:
                # Display question
                st.subheader(f"Question {q['local_id']}")
                st.markdown(f"**{q['text']}**")

                st.markdown("**Choices:**")
                for letter, choice in q.get("choices", {}).items():
                    if letter == q.get("correct_answer"):
                        st.markdown(f"- **{letter}: {choice}** ✓")
                    else:
                        st.markdown(f"- {letter}: {choice}")

                st.markdown(f"**Explanation:** {q.get('explanation', 'N/A')}")

                # Show current status
                if q_id in reviewed:
                    status = reviewed[q_id].get("status", "unknown")
                    if status == "approved":
                        st.success(f"Status: Approved")
                    elif status == "flagged":
                        st.warning(f"Status: Flagged")

            with right_col:
                # Display images assigned to this question
                st.subheader("Images")
                assigned_images = get_images_for_question(q_id)

                if assigned_images:
                    for img in assigned_images:
                        filepath = img["filepath"]
                        if os.path.exists(filepath):
                            st.image(filepath, caption=f"{img['filename']} (page {img['page']})", use_container_width=True)

                            # Reassignment dropdown for this image
                            all_q_options = get_all_question_options()
                            current_idx = all_q_options.index(q_id) if q_id in all_q_options else 0

                            new_assignment = st.selectbox(
                                f"Reassign {img['filename']}:",
                                all_q_options,
                                index=current_idx,
                                key=f"reassign_{img['filename']}"
                            )

                            if new_assignment != q_id:
                                if st.button(f"Save reassignment", key=f"save_{img['filename']}"):
                                    if new_assignment == "(none)":
                                        st.session_state.image_assignments.pop(img["filename"], None)
                                    else:
                                        st.session_state.image_assignments[img["filename"]] = new_assignment
                                    save_image_assignments()
                                    st.success(f"Reassigned to {new_assignment}")
                                    st.rerun()
                        else:
                            st.warning(f"Image not found: {filepath}")
                elif q.get("has_image"):
                    st.info("This question is marked as having an image, but none assigned yet.")

                    # Show unassigned images from same chapter for easy assignment
                    st.markdown("**Unassigned images in this chapter:**")
                    ch_num = int(ch_key[2:])
                    ch_start = next((c["start_page"] for c in st.session_state.chapters if c["chapter_number"] == ch_num), 1)
                    ch_end = 9999
                    for i, c in enumerate(st.session_state.chapters):
                        if c["chapter_number"] == ch_num and i + 1 < len(st.session_state.chapters):
                            ch_end = st.session_state.chapters[i + 1]["start_page"]

                    unassigned = [img for img in st.session_state.images
                                  if img["filename"] not in st.session_state.image_assignments
                                  and ch_start <= img["page"] < ch_end]

                    if unassigned:
                        for img in unassigned[:5]:  # Show first 5
                            filepath = img["filepath"]
                            if os.path.exists(filepath):
                                st.image(filepath, caption=f"{img['filename']} (page {img['page']})", width=200)
                                if st.button(f"Assign to this question", key=f"assign_{img['filename']}"):
                                    st.session_state.image_assignments[img["filename"]] = q_id
                                    save_image_assignments()
                                    st.success("Assigned!")
                                    st.rerun()
                    else:
                        st.caption("No unassigned images in this chapter")
                else:
                    st.caption("No images assigned to this question")

            # QC actions
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if st.button("✓ Approve", type="primary"):
                    reviewed[q_id] = {"status": "approved", "timestamp": datetime.now().isoformat()}
                    st.session_state.qc_progress["reviewed"] = reviewed
                    save_qc_progress()
                    st.success("Approved!")
                    st.rerun()

            with col2:
                if st.button("✗ Flag Issue"):
                    reviewed[q_id] = {"status": "flagged", "timestamp": datetime.now().isoformat()}
                    st.session_state.qc_progress["reviewed"] = reviewed
                    save_qc_progress()
                    st.warning("Flagged for review")
                    st.rerun()

            with col3:
                if st.button("Skip"):
                    st.rerun()

            with col4:
                # Navigate to next unreviewed
                next_unreviewed = None
                for i, (_, nq) in enumerate(filtered_questions):
                    if nq["full_id"] not in reviewed and i > selected_idx:
                        next_unreviewed = i
                        break
                if next_unreviewed:
                    st.caption(f"Next unreviewed: #{next_unreviewed + 1}")


def render_export_step():
    """Render export step."""
    st.header("Step 5: Export to Anki")

    if not st.session_state.questions:
        st.warning("Please extract questions first (Step 3)")
        return

    # Count questions
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


# =============================================================================
# Main
# =============================================================================

def main():
    init_session_state()
    render_sidebar()

    step = st.session_state.current_step

    if step == "source":
        render_source_step()
    elif step == "chapters":
        render_chapters_step()
    elif step == "questions":
        render_questions_step()
    elif step == "qc":
        render_qc_step()
    elif step == "export":
        render_export_step()


if __name__ == "__main__":
    main()
